import torch
import math
import torch.nn.functional as F


class OTRGenerator:
    """
    Generates optical transition radiation (OTR) intensity stacks from
    a 3D charge distribution using both coherent and incoherent models.

    Supports batched inputs of shape (..., H, W, N_z) where N_z is
    the number of longitudinal slices, and produces outputs of shape
    (..., M, H, W) for M distinct wavelengths.

    Convention
    ----------
    The input dist is treated as a probability-mass distribution, not a
    continuous density sample. For example,

        dist.sum(dim=(-3, -2, -1)) = 1

    for each batch item.

    Therefore, the longitudinal projection does not divide by z_res or dz.
    """
    
    def __init__(self, 
                 wavelengths: torch.Tensor, 
                 SVFs: torch.Tensor, 
                 N_e: float | torch.Tensor = 1.0,
                 z_bins_um: torch.Tensor | None = None,
                 z_res: float | torch.Tensor | None = None,
                ):
        """
        Args:
            wavelengths: Wavelengths in um. A 1D tensor of length M 
                containing the wavelengths at which the field will be computed 
                (units: same as z_res).
                Preferred shape is (M,), where M is the number of wavelengths and
                SVFs has shape (3 * M, H, W). Channels are grouped as
                    [E_x(lambda_0), E_y(lambda_0), |E(lambda_0)|^2,
                     E_x(lambda_1), E_y(lambda_1), |E(lambda_1)|^2,
                     ...]
                For backward compatibility, wavelengths may also have shape
                (3 * M,), with one wavelength value per SVF channel.
                
            SVFs: Complex tensor of shape (3 * M, H, W) of single voxel
                functions (SVFs). Channels are E_x, E_y, and |E|^2 for each wavelength.

            N_e: Total number of electrons in the bunch. Default is 1.0.

            z_bins_um: Optional explicit longitudinal bin centers in um with
                shape (N_z,). This is the preferred interface.
            
            z_res: Optional backward-compatible longitudinal sampling 
                rate in pix / um. If z_bins_um is not provided, the phase is computed
                using z_n = n / z_res.

        Raises:
            ValueError: If SVFs and wavelengths are incompatible, or if neither
                z_bins_um nor z_res is provided.
        """
        if SVFs.ndim != 3:
            raise ValueError(f"Expected SVFs with shape (3*M,H,W), got {SVFs.shape}.")
        
        C = SVFs.shape[0]
        if C % 3 != 0: # 3 correspond to SVF_hor, SVF_ver, SVF_IOTR
            raise ValueError(f"Expected SVFs first dimension to be divisible by 3, got {C}.")

        self.SVFs = SVFs
        self.num_wls = C // 3
        device = SVFs.device
        real_dtype = SVFs.real.dtype

        wavelengths = wavelengths.to(device=device, dtype=real_dtype)
        M = wavelengths.numel()
        # Determine if user passed unique wavelengths (M == C // 3) or full (M == C)
        if M == self.num_wls: 
            # preferred case: one wavelength per physical wavelength.
            wl_full = wavelengths.repeat_interleave(3)
        elif M == C:
            # backward-compatible case: one wavelength per SVF channel
            wl_full = wavelengths
            # each triplet should correspond to the same physical wavelength
            if not (
                torch.allclose(wl_full[0::3], wl_full[1::3])
                and torch.allclose(wl_full[0::3], wl_full[2::3])
            ):
                raise ValueError("When wavelengths has shape (3*M,), each SVF triplet must "
                    "contain identical wavelength values."
                )
        else:
            raise ValueError(f"Wavelengths length ({M}) incompatible with SVFs first dim ({C})")
        # Full wavelength tensor, one entry per SVF channel
        # shape: (3 * num_wls,)
        # used for the coherent phase because COTR1 creates one complex 2D field per SVF channel
        self.wavelengths = wl_full
        self.N_e = torch.as_tensor(N_e, device=device, dtype=real_dtype)
        
        # check z_bins_um and z_res
        if z_bins_um is None and z_res is None:
            raise ValueError("Either z_bins_um or z_res must be provided.")
        if z_bins_um is not None:
            self.z_bins_um = z_bins_um.to(device=device, dtype=real_dtype)
        else:
            self.z_bins_um = None
        if z_res is not None:
            self.z_res = torch.as_tensor(z_res, device=device, dtype=real_dtype)
            if torch.any(self.z_res <= 0):
                raise ValueError("z_res must be positive.")
        else:
            self.z_res = None
        
        
        # Backward-compatible phase step. Only used when z_bins_um is
        # not provided.
        if self.z_bins_um is None:
            # phase factors per SVF channel: shape (3*M, )
            self.delta_phase = torch.exp(-1j * 2 * math.pi / self.wavelengths / self.z_res)
        else:
            self.delta_phase = None

    
    def _longitudinal_phase(self, N_z: int) -> torch.Tensor:
        """
        Return the longitudinal phase matrix with shape (3*M, N_z).

        Current convention
        ------------------
        The longitudinal grid is assumed to be uniform.

        If z_bins_um is provided, only its spacing dz is used. The absolute
        z-origin is ignored because it contributes only a wavelength-dependent
        global phase to the coherent field.

        If z_bins_um is not provided, use the backward-compatible convention
            z_n = n / z_res.
        """
        n = torch.arange(N_z, device=self.SVFs.device)

        if self.z_bins_um is not None:
            if self.z_bins_um.numel() != N_z:
                raise ValueError(
                    "z_bins_um length must match the last dimension of dist. "
                    f"Got z_bins_um length {self.z_bins_um.numel()} and N_z {N_z}."
                )

            if N_z == 1:
                return torch.ones(
                    (self.wavelengths.numel(), 1),
                    device=self.SVFs.device,
                    dtype=self.SVFs.dtype,
                )

            # Uniform-grid convention. We use the bin spacing, not the absolute
            # coordinate values. This reproduces the old z_res phase convention
            # while allowing OTRScreen/GPSR code to pass explicit z-bin centers.
            dz_um = (self.z_bins_um[-1] - self.z_bins_um[0]) / (N_z - 1)

            delta_phase = torch.exp(
                -1j * 2 * math.pi * dz_um / self.wavelengths
            )

            return delta_phase[:, None] ** n[None, :]

        return self.delta_phase[:, None] ** n[None, :]


    @staticmethod
    def _conv2d_same(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
        """
        Grouped 2D convolution with output spatial size equal to input spatial size.
    
        This works for both odd and even kernel sizes.
        """
        k_h, k_w = weight.shape[-2:]
    
        pad_h = k_h - 1
        pad_w = k_w - 1
    
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return F.conv2d(x, weight, groups=groups)
        
    
    def _get_COTR1(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Generates the effective 2D complex charge distribution and convolves with the 
        single voxel functions (SVFs). The result has S and P polarizations separated.
        First step that transforms the 3D charge distribution into a 2D complex phase
        distribution.

        Args:
            dist (torch.Tensor): A complex tensor of shape (..., H, W, N), representing 
                the spatial charge distribution over a 2D grid (H, W) and N longitudinal
                slices.

        Returns:
            torch.Tensor: A complex tensor of shape (..., 3*M, H, W). Each slice along 
                the first dimension corresponds to the projected complex field for a
                wavelength.
        """
        if dist.ndim < 3:
            raise ValueError(f"Expected dist with shape (..., H, W, N_z), got {dist.shape}.")

        dist_c = dist.to(device=self.SVFs.device, dtype=self.SVFs.dtype)
        phase = self._longitudinal_phase(dist_c.shape[-1]).to(dtype=self.SVFs.dtype)

        # Since dist is probability mass, do not divide by z_res or dz
        return torch.einsum("cn,...hwn->...chw", phase, dist_c)
    
    def _get_COTR2(self, field2d: torch.Tensor) -> torch.Tensor:
        """
        Performs the second step for COTR generation.
        Takes the 2D phase distribution and convolves it with the SVFs.

        Args:
            field2d (torch.Tensor): Complex tensor of shape (..., 3*M, H, W).

        Returns:
            torch.Tensor: Real tensor of shape (..., 2*M, H, W) containing intensity per
            polarization.
        """
        # SVFs, field2d: (..., C, H, W) complex
        C, H, W = self.SVFs.shape
        assert field2d.shape[-3] == C, \
            f"_get_COTR2 expected {C} channels, got {field2d.shape[-3]}"

        # mask out only E_x and E_y channels (i.e. ch % 3 != 2)
        idx  = torch.arange(C, device=self.SVFs.device)
        mask = (idx % 3) != 2

        # pick only those channels from both the kernels and the field
        SVFs_COTR = self.SVFs[mask]                 # (2*M, H, W)
        field_xy     = field2d[..., mask, :, :]        # (..., 2*M, H, W)

        # get the batch‐shape *after* masking
        B = field_xy.shape[:-3]
        nC = SVFs_COTR.shape[0]             # == 2*M
        F_H, F_W = field_xy.shape[-2:]
        K_H, K_W = SVFs_COTR.shape[-2:]
        
        # prepare real & imaginary convolution kernels
        wr = SVFs_COTR.real.unsqueeze(1)            # (nC, 1, H, W)
        wi = SVFs_COTR.imag.unsqueeze(1)            # (nC, 1, H, W)

        # flatten all batch dims into one for conv2d
        flat = field_xy.reshape(-1, nC, F_H, F_W)
        fr, fi = flat.real, flat.imag
        # dr, dc = K_H//2, K_W//2

        # # perform the 4 real‐valued grouped convolutions
        # rp = F.conv2d(fr, wr, groups=nC, padding=(dr,dc))
        # rp = rp - F.conv2d(fi, wi, groups=nC, padding=(dr,dc))
        # ip = F.conv2d(fr, wi, groups=nC, padding=(dr,dc))
        # ip = ip + F.conv2d(fi, wr, groups=nC, padding=(dr,dc))

        rp = self._conv2d_same(fr, wr, groups=nC)
        rp = rp - self._conv2d_same(fi, wi, groups=nC)
        
        ip = self._conv2d_same(fr, wi, groups=nC)
        ip = ip + self._conv2d_same(fi, wr, groups=nC)
        
        out = (rp + 1j*ip).abs()**2  # (B_flat, nC, F_H, F_W)
        
        # unflatten back into the original batch dims
        return out.view(*B, nC, F_H, F_W)

    def _add_sp(self, cotr: torch.Tensor) -> torch.Tensor:
        """
        Adds the S and P polarizations for each wavelength and stores them in COTRs.
        
        Args:
            cotr: Real tensor of shape (C, H, W) .
        
        Returns:
            Real tensor of shape (C, H, W) .
        """
        
        evens = cotr[..., 0::2, :, :]
        odds = cotr[..., 1::2, :, :]
        evens[..., :odds.shape[-3], :, :] += odds
        return evens

    def _get_IOTR(self, dist):
        """
        Generates the 2D charge distribution and convolves with the incoherent single voxel functions (SVFs).
        The result has S and P polarizations combined.
        
        Args:
            dist (torch.Tensor): A complex tensor of shape (H, W, N), representing the 
                spatial charge distribution over a 2D grid (H, W) and N longitudinal 
                slices.
            wavelengths (torch.Tensor): A 1D tensor of shape (C,) containing the 
            wavelengths
                at which the field will be computed (in the same units as z_res).
            z_res (float): Longitudinal spatial resolution (delta_z) between charge distribution slices.

        Returns:
            torch.Tensor: A real tensor of shape (C, H, W), where each slice along the 
                first dimension corresponds to the projected complex field for a wavelength.
        """
        # collapse the longitudinal dimension and get screen dims
        # dens2d = torch.abs(torch.sum(dist, dim=-1)) / self.z_res  # (..., H_screen, W_screen)
        dens2d = torch.abs(torch.sum(dist, dim=-1)) # note by Ritz: check with Max
        B_shape, H_screen, W_screen = dens2d.shape[:-2], *dens2d.shape[-2:]
        
        # pick only the intensity (|E|^2) SVF channels
        C = self.SVFs.shape[0]
        idx = torch.arange(C, device=self.SVFs.device)
        SVF_IOTR = self.SVFs[idx % 3 == 2]  # (M, H_kernel, W_kernel)
        M, H_kernel, W_kernel = SVF_IOTR.shape
        
        # prepare real‐valued convolution kernels
        wr = SVF_IOTR.real.unsqueeze(1)  # (M, 1, H_kernel, W_kernel)
        
        # replicate the 2D density across M channels at screen resolution
        dens_ch = dens2d.unsqueeze(-3).expand(*B_shape, M, H_screen, W_screen)
        
        # flatten all batch dims for grouped conv2d
        flat = dens_ch.reshape(-1, M, H_screen, W_screen)
        # dr, dc = H_kernel // 2, W_kernel // 2
        # out = F.conv2d(flat, wr, groups=M, padding=(dr, dc))  # (B_flat, M, H_screen, W_screen)
        # # restore original batch dims
        # return out.view(*B_shape, M, H_screen, W_screen)

        out = self._conv2d_same(flat, wr, groups=M)
        # restore original batch dims
        return out.view(*B_shape, M, H_screen, W_screen)

    def forward(self, dist: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Generates OTR intensity images for a provided 3D charge distribution.

        Args:
          dist (torch.Tensor): A complex tensor of shape (..., H, W, N) where 
              N is the number of samples.
          mode (str): Mode of optical transition radiation: 
              'coherent', 'incoherent', or 'mixed'.

        Returns:
          OTRs:   A real tensor of shape (..., M, H, W) 
              representing the summed S and P polarizations.
        """
        if mode not in {"coherent", "incoherent", "mixed"}:
            raise ValueError(
                "mode must be one of 'coherent', 'incoherent', or 'mixed'. "
                f"Got {mode!r}."
            )
        
        # # cotr
        # dist2d = self._get_COTR1(dist) # project 3D charge distribution into 2D fields
        # cotr_per_e = self._get_COTR2(dist2d) # convolve with SVFs and compute intensity
        # cotr_image = self._add_sp(cotr_per_e) # sum polarizations
        # # iotr
        # iotr_image = self._get_IOTR(dist)
        
        # if mode == "incoherent":
        #     images = iotr_image * self.N_e
        # elif mode == "coherent":
        #     images = cotr_image * (self.N_e - 1) * self.N_e
        # elif mode == "mixed":
        #     images = iotr_image * self.N_e + cotr_image * (self.N_e - 1) * self.N_e
        # # flip axes to match orientation
        # return torch.flip(images, dims=[-2, -1])

        if mode == "incoherent":
            images = self._get_IOTR(dist) * self.N_e
    
        elif mode == "coherent":
            dist2d = self._get_COTR1(dist)
            cotr_per_e = self._get_COTR2(dist2d)
            cotr_image = self._add_sp(cotr_per_e)
            images = cotr_image * (self.N_e - 1) * self.N_e
    
        else: # mixed: coherent + incoherent
            dist2d = self._get_COTR1(dist)
            cotr_per_e = self._get_COTR2(dist2d)
            cotr_image = self._add_sp(cotr_per_e)
            iotr_image = self._get_IOTR(dist)
            images = iotr_image * self.N_e + cotr_image * (self.N_e - 1) * self.N_e
    
        return torch.flip(images, dims=[-2, -1])
