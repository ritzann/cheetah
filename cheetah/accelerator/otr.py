import torch
import math
import torch.nn.functional as F


class OTRGenerator:
    """
    Generates optical transition radiation (OTR) intensity stacks from
    a 3D charge distribution using both coherent and incoherent models.

    Supports batched inputs of shape (..., H, W, N) where N is
    the number of longitudinal slices, and produces outputs of shape
    (..., M, H, W) for M distinct wavelengths.
    """
    def __init__(self, 
                 wavelengths: torch.Tensor, 
                 SVFs: torch.Tensor, 
                 z_res: float, 
                 N_e: float):
        """
        Args:
            wavelengths (torch.Tensor): A 1D tensor of length M containing the
                wavelengths at which the field will be computed (units: same as
                z_res).
            SVFs (torch.Tensor): Complex tensor of shape (3*M, H, W) of single voxel
            functions (SVFs). Channels are [E_x, E_y, |E|^2] for each wavelength.
            z_res (float): Longitudinal spatial resolution (delta_z) between charge
                distribution slices.
            N_e (float): Total number of electrons in the bunch.
        Raises:
            ValueError: If SVFs first dimension is not compatible with wavelengths.
        """
        C = SVFs.shape[0]
        M = wavelengths.numel()
        # determine if user passed unique wavelengths (M*3 == C) or full (M == C)
        if M * 3 == C: # 3 correspond to SVF_hor, SVF_ver, SVF_IOTR
            wl_full = wavelengths.repeat_interleave(3)
        elif M == C:
            wl_full = wavelengths
        else:
            raise ValueError(f"Wavelengths length ({M}) incompatible with SVFs first dim ({C})")
        self.wavelengths = wl_full.to(dtype=SVFs.dtype, device=SVFs.device)
        self.SVFs = SVFs
        self.z_res = z_res
        self.N_e = N_e
        self.num_wls = M
        # phase factors per SVF channel: shape (3*M, )
        self.delta_phase = torch.exp(-1j * 2 * math.pi / self.wavelengths / self.z_res)
        
    
    def _get_COTR1(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Generates the effective 2D charge distribution and convolves with the 
        single voxel functions (SVFs). The result has S and P polarizations separated.
        First step that transforms the 3D charge distribution into a 2D complex phase
        distribution.

        Args:
            dist (torch.Tensor): A complex tensor of shape (..., H, W, N), representing the 
                spatial charge distribution over a 2D grid (H, W) and N longitudinal
                slices.

        Returns:
            torch.Tensor: A complex tensor of shape (..., C, H, W), where C=3*M and each 
            slice along the first dimension corresponds to the projected complex field for a
            wavelength.
        """
        # dist:   (H, W, N) complex64; wavelengths: (C,) float32
        assert dist.ndim >= 3, f"Expected dist (...,H,W,N), got {dist.shape}"
        dist_c = dist.to(self.delta_phase.dtype)  # cast float to complex64
        n = torch.arange(dist_c.shape[-1], device=self.delta_phase.device)
        delta_phase_pows = self.delta_phase[:, None] ** n[None, :]        # (3*M, N)
        # sum over longitudinal slices
        return torch.einsum('cn,...hwn->...chw', delta_phase_pows, dist_c / self.z_res)

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
#         # SVFs, field2d: (..., C, H, W) complex
#         C, H, W = self.SVFs.shape
#         assert field2d.shape[-3] == C, \
#             f"_get_COTR2 expected {C} channels, got {field2d.shape[-3]}"
        
#         idx = torch.arange(C, device=self.SVFs.device)
#         mask = (idx % 3) != 2
        
#         SVFs_COTR = self.SVFs[mask]  # (2*M, H, W)
#         print("SVFs_COTR:",SVFs_COTR.shape)
#         field     = field2d[..., mask, :, :] # (..., 2*M, H, W)
        
#         B = field.shape[:-3] # batch dimension
#         nC = SVFs_COTR.shape[0] # = 2*M
        
#         # build real & imaginary kernels of shape (nC, 1, H, W)
#         wr = SVFs_COTR.real.unsqueeze(1)
#         wi = SVFs_COTR.imag.unsqueeze(1)
        
#         # flatten batch dims for grouped conv
#         flat_field = field.reshape(-1, nC, H, W)
#         fr, fi = flat_field.real, flat_field.imag
        
#         # convolution padding to keep the same size
#         dr, dc = H//2, W//2

#         # **ALL** conv2d calls get the same padding=(dr,dc)
#         rp = F.conv2d(fr, wr, groups=nC, padding=(dr,dc))
#         rp = rp - F.conv2d(fi, wi, groups=nC, padding=(dr,dc))
#         ip = F.conv2d(fr, wi, groups=nC, padding=(dr,dc))
#         ip = ip + F.conv2d(fi, wr, groups=nC, padding=(dr,dc))
#         out = (rp + 1j*ip).abs()**2  # shape (B_flat, nC, H, W)
#         return out.view(*B, nC, H, W) # restore batch dims
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
        dr, dc = K_H//2, K_W//2

        # perform the 4 real‐valued grouped convolutions
        rp = F.conv2d(fr, wr, groups=nC, padding=(dr,dc))
        rp = rp - F.conv2d(fi, wi, groups=nC, padding=(dr,dc))
        ip = F.conv2d(fr, wi, groups=nC, padding=(dr,dc))
        ip = ip + F.conv2d(fi, wr, groups=nC, padding=(dr,dc))

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
#         # dist:   (H, W, N) complex64; wavelengths: (C,) float32
#         # sum over longitudinal slices and divide by thickness z_res (pix/um) to get 2D density
#         # dens2d = torch.abs(torch.sum(dist, 2)) / self.z_res # (H, W)
#         dens2d = torch.abs(torch.sum(dist, dim=-1)) / self.z_res # (H, W)
#         # print("dens2d size:",dist2d.shape)
#         B = dens2d.shape[:-2]
        
#         C, H, W = self.SVFs.shape
#         # SVFs, dens2d: (C, H, W) complex
#         # pick out only the incoherent SVFs
#         SVF_IOTR = self.SVFs[torch.arange(C, device=self.SVFs.device) % 3 == 2] # (M, H, W)
#         # print("SVF_IOTR.shape:",SVF_IOTR.shape)
        
#         # prepare convolution kernels: (M, 1, H, W)
#         wr = SVF_IOTR.real.unsqueeze(1)
#         dens_ch = dens2d.unsqueeze(-3).expand(*B, self.num_wls, H, W)
#         # flatten batch dims: (B_flat, M, H, W)
#         flat_dens = dens_ch.reshape(-1, self.num_wls, H, W)
#         dr, dc = H//2, W//2
#         # perform grouped convolution: each channel with its kernel
#         out = F.conv2d(flat_dens, wr, groups=self.num_wls, padding=(dr, dc))
#         # reshape back: (..., M, H, W)
#         return out.view(*B, self.num_wls, H, W)

        # collapse the longitudinal dimension and get screen dims
        dens2d = torch.abs(torch.sum(dist, dim=-1)) / self.z_res  # (..., H_screen, W_screen)
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
        dr, dc = H_kernel // 2, W_kernel // 2
        out = F.conv2d(flat, wr, groups=M, padding=(dr, dc))  # (B_flat, M, H_screen, W_screen)
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
        # cotr
        dist2d = self._get_COTR1(dist) # project 3D charge distribution into 2D fields
        cotr_per_e = self._get_COTR2(dist2d) # convolve with SVFs and compute intensity
        cotr_image = self._add_sp(cotr_per_e) # sum polarizations
        # iotr
        iotr_image = self._get_IOTR(dist)
        
        if mode == "incoherent":
            images = iotr_image * self.N_e
        elif mode == "coherent":
            images = cotr_image * (self.N_e - 1) * self.N_e
        elif mode == "mixed":
            images = iotr_image * self.N_e + cotr_image * (self.N_e - 1) * self.N_e
        # flip axes to match orientation
        return torch.flip(images, dims=[-2, -1])
