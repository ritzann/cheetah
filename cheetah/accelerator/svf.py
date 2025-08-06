import torch
import torch.nn.functional as F
import math
import numpy as np

class SVFGenerator:
    """
    Encapsulates Single Voxel Function (SVF) utilities, including theoretical
    kernels, resizing routines, and z-thickness corrections.
    """
    def __init__(self, gamma: float, theta_max: float, res: int, size: int,
                 z_gauss_size: float, prefactor_x: float, prefactor_y: float,
                 device: torch.device = None):
        """
        Args:
            gamma (float): Lorentz factor for relativistic correction.
            theta_max (float): Maximum collection angle.
            res (int): Pixel resolution multiplier for intermediate upsampling.
            size (int): Base image size (pixels) before upsampling.
            z_gauss_size (float): Gaussian voxel thickness for z-correction.
            prefactor_x (float): Prefactor for horizontal component scaling.
            prefactor_y (float): Prefactor for vertical component scaling.
            device (torch.device, optional): Computation device (cuda or cpu).
        """
        self.gamma = gamma
        self.theta_max = theta_max
        self.res = res
        self.size = size
        self.z_gauss_size = z_gauss_size
        self.prefactor_x = prefactor_x
        self.prefactor_y = prefactor_y
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # prepare grid
        self.grid_res = 10 * self.size * self.res + 1
        coords = torch.linspace(-self.size/2, self.size/2, steps=self.grid_res, device=self.device)
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        R = torch.sqrt(X**2 + Y**2 + 1e-12)
        self.R = R
        self.cos_t, self.sin_t = X/R, Y/R
        self.target = self.res * self.size + 1 # final target resolution
    
    
    def perfect_SPF(self, r: torch.Tensor, k: float) -> torch.Tensor:
        """
        Generates the theoretical single particle function (SPF) for a
        point charge with location r, at wavenumber k, relativistic
        coefficent gamma, and imaging system defined by theta_max.

        Args:
            r (torch.Tensor): Radial distances.
            k (float): Wavenumber.
        Returns:
            torch.Tensor: SPF values.
        """
        r = r.to(device=self.device, dtype=torch.float64) + 1e-12
        term1 = torch.special.modified_bessel_k1(k * r / self.gamma) / self.gamma
        term2 = torch.special.bessel_j0(self.theta_max * k * r) / (k * r)
        return term1 - term2

    
    @staticmethod
    def resize2D(array: torch.Tensor, pixel_num: int, mode: str = 'bicubic') -> torch.Tensor:
        """
        Resizes a 2D tensor preserving total sum. Default interpolation is order 3
        (bicubic).
        
        Args:
            array (torch.Tensor or np.ndarray): Input 2D array to resize.
            pixel_num (int): Target resolution (square: pixel_num × pixel_num).
            mode (str): Interpolation mode. Options include 'bilinear', 'bicubic',
            etc.
        
        Returns:
            torch.Tensor: Resized 2D tensor with preserved total sum.
        """
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        array = array.to(dtype=torch.float32)
        x = array.unsqueeze(0).unsqueeze(0)
        kwargs = {"size":(pixel_num, pixel_num), "mode":mode}
        if mode in ['bilinear','bicubic']:
            kwargs.update({"align_corners":False, "antialias":True})
        resized = F.interpolate(x, **kwargs).squeeze()
        return (array.sum() / resized.sum()) * resized

    
    @staticmethod
    def resize2D_SVF(array: torch.Tensor, pixel_num: int, mode: str = 'bicubic') -> torch.Tensor:
        """
        Resizes a 2D tensor preserving the sum of squares. Default interpolation is
        order 3 (bicubic).
        
        Args:
            array (torch.Tensor or np.ndarray): Input 2D array to resize.
            pixel_num (int): Target resolution (square: pixel_num × pixel_num).
            mode (str): Interpolation mode. Options include 'bilinear', 'bicubic',
            etc.
            
        Returns:
            torch.Tensor: Resized 2D tensor with preserved energy (L2 norm squared).
        """
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        array = array.to(dtype=torch.float32)
        x = array.unsqueeze(0).unsqueeze(0)
        kwargs = {"size":(pixel_num, pixel_num), "mode":mode}
        if mode in ['bilinear','bicubic']:
            kwargs.update({"align_corners":False, "antialias":True})
        resized = F.interpolate(x, **kwargs).squeeze()
        orig = array.square().sum()
        new = resized.square().sum()
        return resized * torch.sqrt(orig / new)

    
    def z_coeff(self, wl: float) -> torch.Tensor:
        """
        Computes the longitudinal (z-axis) Gaussian correction factor for a given
        wavelength.

        This applies a thickness-dependent exponential decay factor to account for
        the finite longitudinal extent of the charge distribution.
        
        Args:
            wl (float): Wavelength in microns.

        Returns:
            torch.Tensor: Complex scalar (complex64) representing the z-thickness
            correction.
        """
        wl_t = torch.as_tensor(wl, dtype=torch.float64, device=self.device)
        z_t = torch.as_tensor(self.z_gauss_size, dtype=torch.float64, device=self.device)
        return torch.exp(-0.5 * (2 * math.pi / wl_t)**2 * z_t**2).to(torch.complex64) # equation to be corrected

    
    def forward(self, wl: float) -> torch.Tensor:
        """
        Generates the horizontal and vertical SVFs for a given wavelength.
        This computes the horizontal and vertical SVFs based on the perfect 
        single particle function (SPF), upsamples the components by 10× to reduce
        aliasing, rescales them to the target resolution while preserving energy,
        and applies a z-thickness correction. The third channel represents the
        incoherent intensity SVF (IOTR) computed from the squared field 
        components.

        Args:
            wl (float): Wavelength in microns.

        Returns:
            torch.Tensor: A tensor of shape (3, H, W), dtype complex64, where:
                - [0] is horizontal component (E_x)
                - [1] is vertical component (E_y)
                - [2] is IOTR = |E_x|^2 + |E_y|^2
        """
        k = 2 * math.pi / wl

        # compute SPF
        spf = self.perfect_SPF(self.R, k).float()

        # high-res components
        hor = (self.cos_t * spf) / 10
        ver = (self.sin_t * spf) / 10

        # resize + z-correction
        target = self.res * self.size + 1
        
        SVF_hor = (self.prefactor_x * self.resize2D_SVF(hor, self.target))
        SVF_ver = (self.prefactor_y * self.resize2D_SVF(ver, self.target))
        SVF_IOTR = SVF_hor ** 2 + SVF_ver ** 2
        print(SVF_IOTR.shape)
        SVF_hor = SVF_hor * self.z_coeff(wl)
        SVF_ver = SVF_ver * self.z_coeff(wl)
        SVFs = torch.stack([SVF_hor, SVF_ver, SVF_IOTR]).to(torch.complex64)
        print(SVFs.shape)

        return SVFs
        