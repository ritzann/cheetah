import torch
import torch.nn.functional as F
from typing import Literal
from cheetah.particles import ParticleBeam
from cheetah.utils import kde_histogram_3d, verify_device_and_dtype
from screen import Screen
from svf import SVFGenerator
from otr import OTRGenerator

class OTRScreen(Screen):
    """
    Screen that computes OTR images end-to-end: from ParticleBeam via 3D KDE to OTR.

    :param resolution: pixel resolution (w, h)
    :param pixel_size: pixel size (m)
    :param binning: spatial binning
    :param misalignment: misalignment offset
    :param kde_bandwidth: KDE smoothing width
    :param z_size: longitudinal extent (um)
    :param z_res: longitudinal sampling (pix/um)
    :param wavelengths: 1D tensor of wavelengths (um)
    :param svf_params: dict of params for SVFGenerator (gamma, theta_max, res, size, z_gauss_size, prefactor_x, prefactor_y)
    :param N_e: number of electrons in bunch
    :param cotr_mode: 'coherent','incoherent','mixed'
    """
    def __init__(
        self,
        resolution, 
        pixel_size=None,
        binning=1,
        misalignment=None,
        kde_bandwidth=None,
        is_blocking=False,
        is_active=False,
        name=None,
        sanitize_name=False,
        device=None,
        dtype=None,
        
        # OTR-specific args (from COTR initial example)
        # Note by Ritz: change to other default params?
        z_size: float = 50.0,       # um
        z_res: float = 17.0,        # pix/um
        wavelengths: torch.Tensor = None,  # um # replace the None; should be required argument
        svf_params: dict = None, # replace the None; should be required argument
        N_e: float = 1.0,
        cotr_mode: Literal['coherent','incoherent','mixed'] = 'mixed',
    ):
        # initialize base Screen with KDE method
        super().__init__(
            resolution=resolution,
            pixel_size=pixel_size,
            binning=binning,
            misalignment=misalignment,
            method='kde',
            kde_bandwidth=kde_bandwidth,
            is_blocking=is_blocking,
            is_active=is_active,
            name=name,
            sanitize_name=sanitize_name,
            device=device,
            dtype=dtype,
        )
        # register OTR parameters
        device = self.pixel_size.device
        dtype = self.pixel_size.dtype
        self.cotr_mode = cotr_mode
        self.N_e = N_e
        self.wavelengths = wavelengths.to(device=device, dtype=dtype)
        self.z_res = z_res
        # build longitudinal bins matching z_size and z_res
        Nz = int(z_size * z_res) + 1
        z_bins = torch.linspace(
            -z_size/2,
             z_size/2,
            steps=Nz,
            device=device,
            dtype=dtype,
        ) # Note by Ritz: how do we really define this?
        self.register_buffer('z_bins', z_bins)

        # instantiate SVFGenerator
        svf_kwargs = { 
            'gamma': svf_params['gamma'],
            'theta_max': svf_params['theta_max'],
            'res': svf_params['res'],
            'size': svf_params['size'],
            'z_gauss_size': svf_params['z_gauss_size'],
            'prefactor_x': svf_params.get('prefactor_x',1.0),
            'prefactor_y': svf_params.get('prefactor_y',1.0),
            'device': device,
        }
        self.svf_gen = SVFGenerator(**svf_kwargs)

        # generate SVFs for each wavelength
        svf_blocks = []
        for wl in self.wavelengths:
            svf = self.svf_gen.forward(wl.item())       # (3,H,W) complex64
            svf_blocks.append(svf)
        SVFs = torch.cat(svf_blocks, dim=0).to(torch.complex64)  # (3*M, H, W)

        # instantiate OTRGenerator
        self.otr = OTRGenerator(
            wavelengths=self.wavelengths,
            SVFs=SVFs,
            z_res=z_res,
            N_e=N_e,
        )

    @property
    def reading(self) -> torch.Tensor:
        # return cached if exists
        if not torch.isnan(self.cached_reading).all():
            return self.cached_reading

        rb = self.get_read_beam()
        if rb is None or not isinstance(rb, ParticleBeam):
            raise RuntimeError('Need an active ParticleBeam for OTRScreen.')

        # extract coords & weights
        x = rb.x
        y = rb.y
        z = rb.tau # tau = -ct (check sign convention)
        w = rb.particle_charges.abs() * rb.survival_probabilities

        # batch dim
        x_, y_, z_, w_ = [t.unsqueeze(0) for t in (x,y,z,w)]

        # 3D KDE: (1,H,W,Nz)
        hist3d = kde_histogram_3d(
            x1=x_, x2=y_, x3=z_,
            bins1=self.pixel_bin_centers[0],
            bins2=self.pixel_bin_centers[1],
            bins3=self.z_bins,
            bandwidth=self.kde_bandwidth,
            weights=w_,
        )
        charge3d = hist3d[0]

        # compute OTR images
        cotr_stack = self.otr.forward(charge3d, mode=self.cotr_mode) 

        self.cached_reading = cotr_stack # shape (N_lambda, H, W)
        return cotr_stack # include batching here

