import torch
from typing import Literal
from cheetah.particles import ParticleBeam
from cheetah.utils import kde_histogram_3d
from cheetah.accelerator.screen import Screen
from cheetah.accelerator.svf import SVFGenerator
from cheetah.accelerator.otr import OTRGenerator

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
    :param otr_mode: 'coherent','incoherent','mixed'
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
        wavelengths: torch.Tensor = torch.tensor([0.4, 0.6, 0.8]),  # um
        svf_params: dict = {  # default SVF parameters
            'gamma': 300/0.511,
            'theta_max': 0.28,
            'res': 1,
            'size': 40,
            'z_gauss_size': 0.08,
            'prefactor_x': 1.0,
            'prefactor_y': 1.0,
        },
        N_e: float = 1.0,
        otr_mode: Literal['coherent','incoherent','mixed'] = 'mixed',
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
        self.otr_mode = otr_mode
        self.N_e = N_e
        self.wavelengths = wavelengths.to(device=device, dtype=dtype)
        self.z_res = z_res
        self.cached_distribution = None
        # build longitudinal bins matching z_size and z_res
        Nz = int(z_size * z_res) + 1
        # z_bins = torch.linspace(
        #     -z_size/2,
        #      z_size/2,
        #     steps=Nz,
        #     device=device,
        #     dtype=dtype,
        # ) # Note by Ritz: how do we really define this?
        # z_size was given in microns, so convert to meters here
        z_bins_um = torch.linspace(-z_size/2, z_size/2, steps=Nz,
                                   device=device, dtype=dtype)  # in microns
        z_bins_m    = z_bins_um * 1e-6                          # now in meters
        self.register_buffer("z_bins", z_bins_m)
        self.register_buffer("z_bins_um", z_bins_um)

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
            z_bins_um=self.z_bins_um,
            N_e=N_e,
        )

    @property
    def distribution(self) -> torch.Tensor:
        """Return the most recently computed 3D charge distribution."""
        if self.cached_distribution is None or self.cached_reading is None:
            _ = self.reading
        return self.cached_distribution

    @property
    def reading(self) -> torch.Tensor:
        # return cached if exists
        if self.cached_reading is not None and not torch.isnan(self.cached_reading).all():
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

        # Use requested grid exactly. Do not silently downsample z, because
        # reducing z to only a few slices destroys the longitudinal projection.
        bins1, bins2, bins3 = (
            self.pixel_bin_centers[0],
            self.pixel_bin_centers[1],
            self.z_bins,
        )

        # 3D KDE: (1,H,W,Nz)
        hist3d = kde_histogram_3d(
            x1=x_, x2=y_, x3=z_,
            bins1=bins1,
            bins2=bins2,
            bins3=bins3,
            bandwidth=self.kde_bandwidth,
            weights=w_,
        )
        dist3d = hist3d[0]
        # Note by Ritz: plot projections of charge dist (create a plotting funtion here)
        # For now, keep dist3d as a normalized probability-mass distribution.
        # Physical electron-number scaling should be handled by N_e in OTRGenerator.
        # total_charge = w_.sum()

        # compute OTR images
        otr_stack = self.otr.forward(dist3d, mode=self.otr_mode) # plot IOTR and COTR outputs within otr generator

        self.cached_distribution = dist3d
        self.cached_reading = otr_stack # shape (N_lambda, H, W)
        return otr_stack

