# test_otrscreen_kde.py

import torch
import matplotlib.pyplot as plt

from cheetah.particles import ParticleBeam
from cheetah.accelerator.screen_otr import OTRScreen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
T = lambda value: torch.tensor(value, device=device, dtype=dtype)

N = 50_000

beam = ParticleBeam.from_parameters(
    num_particles=N,
    energy=T(100e6),
    sigma_x=T(10e-6),
    sigma_y=T(15e-6),
    sigma_tau=T(8e-6),
    total_charge=T(1e-12),
    device=device,
    dtype=dtype,
)

screen = OTRScreen(
    resolution=(96, 96),
    pixel_size=torch.tensor(
        [120e-6 / 96, 150e-6 / 96],
        device=device,
        dtype=dtype,
    ),
    kde_bandwidth=T(4e-6),
    is_active=True,
    z_size=100.0,      # um, gives z range [-50, 50] um
    z_res=1.0,         # pixels / um, gives 101 z slices
    wavelengths=torch.tensor([0.6], device=device, dtype=dtype),
    svf_params={
        "gamma": 300 / 0.511,
        "theta_max": 0.28,
        "res": 1,
        "size": 40,
        "z_gauss_size": 0.08,
        "prefactor_x": 1.0,
        "prefactor_y": 1.0,
    },
    N_e=1.0,
    otr_mode="mixed",
    device=device,
    dtype=dtype,
)

screen.track(beam)

otr_stack = screen.reading
dist3d = screen.distribution

rho = dist3d.detach()
rho = rho / rho.sum()

bins_x = screen.pixel_bin_centers[0].detach()
bins_y = screen.pixel_bin_centers[1].detach()
bins_tau = screen.z_bins.detach()

print("dist3d shape:", dist3d.shape)
print("dist3d sum:", dist3d.sum().item())
print("otr_stack shape:", otr_stack.shape)
print("otr_stack min/max:", otr_stack.min().item(), otr_stack.max().item())

xy = rho.sum(dim=2)
xtau = rho.sum(dim=1)
ytau = rho.sum(dim=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(
    xy.cpu().T,
    origin="lower",
    extent=[
        bins_x[0].item() * 1e6,
        bins_x[-1].item() * 1e6,
        bins_y[0].item() * 1e6,
        bins_y[-1].item() * 1e6,
    ],
    aspect="auto",
)
axes[0].set_title("OTRScreen XY density")
axes[0].set_xlabel(r"x [$\mu$m]")
axes[0].set_ylabel(r"y [$\mu$m]")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    xtau.cpu().T,
    origin="lower",
    extent=[
        bins_x[0].item() * 1e6,
        bins_x[-1].item() * 1e6,
        bins_tau[0].item() * 1e6,
        bins_tau[-1].item() * 1e6,
    ],
    aspect="auto",
)
axes[1].set_title(r"OTRScreen X-$\tau$ density")
axes[1].set_xlabel(r"x [$\mu$m]")
axes[1].set_ylabel(r"$\tau$ [$\mu$m]")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    ytau.cpu().T,
    origin="lower",
    extent=[
        bins_y[0].item() * 1e6,
        bins_y[-1].item() * 1e6,
        bins_tau[0].item() * 1e6,
        bins_tau[-1].item() * 1e6,
    ],
    aspect="auto",
)
axes[2].set_title(r"OTRScreen Y-$\tau$ density")
axes[2].set_xlabel(r"y [$\mu$m]")
axes[2].set_ylabel(r"$\tau$ [$\mu$m]")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()