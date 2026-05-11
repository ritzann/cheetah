# test_otrscreen_modes.py

import torch
import matplotlib.pyplot as plt

from cheetah.particles import ParticleBeam
from cheetah.accelerator.screen_otr import OTRScreen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
T = lambda value: torch.tensor(value, device=device, dtype=dtype)

# ------------------------------------------------------------
# 1. Controlled Gaussian beam
# ------------------------------------------------------------
N_particles = 50_000
N_e = 1_000.0

beam = ParticleBeam.from_parameters(
    num_particles=N_particles,
    energy=T(100e6),
    sigma_x=T(10e-6),
    sigma_y=T(15e-6),
    sigma_tau=T(8e-6),
    total_charge=T(1e-12),
    device=device,
    dtype=dtype,
)

print("Beam statistics:")
print("x std   [um]:", beam.x.std().item() * 1e6)
print("y std   [um]:", beam.y.std().item() * 1e6)
print("tau std [um]:", beam.tau.std().item() * 1e6)
print()

# ------------------------------------------------------------
# 2. Common screen settings
# ------------------------------------------------------------
wavelengths = torch.tensor([0.4, 0.6, 0.8], device=device, dtype=dtype)

screen_kwargs = dict(
    resolution=(96, 96),
    pixel_size=torch.tensor(
        [120e-6 / 96, 150e-6 / 96],
        device=device,
        dtype=dtype,
    ),
    kde_bandwidth=T(4e-6),
    is_active=True,
    z_size=100.0,    # um
    z_res=1.0,       # pix / um
    wavelengths=wavelengths,
    svf_params={
        "gamma": 300 / 0.511,
        "theta_max": 0.28,
        "res": 1,
        "size": 40,
        "z_gauss_size": 0.08,
        "prefactor_x": 1.0,
        "prefactor_y": 1.0,
    },
    N_e=N_e,
    device=device,
    dtype=dtype,
)

# ------------------------------------------------------------
# 3. Run modes
# ------------------------------------------------------------
results = {}

for mode in ["incoherent", "coherent", "mixed"]:
    print(f"\nRunning mode: {mode}")

    screen = OTRScreen(
        **screen_kwargs,
        otr_mode=mode,
    )

    screen.track(beam)
    image_stack = screen.reading
    dist3d = screen.distribution

    print("dist3d shape:", tuple(dist3d.shape))
    print("dist3d sum:", dist3d.sum().item())
    print("image_stack shape:", tuple(image_stack.shape))
    print("image_stack dtype:", image_stack.dtype)
    print("image_stack min/max:", image_stack.min().item(), image_stack.max().item())
    print("has nan:", torch.isnan(image_stack).any().item())
    print("has inf:", torch.isinf(image_stack).any().item())

    results[mode] = image_stack.detach().cpu()

# ------------------------------------------------------------
# 4. Plot image stacks
# ------------------------------------------------------------
num_modes = len(results)
num_wls = wavelengths.numel()

fig, axes = plt.subplots(num_modes, num_wls, figsize=(4 * num_wls, 4 * num_modes))

if num_modes == 1:
    axes = axes[None, :]
if num_wls == 1:
    axes = axes[:, None]

for row, mode in enumerate(["incoherent", "coherent", "mixed"]):
    stack = results[mode]

    for col, wl in enumerate(wavelengths.detach().cpu()):
        ax = axes[row, col]
        image = stack[col]

        im = ax.imshow(image, origin="lower")
        ax.set_title(fr"{mode}, $\lambda$={wl.item():.1f} $\mu$m")
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Simple consistency check
# ------------------------------------------------------------
incoh = results["incoherent"]
coh = results["coherent"]
mixed = results["mixed"]

residual = (mixed - incoh - coh).abs()
print("\nConsistency check:")
print("max |mixed - incoherent - coherent|:", residual.max().item())
print("relative residual:", residual.max().item() / mixed.abs().max().item())
