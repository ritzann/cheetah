import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import Bbox

from cheetah.accelerator.svf import SVFGenerator
from cheetah.accelerator.otr import OTRGenerator


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

out_dir = Path("tmp_tests/cotr_custom_distribution_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

OTR_size_um = 41.0
theta_max = 0.28
gamma = 300 / 0.511

# increase this from 1 to 2 for smoother SVFs and images
# res = 1 gives ~41 x 41 SVFs
# res = 2 gives ~83 x 83 SVFs
res = 2 # pixels / um

epsilon_small = 1e-10

e_charge = 1.60217663e-19
measured_charge = 1e-10  # 100 pC
N_e = measured_charge / e_charge # note by ritz: keep but comment out for debugging
# N_e = 1000.0 # for visualization/debugging
print("N_e:",N_e)

# spatial grids in um
q_size_um = 40.0
q_res = res      # pixels / um
if OTR_size_um < q_size_um:
    raise ValueError("OTR_size_um must be larger than or equal to q_size_um.")
z_size_um = 50.0
z_res = 17.0      # pixels / um

z_gauss_size_um = 0.08
z_HWHM_um = z_gauss_size_um * 1.177

Nx = int(round(q_size_um * q_res)) + 1
Ny = int(round(q_size_um * q_res)) + 1
Nz = int(round(z_size_um * z_res)) + 1

x_um = torch.linspace(-q_size_um / 2, q_size_um / 2, Nx, device=device, dtype=dtype)
y_um = torch.linspace(-q_size_um / 2, q_size_um / 2, Ny, device=device, dtype=dtype)
z_um = torch.linspace(-z_size_um / 2, z_size_um / 2, Nz, device=device, dtype=dtype)

# mesh has shape (Ny, Nx, Nz) = (y, x, z).
y_grid, x_grid, z_grid = torch.meshgrid(y_um, x_um, z_um, indexing="ij")

print("grid shape:", x_grid.shape)

# wavelengths in um
wavelengths_um = torch.tensor([0.4, 0.8], device=device, dtype=dtype)

# ---------------------------------------------------------------------
# Custom distribution parameters (from NAPAC 2025)
# ---------------------------------------------------------------------
param_sets = [
    {
        "name": "single_curved",
        "A": 0.0,
        "B": 0.0,
        "C": 0.8,
        "D": 6.0,
        "E": 0.0,
        "Fp": 2.0,
        "G": 0.0,
        "H": 2.0,
        "I": 1.0,
        "J": 0.0,
        "L": -2.0,
        "M": 1.0,
        "N": 0.0,
        "P": 1.0,
        "Q": -1.0,
        "R": 0.2,
    },
    {
        "name": "curved_plus_spike",
        "A": 0.0,
        "B": 4.0,
        "C": 12.0,
        "D": 2.0,
        "E": 0.0,
        "Fp": 2.0,
        "G": 0.0,
        "H": 2.0,
        "I": 1.0,
        "J": 1.0,
        "L": -2.0,
        "M": 2.0,
        "N": 0.0,
        "P": 2.0,
        "Q": -1.0,
        "R": 0.2,
    },
]


def make_custom_dist(params: dict, sample_index: int) -> torch.Tensor:
    """
    Return unnormalized rho(y, x, z) for one custom distribution.
    """
    A = params["A"]
    B = params["B"]
    C = params["C"]
    D = params["D"]
    E = params["E"]
    Fp = params["Fp"]
    G = params["G"]
    H = params["H"]
    I = params["I"]
    J = params["J"]
    L = params["L"]
    M = params["M"]
    N = params["N"]
    P = params["P"]
    Q = params["Q"]
    R = params["R"]

    if sample_index == 1:
        main = I * torch.exp(
            -((x_grid - A - B * torch.sin(2 * math.pi / C * z_grid)) ** 2 / D**2)
            -((y_grid - E) ** 2 / Fp**2)
            -((z_grid - G) ** 2 / H**2)
        )

        spike = J * torch.exp(
            -((x_grid - L) ** 2 / M**2)
            -((y_grid - N) ** 2 / P**2)
            -((z_grid - Q) ** 2 / R**2)
        )

        rho = main + spike

    else:
        rho = I * torch.exp(
            -((x_grid - A - B * torch.cos(2 * math.pi / C * z_grid)) ** 2 / D**2)
            -((y_grid - E) ** 2 / Fp**2)
            -((z_grid - G) ** 2 / H**2)
        )

    return rho.clamp_min(0.0)


def smooth_along_z(rho: torch.Tensor, z_hwhm_um: float = 0.25) -> torch.Tensor:
    """
    Smooth rho(y, x, z) along z with a Gaussian-like kernel.

    Kernel convention:
        kernel = 2 ** (-(z / z_hwhm)^2)
    """
    kernel_z = torch.linspace(
        -z_size_um / 2,
        z_size_um / 2,
        Nz,
        device=device,
        dtype=dtype,
    )

    kernel = 2.0 ** (-(kernel_z / z_hwhm_um) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    pad_width = kernel.shape[-1] // 2

    rho_padded = F.pad(rho, (pad_width, pad_width), mode="reflect")
    ny, nx, nz_padded = rho_padded.shape

    rho_reshaped = rho_padded.reshape(-1, 1, nz_padded)
    smoothed = F.conv1d(rho_reshaped, kernel, padding=0)
    smoothed = smoothed.reshape(ny, nx, -1)

    return smoothed


# ---------------------------------------------------------------------
# Build distributions
# ---------------------------------------------------------------------
dist_list = []

for i, params in enumerate(param_sets):
    rho = make_custom_dist(params, i)
    rho = smooth_along_z(rho, z_hwhm_um=z_HWHM_um)
    rho = rho / rho.sum()
    dist_list.append(rho)

    print(
        f"sample {i}: {params['name']}, shape={tuple(rho.shape)}, "
        f"sum={rho.sum().item():.6f}, min={rho.min().item():.3e}, max={rho.max().item():.3e}"
    )


# ---------------------------------------------------------------------
# Plot distribution projections
# ---------------------------------------------------------------------
def plot_distribution_projections(
    dist_list,
    param_sets,
    x_axis_um,
    y_axis_um,
    z_axis_um,
    out_dir,
    cmap="viridis",
):
    """
    Plot XY, XZ, and YZ projections for each distribution.

    Input distribution shape is (Ny, Nx, Nz) = (y, x, z).
    """
    titles = [
        r"XY ($\Sigma_z \rho$)",
        r"XZ ($\Sigma_y \rho$)",
        r"YZ ($\Sigma_x \rho$)",
    ]

    xlabels = ["x [um]", "z [um]", "z [um]"]
    ylabels = ["y [um]", "x [um]", "y [um]"]

    x_np = x_axis_um.detach().cpu().numpy()
    y_np = y_axis_um.detach().cpu().numpy()
    z_np = z_axis_um.detach().cpu().numpy()

    extents = [
        [x_np.min(), x_np.max(), y_np.min(), y_np.max()],  # XY
        [z_np.min(), z_np.max(), x_np.min(), x_np.max()],  # XZ
        [z_np.min(), z_np.max(), y_np.min(), y_np.max()],  # YZ
    ]

    # match limits to each panel's actual coordinates
    xlims = [
        (x_np.min(), x_np.max()),
        (z_np.min(), z_np.max()),
        (z_np.min(), z_np.max()),
    ]
    ylims = [
        (y_np.min(), y_np.max()),
        (x_np.min(), x_np.max()),
        (y_np.min(), y_np.max()),
    ]

    for sample_idx, rho in enumerate(dist_list):
        rho_np = rho.detach().cpu().numpy()

        # rho_np shape: (y, x, z)
        proj_xy = rho_np.sum(axis=2)      # (Ny, Nx)
        proj_xz = rho_np.sum(axis=0)      # (Nx, Nz)
        proj_yz = rho_np.sum(axis=1)      # (Ny, Nz)

        projections = [proj_xy, proj_xz, proj_yz]

        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)

        for proj_idx, (ax, data) in enumerate(zip(axes, projections)):
            im = ax.imshow(
                data,
                origin="lower",
                extent=extents[proj_idx],
                aspect="equal",
                cmap=cmap,
            )

            ax.set_xlim(*xlims[proj_idx])
            ax.set_ylim(*ylims[proj_idx])

            ax.set_title(titles[proj_idx], fontsize=10)
            ax.set_xlabel(xlabels[proj_idx], fontsize=9)
            ax.set_ylabel(ylabels[proj_idx], fontsize=9)
            ax.tick_params(labelsize=8)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.minorticks_off()

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.locator = ticker.MaxNLocator(nbins=3)
            cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.get_offset_text().set_size(8)

        # fig.suptitle(
        #     f"sample {sample_idx}: {param_sets[sample_idx]['name']}",
        #     fontsize=11,
        # )

        outfile = out_dir / f"sample_{sample_idx:02d}_projections.png"
        fig.savefig(outfile, dpi=250, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.show()
        

plot_distribution_projections(
    dist_list=dist_list,
    param_sets=param_sets,
    x_axis_um=x_um,
    y_axis_um=y_um,
    z_axis_um=z_um,
    out_dir=out_dir,
)



#### vertical distribution plots
from matplotlib import ticker

def compute_projection_data(rho, x_um, y_um, z_um):
    """
    Compute XY, XZ, and YZ projections for rho(y, x, z).
    """
    rho_np = rho.detach().cpu().numpy()
    x_np = x_um.detach().cpu().numpy()
    y_np = y_um.detach().cpu().numpy()
    z_np = z_um.detach().cpu().numpy()

    proj_xy = rho_np.sum(axis=2)  # (Ny, Nx)
    proj_xz = rho_np.sum(axis=0)  # (Nx, Nz)
    proj_yz = rho_np.sum(axis=1)  # (Ny, Nz)

    return [
        {
            "data": proj_xy,
            "extent": [x_np.min(), x_np.max(), y_np.min(), y_np.max()],
            "title": r"XY: $\Sigma_z \rho$",
            "xlabel": "x [um]",
            "ylabel": "y [um]",
        },
        {
            "data": proj_xz,
            "extent": [z_np.min(), z_np.max(), x_np.min(), x_np.max()],
            "title": r"XZ: $\Sigma_y \rho$",
            "xlabel": "z [um]",
            "ylabel": "x [um]",
        },
        {
            "data": proj_yz,
            "extent": [z_np.min(), z_np.max(), y_np.min(), y_np.max()],
            "title": r"XZ: $\Sigma_y \rho$",
            "xlabel": "z [um]",
            "ylabel": "y [um]",
        },
    ]


def plot_sample_vertical(
    rho,
    x_um,
    y_um,
    z_um,
    sample_name,
    out_path,
    cmap="viridis",
    fs=14,
    tick_fs=10,
    cbar_tick_fs=9,
    axis_limit_um=20.0,
):
    """
    Clean vertical projection plot.

    Input rho shape is (Ny, Nx, Nz) = (y, x, z).
    """
    panels = compute_projection_data(rho, x_um, y_um, z_um)

    fig = plt.figure(figsize=(4.8, 8.6))
    
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.15,
        right=0.84,
        bottom=0.07,
        top=0.94,
        hspace=0.48,
        wspace=0.07,
    )

    # fig.suptitle(sample_name, fontsize=fs + 2, y=0.985)

    for j, panel in enumerate(panels):
        ax = fig.add_subplot(gs[j, 0])
        cax = fig.add_subplot(gs[j, 1])

        data = panel["data"]
        norm = Normalize(vmin=float(data.min()), vmax=float(data.max()))

        im = ax.imshow(
            data,
            origin="lower",
            extent=panel["extent"],
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )

        ax.set_xlim(-axis_limit_um, axis_limit_um)
        ax.set_ylim(-axis_limit_um, axis_limit_um)

        ax.set_title(panel["title"], fontsize=fs, pad=8)
        ax.set_xlabel(panel["xlabel"], fontsize=fs)
        ax.set_ylabel(panel["ylabel"], fontsize=fs)
        ax.tick_params(labelsize=tick_fs)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.minorticks_off()

        cbar = fig.colorbar(im, cax=cax)
        cbar.locator = ticker.MaxNLocator(nbins=3, prune="both")
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        cbar.ax.yaxis.get_offset_text().set_size(cbar_tick_fs)

        if j == 1:
            cbar.set_label(
                "charge density [a.u.]",
                rotation=90,
                labelpad=5,
                fontsize=fs,
            )

    fig.savefig(out_path, dpi=300)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.show()
    plt.close(fig)


for sample_idx, rho in enumerate(dist_list):
    plot_sample_vertical(
        rho=rho,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        sample_name=f"sample {sample_idx}: {param_sets[sample_idx]['name']}",
        out_path=out_dir / f"sample_{sample_idx:02d}_projections_vertical_clean.png",
        fs=12,
        tick_fs=9,
        cbar_tick_fs=8,
        axis_limit_um=20.0,
    )


# ---------------------------------------------------------------------
# Build SVFs and OTR model
# ---------------------------------------------------------------------
svf_gen = SVFGenerator(
    gamma=gamma,
    theta_max=theta_max,
    res=res,
    size=OTR_size_um,
    z_gauss_size=z_gauss_size_um,
    prefactor_x=1.0,
    prefactor_y=1.0,
    device=device,
)

svf_blocks = [svf_gen.forward(float(wl.item())) for wl in wavelengths_um]
SVFs = torch.cat(svf_blocks, dim=0).to(torch.complex64)

otr = OTRGenerator(
    wavelengths=wavelengths_um,
    SVFs=SVFs,
    N_e=N_e,
    z_bins_um=z_um,
)

# ---------------------------------------------------------------------
# Plot SVFs
# ---------------------------------------------------------------------
def plot_svfs(SVFs, wavelengths_um, out_dir):
    """
    Plot SVF channels.

    Channel convention per wavelength:
        0: E_x
        1: E_y
        2: |E|^2
    """
    svf_cpu = SVFs.detach().cpu()

    row_info = [
        (2, r"$|E|^2$ / PSF", "viridis", "fluence [a.u.]"),
        (0, r"Re($E_x$)", "bwr", "field [a.u.]"),
        (1, r"Re($E_y$)", "bwr", "field [a.u.]"),
    ]

    n_rows = len(row_info)
    n_wl = wavelengths_um.numel()

    fig, axes = plt.subplots(
        n_rows,
        n_wl,
        figsize=(3.5 * n_wl, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (channel_offset, row_label, cmap, cbar_label) in enumerate(row_info):
        row_data = []

        for j in range(n_wl):
            idx = j * 3 + channel_offset
            data = svf_cpu[idx].real
            row_data.append(data)

        all_row = torch.stack(row_data)
        vmin = float(all_row.min())
        vmax = float(all_row.max())

        if channel_offset != 2:
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs

        last_im = None

        for j in range(n_wl):
            ax = axes[row, j]
            data = row_data[j].numpy()

            last_im = ax.imshow(
                data,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                wl_nm = wavelengths_um[j].detach().cpu().item() * 1000.0
                ax.set_title(f"{wl_nm:.0f} nm", fontsize=12)

            if j == 0:
                ax.set_ylabel(row_label, fontsize=12)

        cbar = fig.colorbar(last_im, ax=axes[row, -1], fraction=0.046, pad=0.04)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.get_offset_text().set_size(8)

    outfile = out_dir / "svfs.png"
    fig.savefig(outfile, dpi=250, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.show()

plot_svfs(SVFs, wavelengths_um, out_dir)


# ---------------------------------------------------------------------
# Compute OTR modes
# ---------------------------------------------------------------------
otr_results = []

for sample_idx, rho in enumerate(dist_list):
    with torch.no_grad():
        iotr = otr.forward(rho, mode="incoherent")
        cotr = otr.forward(rho, mode="coherent")
        mixed = otr.forward(rho, mode="mixed")

    residual = (mixed - iotr - cotr).abs().max() / mixed.abs().max().clamp_min(1e-30)

    print(
        f"sample {sample_idx}: "
        f"iotr max={iotr.max().item():.3e}, "
        f"cotr max={cotr.max().item():.3e}, "
        f"mixed max={mixed.max().item():.3e}, "
        f"mode residual={residual.item():.3e}"
    )

    otr_results.append(
        {
            "name": param_sets[sample_idx]["name"],
            "iotr": iotr,
            "cotr": cotr,
            "mixed": mixed,
        }
    )


# ---------------------------------------------------------------------
# Plot OTR modes for each sample
# ---------------------------------------------------------------------
def _to_numpy(array):
    """Convert torch.Tensor or array-like to numpy array."""
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _make_norm(data_list, mode="linear"):
    """
    Create a Normalize object from a list of arrays.

    mode:
        "linear" -> standard linear Normalize
    """
    flat = np.concatenate([np.ravel(_to_numpy(d)) for d in data_list])
    vmin = float(np.nanmin(flat))
    vmax = float(np.nanmax(flat))

    # Avoid zero-width color scale.
    if np.isclose(vmin, vmax):
        eps = 1e-12 if vmax == 0 else abs(vmax) * 1e-6
        vmin -= eps
        vmax += eps

    return Normalize(vmin=vmin, vmax=vmax)


def _format_colorbar(cbar, tick_fs=10, label=None, label_fs=12):
    """Apply consistent scientific colorbar formatting."""
    cbar.locator = ticker.MaxNLocator(nbins=4, prune="both")
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.ax.yaxis.get_offset_text().set_size(tick_fs)

    if label is not None:
        cbar.set_label(label, rotation=90, labelpad=10, fontsize=label_fs)


def plot_otr_modes_multi_sample(
    otr_results,
    wavelengths_um,
    out_dir,
    cmap="viridis",
    norm_mode="per_wavelength",
    img_size=(3.4, 3.4),
    col_gap=0.18,
    row_gap=0.10,
    cbar_frac=0.055,
    cbar_gap=0.035,
    title_fs=14,
    label_fs=14,
    tick_fs=10,
    cbar_tick_fs=10,
    cbar_label_fs=12,
    label_rightmost_only=True,
):
    """
    Plot IOTR, COTR, and mixed OTR for each sample.

    norm_mode options
    -----------------
    "per_panel":
        Each subplot has its own colorbar and normalization.
        Best diagnostic mode when COTR is much weaker/stronger than IOTR.

    "per_wavelength":
        One shared colorbar per wavelength column across IOTR, COTR, mixed.
        Best publication-style comparison across radiation modes at fixed wavelength.

    "per_mode":
        One normalization per radiation mode row across wavelengths.
        Best for comparing wavelength dependence within IOTR, COTR, or mixed separately.
    """
    mode_keys = ["iotr", "cotr", "mixed"]
    mode_names = ["IOTR", "COTR", "IOTR + COTR"]

    wavelengths_nm = _to_numpy(wavelengths_um) * 1e3
    n_wl = len(wavelengths_nm)
    n_modes = len(mode_keys)

    for sample_idx, sample in enumerate(otr_results):
        data_np = {
            key: _to_numpy(sample[key])
            for key in mode_keys
        }

        fig = plt.figure(
            figsize=(n_wl * img_size[0] * (1.0 + cbar_frac), n_modes * img_size[1])
        )

        outer = GridSpec(
            1,
            n_wl,
            figure=fig,
            wspace=col_gap,
            left=0.08,
            right=0.92,
            top=0.88,
            bottom=0.10,
        )

        # Precompute normalizations.
        if norm_mode == "per_wavelength":
            norms = []
            for col in range(n_wl):
                norms.append(
                    _make_norm([data_np[key][col] for key in mode_keys])
                )

        elif norm_mode == "per_mode":
            norms = {}
            for key in mode_keys:
                norms[key] = _make_norm([data_np[key][col] for col in range(n_wl)])

        elif norm_mode == "per_panel":
            norms = None

        else:
            raise ValueError(
                "norm_mode must be one of 'per_panel', 'per_wavelength', or 'per_mode'. "
                f"Got {norm_mode!r}."
            )

        all_axes = []

        for col in range(n_wl):
            inner = GridSpecFromSubplotSpec(
                n_modes,
                1,
                subplot_spec=outer[0, col],
                hspace=row_gap,
            )

            col_axes = []
            last_im = None

            for row, key in enumerate(mode_keys):
                ax = fig.add_subplot(inner[row, 0])

                if norm_mode == "per_wavelength":
                    norm = norms[col]
                elif norm_mode == "per_mode":
                    norm = norms[key]
                else:
                    norm = _make_norm([data_np[key][col]])

                im = ax.imshow(
                    data_np[key][col],
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                )

                last_im = im
                col_axes.append(ax)
                all_axes.append(ax)

                ax.set_box_aspect(1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(labelsize=tick_fs)

                if row == 0:
                    ax.set_title(f"{wavelengths_nm[col]:.0f} nm", fontsize=title_fs)

                if col == 0:
                    ax.set_ylabel(
                        mode_names[row],
                        rotation=90,
                        va="center",
                        labelpad=14,
                        fontsize=label_fs,
                    )

                # Per-panel mode: colorbar attached to every panel.
                if norm_mode == "per_panel":
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    _format_colorbar(cbar, tick_fs=cbar_tick_fs)

            # Shared column colorbar for per-wavelength mode.
            if norm_mode == "per_wavelength":
                col_bbox = Bbox.union([ax.get_position() for ax in col_axes])

                image_width = col_axes[0].get_position().width
                cbar_width = cbar_frac * image_width
                gap = cbar_gap * image_width

                cax = fig.add_axes(
                    [col_bbox.x1 + gap, col_bbox.y0, cbar_width, col_bbox.height]
                )

                cbar = fig.colorbar(last_im, cax=cax, orientation="vertical")

                label = None
                if (not label_rightmost_only) or (col == n_wl - 1):
                    label = "fluence [a.u.]"

                _format_colorbar(
                    cbar,
                    tick_fs=cbar_tick_fs,
                    label=label,
                    label_fs=cbar_label_fs,
                )

        # Shared row colorbars for per-mode mode.
        if norm_mode == "per_mode":
            for row, key in enumerate(mode_keys):
                row_axes = [all_axes[col * n_modes + row] for col in range(n_wl)]
                row_bbox = Bbox.union([ax.get_position() for ax in row_axes])

                image_width = row_axes[-1].get_position().width
                cbar_width = cbar_frac * image_width
                gap = cbar_gap * image_width

                cax = fig.add_axes(
                    [row_bbox.x1 + gap, row_bbox.y0, cbar_width, row_bbox.height]
                )

                # Use the last image in this row. It already has the row norm.
                row_im = row_axes[-1].images[0]
                cbar = fig.colorbar(row_im, cax=cax, orientation="vertical")

                label = "fluence [a.u.]" if row == 1 else None
                _format_colorbar(
                    cbar,
                    tick_fs=cbar_tick_fs,
                    label=label,
                    label_fs=cbar_label_fs,
                )

        fig.suptitle(
            f"sample {sample_idx}: {sample['name']} ({norm_mode})",
            fontsize=title_fs + 2,
        )

        outfile = out_dir / f"sample_{sample_idx:02d}_otr_modes_{norm_mode}.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.show()


# each panel has its own colorbar
plot_otr_modes_multi_sample(
    otr_results=otr_results,
    wavelengths_um=wavelengths_um,
    out_dir=out_dir,
    cmap="viridis",
    norm_mode="per_panel",
    img_size=(3.2, 3.2),
    title_fs=13,
    label_fs=13,
    cbar_tick_fs=9,
    cbar_label_fs=11,
)

# one shared colorbar per wavelength column
plot_otr_modes_multi_sample(
    otr_results=otr_results,
    wavelengths_um=wavelengths_um,
    out_dir=out_dir,
    cmap="viridis",
    norm_mode="per_wavelength",
    img_size=(3.6, 3.6),
    col_gap=0.25,
    row_gap=0.08,
    cbar_frac=0.06,
    cbar_gap=0.035,
    title_fs=16,
    label_fs=16,
    cbar_tick_fs=11,
    cbar_label_fs=14,
    label_rightmost_only=True,
)

# ne shared colorbar per radiation mode row across wavelengths
plot_otr_modes_multi_sample(
    otr_results=otr_results,
    wavelengths_um=wavelengths_um,
    out_dir=out_dir,
    cmap="viridis",
    norm_mode="per_mode",
    img_size=(3.6, 3.6),
    col_gap=0.25,
    row_gap=0.08,
    cbar_frac=0.06,
    cbar_gap=0.035,
    title_fs=16,
    label_fs=16,
    cbar_tick_fs=11,
    cbar_label_fs=14,
)


###
def summarize_otr_result(sample_name, wavelengths_um, iotr, cotr, mixed):
    print()
    print(f"Sample: {sample_name}")
    print(
        "lambda_um | IOTR_sum | COTR_sum | mixed_sum | "
        "COTR/IOTR_sum | IOTR_max | COTR_max | mixed_max"
    )
    print("-" * 112)

    for j, wl in enumerate(wavelengths_um.detach().cpu()):
        i_sum = iotr[j].sum().item()
        c_sum = cotr[j].sum().item()
        m_sum = mixed[j].sum().item()

        i_max = iotr[j].max().item()
        c_max = cotr[j].max().item()
        m_max = mixed[j].max().item()

        ratio = c_sum / max(i_sum, 1e-30)

        print(
            f"{wl.item():8.3f} | "
            f"{i_sum:9.3e} | {c_sum:9.3e} | {m_sum:9.3e} | "
            f"{ratio:13.3e} | "
            f"{i_max:9.3e} | {c_max:9.3e} | {m_max:9.3e}"
        )



### N_e sweep
def run_ne_sweep(dist_list, param_sets, wavelengths_um, SVFs, z_um):
    e_charge = 1.60217663e-19
    measured_charge = 1e-10
    N_e_physical = measured_charge / e_charge

    N_e_values = [
        1.0,
        1_000.0,
        1.0e5,
        1.0e7,
        N_e_physical,
    ]

    print()
    print("=" * 100)
    print("N_e sweep")
    print(f"N_e physical for 100 pC = {N_e_physical:.6e}")
    print("=" * 100)

    for N_e in N_e_values:
        otr = OTRGenerator(
            wavelengths=wavelengths_um,
            SVFs=SVFs,
            N_e=N_e,
            z_bins_um=z_um,
        )

        print()
        print("#" * 100)
        print(f"N_e = {N_e:.6e}")
        print("#" * 100)

        for sample_idx, rho in enumerate(dist_list):
            with torch.no_grad():
                iotr = otr.forward(rho, mode="incoherent")
                cotr = otr.forward(rho, mode="coherent")
                mixed = otr.forward(rho, mode="mixed")

            mode_residual = (
                (mixed - iotr - cotr).abs().max()
                / mixed.abs().max().clamp_min(1e-30)
            )

            print(f"mode residual = {mode_residual.item():.3e}")

            summarize_otr_result(
                sample_name=param_sets[sample_idx]["name"],
                wavelengths_um=wavelengths_um,
                iotr=iotr,
                cotr=cotr,
                mixed=mixed,
            )

run_ne_sweep(
    dist_list=dist_list,
    param_sets=param_sets,
    wavelengths_um=wavelengths_um,
    SVFs=SVFs,
    z_um=z_um,
)

### J sweep
def run_spike_amplitude_sweep(
    base_params,
    wavelengths_um,
    SVFs,
    z_um,
    N_e,
    J_values=(0.0, 0.25, 0.5, 1.0, 2.0),
):
    print()
    print("=" * 100)
    print("Spike amplitude sweep for sample 1")
    print("=" * 100)

    otr = OTRGenerator(
        wavelengths=wavelengths_um,
        SVFs=SVFs,
        N_e=N_e,
        z_bins_um=z_um,
    )

    print("J | lambda_um | COTR/IOTR_sum | IOTR_sum | COTR_sum | mixed_sum")
    print("-" * 85)

    for J in J_values:
        params = dict(base_params)
        params["J"] = J

        rho = make_custom_dist(params, sample_index=1)
        rho = smooth_along_z(rho, z_hwhm_um=z_HWHM_um)
        rho = rho / rho.sum()

        with torch.no_grad():
            iotr = otr.forward(rho, mode="incoherent")
            cotr = otr.forward(rho, mode="coherent")
            mixed = otr.forward(rho, mode="mixed")

        for k, wl in enumerate(wavelengths_um.detach().cpu()):
            i_sum = iotr[k].sum().item()
            c_sum = cotr[k].sum().item()
            m_sum = mixed[k].sum().item()
            ratio = c_sum / max(i_sum, 1e-30)

            print(
                f"{J:4.2f} | {wl.item():8.3f} | "
                f"{ratio:14.3e} | {i_sum:9.3e} | {c_sum:9.3e} | {m_sum:9.3e}"
            )


run_spike_amplitude_sweep(
    base_params=param_sets[1],
    wavelengths_um=wavelengths_um,
    SVFs=SVFs,
    z_um=z_um,
    N_e=N_e,
)

### R sweeo
def run_spike_width_sweep(
    base_params,
    wavelengths_um,
    SVFs,
    z_um,
    N_e,
    R_values=(0.1, 0.2, 0.5, 1.0),
):
    print()
    print("=" * 100)
    print("Spike width sweep for sample 1")
    print("=" * 100)

    otr = OTRGenerator(
        wavelengths=wavelengths_um,
        SVFs=SVFs,
        N_e=N_e,
        z_bins_um=z_um,
    )

    print("R_um | lambda_um | COTR/IOTR_sum | IOTR_sum | COTR_sum | mixed_sum")
    print("-" * 88)

    for R in R_values:
        params = dict(base_params)
        params["R"] = R

        rho = make_custom_dist(params, sample_index=1)
        rho = smooth_along_z(rho, z_hwhm_um=z_HWHM_um)
        rho = rho / rho.sum()

        with torch.no_grad():
            iotr = otr.forward(rho, mode="incoherent")
            cotr = otr.forward(rho, mode="coherent")
            mixed = otr.forward(rho, mode="mixed")

        for k, wl in enumerate(wavelengths_um.detach().cpu()):
            i_sum = iotr[k].sum().item()
            c_sum = cotr[k].sum().item()
            m_sum = mixed[k].sum().item()
            ratio = c_sum / max(i_sum, 1e-30)

            print(
                f"{R:4.2f} | {wl.item():8.3f} | "
                f"{ratio:14.3e} | {i_sum:9.3e} | {c_sum:9.3e} | {m_sum:9.3e}"
            )


run_spike_width_sweep(
    base_params=param_sets[1],
    wavelengths_um=wavelengths_um,
    SVFs=SVFs,
    z_um=z_um,
    N_e=N_e,
)


print()
print("Saved outputs to:", out_dir)
