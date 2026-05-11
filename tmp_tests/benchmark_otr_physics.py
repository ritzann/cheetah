from __future__ import annotations

import csv
import math
from pathlib import Path
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cheetah.accelerator.screen_otr import OTRScreen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
T = lambda value: torch.tensor(value, device=device, dtype=dtype)

OUTDIR = Path("tmp_tests/otr_benchmark_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def latex_num(value: float, precision: int = 3) -> str:
    """Return a siunitx-compatible number, e.g. \num{1.234e-5}."""
    if value == 0:
        return r"\num{0}"
    return rf"\num{{{value:.{precision}e}}}"


def lambda_key(lambda_um: float) -> str:
    """Create a CSV-safe key for a wavelength."""
    return f"lambda_{lambda_um:.1f}um".replace(".", "p")


def ideal_log10_F2(sigma_z_um: float, lambda_um: float) -> float:
    r"""
    Compute log10 of the ideal Gaussian longitudinal form factor intensity.
        |F(\lambda)|^2 = exp[-(2 pi \sigma_z / \lambda)^2]

    Returning log10 avoids underflow for long bunches.
    """
    exponent = -((2.0 * math.pi * sigma_z_um / lambda_um) ** 2)
    return exponent / math.log(10.0)


def ideal_F2_from_log10(log10_F2: float) -> float:
    """Convert log10(|F|^2) to |F|^2, with safe underflow handling."""
    if log10_F2 < -300:
        return 0.0
    return 10.0 ** log10_F2


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Model construction
# ------------------------------------------------------------
def make_screen(
    *,
    wavelengths_um: torch.Tensor,
    z_size_um: float,
    z_res: float,
    resolution: int,
    N_e: float,
) -> OTRScreen:
    r"""
    Create an OTRScreen.

    Here z_res is in pix / um, so dz = 1 / z_res.

    Wavelengths \lambda are in um. Transverse screen coordinates are in m.
    """
    return OTRScreen(
        resolution=(resolution, resolution),
        pixel_size=torch.tensor(
            [120e-6 / resolution, 150e-6 / resolution],
            device=device,
            dtype=dtype,
        ),
        kde_bandwidth=T(1e-6),
        is_active=True,
        z_size=z_size_um,
        z_res=z_res,
        wavelengths=wavelengths_um,
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
        otr_mode="mixed",
        device=device,
        dtype=dtype,
    )


def make_gaussian_dist(
    screen: OTRScreen,
    *,
    sigma_x_um: float,
    sigma_y_um: float,
    sigma_z_um: float,
) -> torch.Tensor:
    r"""
    Build analytic separable 3D Gaussian density:
        \rho(x, y, z) = \rho_xy(x, y) \rho_z(z)

    normalized such that
        sum_{x,y,z} \rho(x,y,z) = 1.

    The returned tensor has shape (H, W, N_z).
    """
    x_um = screen.pixel_bin_centers[0] * 1e6
    y_um = screen.pixel_bin_centers[1] * 1e6
    z_um = screen.z_bins * 1e6

    g_x = torch.exp(-0.5 * (x_um / sigma_x_um) ** 2)
    g_y = torch.exp(-0.5 * (y_um / sigma_y_um) ** 2)
    g_z = torch.exp(-0.5 * (z_um / sigma_z_um) ** 2)

    rho_xy = g_x[:, None] * g_y[None, :]
    rho_xy = rho_xy / rho_xy.sum()

    rho_z = g_z / g_z.sum()

    dist = rho_xy[:, :, None] * rho_z[None, None, :]
    dist = dist / dist.sum()

    return dist


@torch.no_grad()
def evaluate_dist(screen: OTRScreen, dist: torch.Tensor) -> dict:
    """
    Evaluate incoherent, coherent, and mixed OTR/COTR images.

    Returns integrated intensities and consistency residual:
        residual = max |I_mixed - I_incoh - I_coh| / max |I_mixed|.
    """
    I_incoh = screen.otr.forward(dist, mode="incoherent")
    I_coh = screen.otr.forward(dist, mode="coherent")
    I_mixed = screen.otr.forward(dist, mode="mixed")

    I_incoh_sum = I_incoh.sum(dim=(-2, -1))
    I_coh_sum = I_coh.sum(dim=(-2, -1))
    I_mixed_sum = I_mixed.sum(dim=(-2, -1))

    ratio = I_coh_sum / I_incoh_sum.clamp_min(1e-30)

    residual = (
        (I_mixed - I_incoh - I_coh).abs().max()
        / I_mixed.abs().max().clamp_min(1e-30)
    )

    return {
        "I_incoh_sum": I_incoh_sum.detach().cpu(),
        "I_coh_sum": I_coh_sum.detach().cpu(),
        "I_mixed_sum": I_mixed_sum.detach().cpu(),
        "ratio": ratio.detach().cpu(),
        "residual": float(residual.detach().cpu()),
    }


# ------------------------------------------------------------
# Bunch-length scans
# ------------------------------------------------------------
def run_bunch_scan(
    *,
    scan_name: str,
    wavelengths_um: torch.Tensor,
    sigma_z_list_um: list[float],
    z_size_um: float,
    z_res: float,
    resolution: int,
    N_e: float,
) -> list[dict]:
    """
    Run a bunch-length scan.

    For long bunches, use a large z_size so the Gaussian is not clipped.
    For short bunches, use a smaller z_size and finer z_res.
    """
    print()
    print(f"Running bunch scan: {scan_name}")
    print(f"  z_size = {z_size_um:g} um")
    print(f"  z_res  = {z_res:g} pix/um")
    print(f"  dz     = {1.0 / z_res:g} um")
    print(f"  resolution = {resolution} x {resolution}")

    screen = make_screen(
        wavelengths_um=wavelengths_um,
        z_size_um=z_size_um,
        z_res=z_res,
        resolution=resolution,
        N_e=N_e,
    )

    wavelengths_cpu = wavelengths_um.detach().cpu().tolist()
    rows = []

    for sigma_z_um in sigma_z_list_um:
        dist = make_gaussian_dist(
            screen,
            sigma_x_um=10.0,
            sigma_y_um=15.0,
            sigma_z_um=sigma_z_um,
        )

        metrics = evaluate_dist(screen, dist)

        row = {
            "scan": scan_name,
            "sigma_z_um": sigma_z_um,
            "z_size_um": z_size_um,
            "z_res_pix_per_um": z_res,
            "dz_um": 1.0 / z_res,
            "resolution": resolution,
            "residual": metrics["residual"],
        }

        for j, lambda_um in enumerate(wavelengths_cpu):
            key = lambda_key(lambda_um)
            log10_F2 = ideal_log10_F2(sigma_z_um, lambda_um)

            row[f"{key}_I_incoh_sum"] = float(metrics["I_incoh_sum"][j])
            row[f"{key}_I_coh_sum"] = float(metrics["I_coh_sum"][j])
            row[f"{key}_I_mixed_sum"] = float(metrics["I_mixed_sum"][j])
            row[f"{key}_ratio"] = float(metrics["ratio"][j])
            row[f"{key}_ideal_log10_F2"] = log10_F2
            row[f"{key}_ideal_F2"] = ideal_F2_from_log10(log10_F2)

        rows.append(row)

    return rows


def write_bunch_scan_latex_table(
    rows: list[dict],
    *,
    wavelengths_um: list[float],
    path: Path,
    caption: str,
    label: str,
) -> None:
    """
    Write a publication-style LaTeX table for I_coh/I_incoh.

    This table intentionally excludes the ideal |F|^2 columns to keep it compact.
    The ideal values are saved separately in the CSV.
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\sigma_z$ [$\mu$m] & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.4\,\mu$m & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.6\,\mu$m & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.8\,\mu$m & "
        r"residual \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        vals = []
        for lambda_um in wavelengths_um:
            key = lambda_key(lambda_um)
            vals.append(latex_num(row[f"{key}_ratio"], precision=3))

        line = (
            f"{row['sigma_z_um']:g} & "
            f"{vals[0]} & {vals[1]} & {vals[2]} & "
            f"{latex_num(row['residual'], precision=3)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    lines.append("")

    path.write_text("\n".join(lines))


# ------------------------------------------------------------
# Normalized comparison with ideal |F(\lambda)|^2
# ------------------------------------------------------------
def plot_bunch_scan(
    rows: list[dict],
    *,
    wavelengths_um: list[float],
    stem: str,
    title: str,
) -> None:
    r"""
    Save two figures:

    1. Measured I_coh/I_incoh versus \sigma_z.
    2. Normalized measured ratio compared with normalized ideal |F|^2.

    The normalized comparison uses the shortest \sigma_z as reference,
    which cancels an unknown wavelength-dependent prefactor.
    """
    rows_sorted = sorted(rows, key=lambda row: row["sigma_z_um"], reverse=True)

    sigma_vals = torch.tensor([row["sigma_z_um"] for row in rows_sorted])

    # Figure 1: measured ratio.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for lambda_um in wavelengths_um:
        key = lambda_key(lambda_um)
        ratios = torch.tensor([row[f"{key}_ratio"] for row in rows_sorted])

        ax.loglog(
            sigma_vals,
            ratios,
            marker="o",
            label=rf"$\lambda = {lambda_um:.1f}\,\mu\mathrm{{m}}$",
        )

    ax.invert_xaxis()
    ax.set_xlabel(r"rms bunch length $\sigma_z$ [$\mu$m]")
    ax.set_ylabel(r"integrated ratio $I_{\mathrm{coh}}/I_{\mathrm{incoh}}$")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTDIR / f"{stem}_ratio.png")
    fig.savefig(OUTDIR / f"{stem}_ratio.pdf")
    plt.close(fig)

    # Figure 2: normalized comparison with ideal |F|^2.
    # Use the shortest sigma_z as the reference.
    ref_row = min(rows_sorted, key=lambda row: row["sigma_z_um"])
    fig, axes = plt.subplots(1, len(wavelengths_um), figsize=(13, 4), sharey=True)

    if len(wavelengths_um) == 1:
        axes = [axes]

    for ax, lambda_um in zip(axes, wavelengths_um):
        key = lambda_key(lambda_um)

        measured = torch.tensor([row[f"{key}_ratio"] for row in rows_sorted])
        measured_ref = ref_row[f"{key}_ratio"]
        measured_norm = measured / max(measured_ref, 1e-300)

        log10_F2 = torch.tensor(
            [row[f"{key}_ideal_log10_F2"] for row in rows_sorted],
            dtype=torch.float64,
        )
        log10_F2_ref = ref_row[f"{key}_ideal_log10_F2"]

        # Normalized ideal: |F(\sigma_z)|^2 / |F(\sigma_z,ref)|^2
        # Compute using log10 to avoid underflow
        ideal_norm = torch.pow(10.0, log10_F2 - log10_F2_ref)

        ax.loglog(
            sigma_vals,
            measured_norm,
            marker="o",
            label=r"measured",
        )
        ax.loglog(
            sigma_vals,
            ideal_norm,
            linestyle="--",
            marker="s",
            label=r"ideal Gaussian",
        )

        ax.invert_xaxis()
        ax.set_xlabel(r"$\sigma_z$ [$\mu$m]")
        ax.set_title(rf"$\lambda = {lambda_um:.1f}\,\mu\mathrm{{m}}$")
        ax.grid(True, which="both")
        ax.legend()

    axes[0].set_ylabel(r"normalized ratio")
    fig.suptitle(
        r"Normalized comparison with ideal $|F(\lambda)|^2$",
        y=1.03,
    )
    fig.tight_layout()

    fig.savefig(OUTDIR / f"{stem}_normalized_vs_ideal.png")
    fig.savefig(OUTDIR / f"{stem}_normalized_vs_ideal.pdf")
    plt.close(fig)


# ------------------------------------------------------------
# z_res invariance benchmark
# ------------------------------------------------------------
def run_zres_invariance(
    *,
    wavelengths_um: torch.Tensor,
    sigma_z_um: float,
    z_size_um: float,
    z_res_list: list[float],
    resolution: int,
    N_e: float,
) -> list[dict]:
    """
    Test whether the result is stable when changing z_res.

    Since dist is normalized as a probability mass, dist.sum() = 1,

    changing z_res should not strongly change I_coh/I_incoh
    once the bunch and the phase oscillations are resolved.
    """
    print()
    print("Running z_res invariance scan")
    print(f"  sigma_z = {sigma_z_um:g} um")
    print(f"  z_size  = {z_size_um:g} um")
    print(f"  z_res values = {z_res_list}")

    wavelengths_cpu = wavelengths_um.detach().cpu().tolist()
    rows = []

    for z_res in z_res_list:
        screen = make_screen(
            wavelengths_um=wavelengths_um,
            z_size_um=z_size_um,
            z_res=z_res,
            resolution=resolution,
            N_e=N_e,
        )

        dist = make_gaussian_dist(
            screen,
            sigma_x_um=10.0,
            sigma_y_um=15.0,
            sigma_z_um=sigma_z_um,
        )

        metrics = evaluate_dist(screen, dist)

        row = {
            "sigma_z_um": sigma_z_um,
            "z_size_um": z_size_um,
            "z_res_pix_per_um": z_res,
            "dz_um": 1.0 / z_res,
            "resolution": resolution,
            "residual": metrics["residual"],
        }

        for j, lambda_um in enumerate(wavelengths_cpu):
            key = lambda_key(lambda_um)
            row[f"{key}_I_incoh_sum"] = float(metrics["I_incoh_sum"][j])
            row[f"{key}_ratio"] = float(metrics["ratio"][j])

        rows.append(row)

    return rows


def write_zres_latex_table(
    rows: list[dict],
    *,
    wavelengths_um: list[float],
    path: Path,
) -> None:
    """Write a compact LaTeX table for the z_res invariance benchmark."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$z_{\mathrm{res}}$ [pix/$\mu$m] & "
        r"$\Delta z$ [$\mu$m] & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.4\,\mu$m & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.6\,\mu$m & "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$, $\lambda=0.8\,\mu$m & "
        r"residual \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        vals = []
        for lambda_um in wavelengths_um:
            key = lambda_key(lambda_um)
            vals.append(latex_num(row[f"{key}_ratio"], precision=3))

        line = (
            f"{row['z_res_pix_per_um']:g} & "
            f"{row['dz_um']:g} & "
            f"{vals[0]} & {vals[1]} & {vals[2]} & "
            f"{latex_num(row['residual'], precision=3)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\caption{Longitudinal-grid invariance benchmark at fixed "
        r"$\sigma_z$. The ratio "
        r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$ should remain stable once "
        r"the bunch and the longitudinal phase oscillations are resolved.}"
    )
    lines.append(r"\label{tab:otr_zres_invariance}")
    lines.append(r"\end{table*}")
    lines.append("")

    path.write_text("\n".join(lines))


def plot_zres_invariance(
    rows: list[dict],
    *,
    wavelengths_um: list[float],
    stem: str,
) -> None:
    """Save z_res invariance figure."""
    rows_sorted = sorted(rows, key=lambda row: row["z_res_pix_per_um"])

    z_res_vals = torch.tensor([row["z_res_pix_per_um"] for row in rows_sorted])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for lambda_um in wavelengths_um:
        key = lambda_key(lambda_um)
        ratios = torch.tensor([row[f"{key}_ratio"] for row in rows_sorted])

        ax.plot(
            z_res_vals,
            ratios,
            marker="o",
            label=rf"$\lambda = {lambda_um:.1f}\,\mu\mathrm{{m}}$",
        )

    ax.set_xlabel(r"$z_{\mathrm{res}}$ [pix/$\mu$m]")
    ax.set_ylabel(r"$I_{\mathrm{coh}}/I_{\mathrm{incoh}}$")
    ax.set_title(r"Longitudinal-grid invariance at fixed $\sigma_z$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTDIR / f"{stem}.png")
    fig.savefig(OUTDIR / f"{stem}.pdf")
    plt.close(fig)


# ------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------
def main() -> None:
    print(f"device = {device}")
    print(f"dtype  = {dtype}")
    print(f"output = {OUTDIR}")

    wavelengths_um = torch.tensor([0.4, 0.6, 0.8], device=device, dtype=dtype)
    wavelengths_list = wavelengths_um.detach().cpu().tolist()

    N_e = 1_000.0

    # --------------------------------------------------------
    # a: short-bunch scan
    # --------------------------------------------------------
    # This scan is the physically clean COTR coherence test:
    #   shorter \sigma_z -> larger coherent contribution
    #   longer \lambda   -> larger coherent contribution
    # --------------------------------------------------------
    short_rows = run_bunch_scan(
        scan_name="short_bunch",
        wavelengths_um=wavelengths_um,
        sigma_z_list_um=[0.5, 0.2, 0.1, 0.05],
        z_size_um=20.0,
        z_res=20.0,
        resolution=48,
        N_e=N_e,
    )

    write_csv(OUTDIR / "short_bunch_scan.csv", short_rows)
    write_bunch_scan_latex_table(
        short_rows,
        wavelengths_um=wavelengths_list,
        path=OUTDIR / "short_bunch_scan_table.tex",
        caption=(
            r"Short-bunch COTR scan. The integrated coherent-to-incoherent "
            r"ratio increases as the bunch length $\sigma_z$ decreases and "
            r"as the wavelength $\lambda$ increases."
        ),
        label="tab:otr_short_bunch_scan",
    )
    plot_bunch_scan(
        short_rows,
        wavelengths_um=wavelengths_list,
        stem="short_bunch_scan",
        title=r"COTR coherence increases for shorter $\sigma_z$",
    )

    # --------------------------------------------------------
    # b. long-bunch scan with larger z-window
    # --------------------------------------------------------
    # This scan is meant to diagnose the artificial dip observed in
    # the first quick scan. Long bunches require a larger z_size to
    # reduce clipping of \rho_z(z).
    # --------------------------------------------------------
    long_rows = run_bunch_scan(
        scan_name="long_bunch_large_window",
        wavelengths_um=wavelengths_um,
        sigma_z_list_um=[8.0, 4.0, 2.0, 1.0, 0.5],
        z_size_um=100.0,
        z_res=10.0,
        resolution=40,
        N_e=N_e,
    )

    write_csv(OUTDIR / "long_bunch_large_window_scan.csv", long_rows)
    write_bunch_scan_latex_table(
        long_rows,
        wavelengths_um=wavelengths_list,
        path=OUTDIR / "long_bunch_large_window_scan_table.tex",
        caption=(
            r"Long-bunch scan with an enlarged longitudinal window to reduce "
            r"truncation of $\rho_z(z)$. In this regime the coherent signal "
            r"is strongly suppressed and can approach the numerical floor."
        ),
        label="tab:otr_long_bunch_scan",
    )
    plot_bunch_scan(
        long_rows,
        wavelengths_um=wavelengths_list,
        stem="long_bunch_large_window_scan",
        title=r"Long-bunch COTR scan with enlarged $z$ window",
    )

    # --------------------------------------------------------
    # z_res invariance benchmark
    # --------------------------------------------------------
    zres_rows = run_zres_invariance(
        wavelengths_um=wavelengths_um,
        sigma_z_um=0.1,
        z_size_um=20.0,
        z_res_list=[10.0, 20.0, 40.0],
        resolution=40,
        N_e=N_e,
    )

    write_csv(OUTDIR / "zres_invariance.csv", zres_rows)
    write_zres_latex_table(
        zres_rows,
        wavelengths_um=wavelengths_list,
        path=OUTDIR / "zres_invariance_table.tex",
    )
    plot_zres_invariance(
        zres_rows,
        wavelengths_um=wavelengths_list,
        stem="zres_invariance",
    )

    print()
    print("Done.")
    print("Generated files:")
    for path in sorted(OUTDIR.glob("*")):
        print(" ", path)


if __name__ == "__main__":
    main()