import matplotlib.pyplot as plt
import numpy as np

from .utils.physics import sld_to_density


def plot_analysis_result(
    q: np.ndarray, R_measured: np.ndarray, result: dict, title: str = "XRR Analysis"
):
    """
    Plot the measured data with the AI guess and the final fit.

    Args:
        q: q vector
        R_measured: measured reflectivity
        result: return value of reflecto.analyze()
    """
    plt.figure(figsize=(10, 6))

    plt.plot(q, R_measured, "ko", label="Measured", markersize=3, alpha=0.6)

    if "ai_curve" in result:
        plt.plot(q, result["ai_curve"], "b--", label="AI Guess", linewidth=1.5, alpha=0.8)

    if "fit_curve" in result:
        plt.plot(q, result["fit_curve"], "r-", label="Final Fit", linewidth=2)

    plt.yscale("log")
    plt.xlabel(r"$q \ [\AA^{-1}]$")
    plt.ylabel("Reflectivity")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    if "fit_params" in result:
        fp = result["fit_params"]
        info_text = (
            f"Thickness: {fp['set_f_d']:.2f} Å\n"
            f"Roughness: {fp['set_f_sig']:.2f} Å\n"
            f"SLD: {fp['set_f_sld']:.2f}"
        )
        plt.text(
            0.95,
            0.95,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def plot_profile(z: np.ndarray, sld_profile: np.ndarray, title: str = "Electron Density Profile"):
    """Plot the SLD profile as an electron density graph."""
    density_cm3 = sld_to_density(sld_profile)

    y_plot = density_cm3 / 1e23

    styles = {
        "title_fontsize": 16,
        "title_fontweight": "bold",
        "label_fontsize": 14,
        "label_fontweight": "bold",
        "tick_labelsize": 12,
        "tick_width": 1.5,
        "spine_width": 2.0,
        "legend_fontsize": 12,
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(z, y_plot, color="black", lw=2.0, label="Model")

    ax.fill_between(z, y_plot, 0, color="gray", alpha=0.1)

    ax.set_title(
        title, fontsize=styles["title_fontsize"], weight=styles["title_fontweight"], pad=15
    )
    ax.set_xlabel(
        r"Depth from Surface ($\AA$)",
        fontsize=styles["label_fontsize"],
        weight=styles["label_fontweight"],
    )
    ax.set_ylabel(
        r"Electron Density ($e/\mathrm{cm}^3$)",
        fontsize=styles["label_fontsize"],
        weight=styles["label_fontweight"],
    )

    # Annotation (x10^23)
    ax.annotate(
        r"$\times10^{23}$",
        xy=(0, 1),
        xytext=(5, 10),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=15,
        ha="left",
        va="center",
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=styles["tick_labelsize"],
        width=styles["tick_width"],
        length=6,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both", which="minor", width=1.0, length=3, direction="in", top=True, right=True
    )

    for spine in ax.spines.values():
        spine.set_linewidth(styles["spine_width"])

    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()
