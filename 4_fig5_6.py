"""
Weekly activity profiles and temporal clustering (Figs. 5 and 6).

Figure 5 uses the number of edits and Figure 6 the number of editors.
Each figure has four panels:
    (a) absolute weekly activity of each edition;
    (b) the same profiles, each edition rescaled to the interval [0, 1];
    (c) dendrogram of the timing-only representation;
    (d) dendrogram of the composition-and-timing representation.

The two representations are built from the 11 editions x 13 categories
x 168 hours tensor:
    (c) timing only: each (edition, category) block is divided by its sum
        over the 168 hours, which removes the size of the edition and the
        weight of each category and keeps only the shape of the weekly
        cycle.
    (d) composition and timing: each edition is divided by its total
        activity, which removes the size of the edition but keeps the
        relative weight of the categories.

Input:
    The tensor cache tensor_cat_<metric>.csv produced by
    dendrogramas_dos_normalizaciones.py, which must be in the same folder
    as this script. With --recompute the tensor is rebuilt from the raw
    data files.

Output:
    figuras/fig5_edits.pdf
    figuras/fig6_editors.pdf

Usage:
    python 4_fig5_6.py
    python 4_fig5_6.py --recompute
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# The tensor and the two normalisations are imported from the module used
# in the rest of the analysis, so that all figures rely on the same
# definitions.
try:
    from dendrogramas_dos_normalizaciones import (
        LANGUAGES, OUT_DIR, load_tensor, block_mask, normalise, ward,
    )
except ImportError as e:
    raise SystemExit(
        "dendrogramas_dos_normalizaciones.py not found in this folder.\n"
        f"Details: {e}")


# =================================================================
# Configuration
# =================================================================

HOURS_PER_WEEK = 168
DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# One colour per edition, the same in both figures.
COLORS = {lang: c for lang, c in zip(
    LANGUAGES, plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:len(LANGUAGES)])}

# Normalisation modes of the imported module and the title of their panel.
DENDROGRAM_TITLES = {
    "timing": "(c) Timing only\n(each category normalised to unit sum)",
    "composition": "(d) Composition + timing\n(each language normalised by total activity)",
}


# =================================================================
# Weekly profiles
# =================================================================

def aggregate_profile(T):
    """
    Sum the 13 categories of the tensor.

    Args:
        T: array of shape (11, 13, 168).

    Returns:
        Array of shape (11, 168) with the weekly profile of each edition.
    """
    return T.sum(axis=1)


def minmax_rescale(P):
    """
    Rescale each edition's profile to the interval [0, 1], so that the
    shape of the weekly cycle can be compared across editions of very
    different size.
    """
    lo = P.min(axis=1, keepdims=True)
    hi = P.max(axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (P - lo) / span


# =================================================================
# Plotting
# =================================================================

def format_week_axis(ax):
    """Label the x axis with the hour of the week, one tick every 6 hours."""
    ax.set_xticks(np.arange(0, HOURS_PER_WEEK, 6))
    ax.set_xticklabels([f"{d} {h:02d}h" for d in DAY_NAMES
                        for h in range(0, 24, 6)], rotation=90, fontsize=6)
    ax.set_xlabel("Hour of the week", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_profiles(ax, P, ylabel, title, legend=False):
    """Draw one weekly profile per edition."""
    for i, lang in enumerate(LANGUAGES):
        ax.plot(np.arange(HOURS_PER_WEEK), P[i], color=COLORS[lang],
                lw=1.0, label=lang)
    format_week_axis(ax)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    if legend:
        ax.legend(title="Languages", fontsize=7, title_fontsize=7,
                  ncol=2, loc="upper right", framealpha=0.85)


def plot_dendrogram(ax, Z, title):
    """Draw a dendrogram with the height of each merge written on it."""
    d = dendrogram(Z, labels=LANGUAGES, ax=ax, distance_sort="descending",
                   leaf_rotation=90, leaf_font_size=9, show_leaf_counts=True)
    for ic, dc, col in zip(d["icoord"], d["dcoord"], d["color_list"]):
        x = 0.5 * sum(ic[1:3])
        y = dc[1]
        if y > 0:
            ax.plot(x, y, "o", ms=3, c=col)
            ax.annotate(f"{y:.3g}", (x, y), xytext=(0, -8),
                        textcoords="offset points", va="top", ha="center",
                        fontsize=6)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("Distance", fontsize=9)
    ax.grid(False)


def make_figure(metric, number, recompute=False):
    """
    Build and save one figure.

    Args:
        metric: 'edits' or 'editors'.
        number: figure number, used in the output file name.
        recompute: rebuild the tensor from the raw data instead of the cache.
    """
    # Load the tensor and mark the (edition, category) blocks with activity.
    T = load_tensor(metric, recompute=recompute)
    mask = block_mask(T)

    # Weekly profiles for panels (a) and (b).
    P = aggregate_profile(T)
    P_rescaled = minmax_rescale(P)

    # Ward clustering of the two normalised representations.
    Z = {mode: ward(normalise(T, mode, mask)) for mode in DENDROGRAM_TITLES}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_profiles(axes[0][0], P, f"Number of {metric}",
                  "(a) Absolute activity", legend=True)
    plot_profiles(axes[0][1], P_rescaled, f"Normalised number of {metric}",
                  "(b) Normalised activity")
    plot_dendrogram(axes[1][0], Z["timing"], DENDROGRAM_TITLES["timing"])
    plot_dendrogram(axes[1][1], Z["composition"], DENDROGRAM_TITLES["composition"])

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"fig{number}_{metric}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# =================================================================
# Main
# =================================================================

def main():
    recompute = "--recompute" in sys.argv[1:]
    make_figure("edits", 5, recompute=recompute)
    make_figure("editors", 6, recompute=recompute)


if __name__ == "__main__":
    main()
