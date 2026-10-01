"""
Robustness of the language pairs to the dimensionality reduction method
(Supplementary Information, Fig. S1).

Each of the two temporal representations (timing only; composition and
timing) is reduced with four methods before Ward clustering (PCA,
autoencoder, t-SNE and UMAP) and compared with the unreduced clustering.
Repeating this for edits and editors gives 4 conditions x 5 treatments =
20 combinations.

In every clustering all 55 pairs of editions are evaluated. A pair "forms"
when the two editions join each other directly in a merge, before either
of them has joined any other edition. The figure shows, for each pair that
forms at least once, whether it forms in each combination and the height
at which the two editions first meet, relative to the root.

Input:
    The tensor cache produced by dendrogramas_dos_normalizaciones.py.
    That module and dimensionality_reduction.py must be in the same folder
    as this script.

Output:
    figuras/figSI_test_pares.pdf
    resultados_figSI_pares.txt  (counts and merge heights)

Usage:
    python 6_figSI.py
"""

import random
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from dendrogramas_dos_normalizaciones import (
        LANGUAGES, DATA_DIR, OUT_DIR, load_tensor, block_mask, normalise, ward,
    )
    from dimensionality_reduction import (
        REDUCTIONS, METHOD_ORDER, global_scale, HAS_TF, HAS_UMAP,
    )
except ImportError as e:
    raise SystemExit(
        "dendrogramas_dos_normalizaciones.py and dimensionality_reduction.py "
        f"must be in this folder.\nDetails: {e}")


# =================================================================
# Parameters
# =================================================================

# Global seed for Python, NumPy and TensorFlow. Each reduction method also
# fixes its own seed (RANDOM_STATE in dimensionality_reduction.py); this
# additionally fixes the global state and makes TensorFlow operations
# deterministic. The results reported in the SI were obtained with it.
SEED = 0

# The four conditions: (metric, normalisation).
CONDITIONS = [(m, n) for m in ("edits", "editors")
              for n in ("timing", "composition")]
CONDITION_LABELS = {
    ("edits", "timing"): "edits — timing only",
    ("edits", "composition"): "edits — composition + timing",
    ("editors", "timing"): "editors — timing only",
    ("editors", "composition"): "editors — composition + timing",
}
METHOD_LABELS = {"none": "none", "pca": "PCA(5)", "autoencoder": "autoencoder",
                 "tsne": "t-SNE", "umap": "UMAP"}

# Pairs discussed in the main text; their labels are shown in bold.
MANUSCRIPT_PAIRS = [("ru", "zh"), ("es", "pt"), ("vi", "ar"), ("de", "fr")]

# Cell colours: light grey when the pair does not form, blue when it does.
BINARY_COLORS = ListedColormap(["#f2f2f2", "#2c7fb8"])

ALL_PAIRS = list(combinations(LANGUAGES, 2))


# =================================================================
# Measures on a dendrogram
# =================================================================

def merge_height(Z, a, b):
    """
    Find where two editions first meet in a dendrogram.

    Returns:
        Tuple (relative height, direct): the height of the merge that first
        joins a and b, divided by the height of the root; and whether that
        merge joins a and b directly, before either joined another edition.
    """
    n = len(LANGUAGES)
    ia, ib = LANGUAGES.index(a), LANGUAGES.index(b)
    members = {i: {i} for i in range(n)}
    root = Z[-1, 2]
    for step, (x, y, dist, _) in enumerate(Z):
        x, y = int(x), int(y)
        merged = members[x] | members[y]
        members[n + step] = merged
        if ia in merged and ib in merged:
            direct = (members[x] == {ia} and members[y] == {ib}) or \
                     (members[x] == {ib} and members[y] == {ia})
            return (dist / root if root > 0 else np.nan), direct
    return np.nan, False


def measure_pairs(Z):
    """Relative height and direct-pair flag for all 55 pairs."""
    return {pair: merge_height(Z, *pair) for pair in ALL_PAIRS}


# =================================================================
# Analysis
# =================================================================

def analyse():
    """Cluster every condition with every method."""
    results = {}
    tensors = {m: load_tensor(m) for m in ("edits", "editors")}
    for cond in CONDITIONS:
        metric, mode = cond
        T = tensors[metric]
        mask = block_mask(T)
        X = global_scale(normalise(T, mode, mask))

        # Ward clustering after each reduction.
        Zs, skipped = {}, []
        for name in METHOD_ORDER:
            Y = REDUCTIONS[name](X)
            if Y is None:
                skipped.append(name)
                continue
            Zs[name] = ward(Y)
        print(f"[{CONDITION_LABELS[cond]}] methods: {list(Zs)}")

        measures = {m: measure_pairs(Z) for m, Z in Zs.items()}
        results[cond] = dict(measures=measures, skipped=skipped)
    return results


def pairs_to_show(results):
    """
    Pairs that form at least once, sorted by the number of combinations in
    which they form. Returns the sorted list and the count of every pair.
    """
    count = {pair: 0 for pair in ALL_PAIRS}
    for r in results.values():
        for m in r["measures"].values():
            for pair, (_h, direct) in m.items():
                count[pair] += int(bool(direct))
    shown = [p for p in ALL_PAIRS if count[p] > 0]
    return sorted(shown, key=lambda p: -count[p]), count


# =================================================================
# Figure
# =================================================================

def plot_figure(results, pairs):
    """Draw one panel per condition, with one column per method."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 3 + 0.40 * len(pairs) * 2))
    labels = [f"{a}-{b}" for a, b in pairs]
    for k, (ax, cond) in enumerate(zip(axes.flat, CONDITIONS)):
        r = results[cond]
        methods = list(r["measures"])

        # H: relative merge heights; B: 1 where the pair forms.
        H = np.full((len(pairs), len(methods)), np.nan)
        B = np.zeros_like(H)
        for j, m in enumerate(methods):
            for i, pair in enumerate(pairs):
                h, direct = r["measures"][m][pair]
                H[i, j], B[i, j] = h, 1.0 if direct else 0.0

        # Cells drawn as vector rectangles (pcolormesh rather than imshow,
        # which some PDF viewers distort).
        ax.pcolormesh(np.arange(len(methods) + 1) - 0.5,
                      np.arange(len(pairs) + 1) - 0.5,
                      B, cmap=BINARY_COLORS, vmin=0, vmax=1)
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.set_ylim(len(pairs) - 0.5, -0.5)      # first pair at the top
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(labels, fontsize=8)

        # White lines between cells.
        ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

        # Relative height written in each cell; manuscript pairs in bold.
        for i in range(len(pairs)):
            for j in range(len(methods)):
                if np.isnan(H[i, j]):
                    continue
                ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if B[i, j] else "#555555")
            if pairs[i] in MANUSCRIPT_PAIRS or pairs[i][::-1] in MANUSCRIPT_PAIRS:
                ax.get_yticklabels()[i].set_fontweight("bold")
        ax.set_title(f"({chr(97 + k)}) {CONDITION_LABELS[cond]}", fontsize=9)

    fig.legend(handles=[Patch(facecolor="#2c7fb8", label="pair"),
                        Patch(facecolor="#f2f2f2", edgecolor="#cccccc",
                              label="no pair")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "figSI_test_pares.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# =================================================================
# Report
# =================================================================

def write_report(results, pairs, count):
    """Save and print the counts and the merge heights."""
    total = sum(len(r["measures"]) for r in results.values())
    lines = ["SUPPLEMENTARY FIGURE - robustness of the pairs", ""]

    # Number of combinations in which each pair forms.
    lines.append(f"COUNTS over {total} combinations (method x condition)")
    for p in pairs:
        mark = "  <-- main text" if (p in MANUSCRIPT_PAIRS or
                                     p[::-1] in MANUSCRIPT_PAIRS) else ""
        lines.append(f"  {p[0]}-{p[1]:<4} forms {count[p]:>2}/{total}{mark}")
    lines.append("")

    # Relative merge height per condition and method (* = pair forms).
    for cond in CONDITIONS:
        r = results[cond]
        lines.append(f"=== {CONDITION_LABELS[cond]} ===")
        if r["skipped"]:
            lines.append(f"  (skipped: {r['skipped']})")
        lines.append("  " + "pair".ljust(9) +
                     "".join(m[:9].ljust(11) for m in r["measures"]))
        for pair in pairs:
            row = "  " + f"{pair[0]}-{pair[1]}".ljust(9)
            for m in r["measures"]:
                h, direct = r["measures"][m][pair]
                row += (f"{h:.2f}{'*' if direct else ' '}").ljust(11)
            lines.append(row)
        lines.append("")

    text = "\n".join(lines)
    (DATA_DIR / "resultados_figSI_pares.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# =================================================================
# Main
# =================================================================

def set_seeds(seed=SEED):
    """Fix the Python, NumPy and TensorFlow seeds (see SEED above)."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TF:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()


def main():
    set_seeds()
    if not HAS_TF:
        print("WARNING: tensorflow not installed -> autoencoder skipped")
    if not HAS_UMAP:
        print("WARNING: umap-learn not installed -> UMAP skipped")
    results = analyse()
    pairs, count = pairs_to_show(results)
    write_report(results, pairs, count)
    plot_figure(results, pairs)


if __name__ == "__main__":
    main()
