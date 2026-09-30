"""
Static clustering of the eleven Wikipedia editions by category (Fig. 4).

Each edition is represented by a 13-dimensional vector with its activity
in each thematic category, accumulated over the whole data span, and the
editions are grouped by hierarchical clustering (Ward's linkage).

Two metrics are computed in a single pass over each data file:
    edits   : each edit is counted once for every category its page
              belongs to (Section 2.1).
    editors : number of distinct editors per category over the whole
              period; an editor with many edits in a category counts once.

Figure layout:
    (a) edits    main panel: normalised shares | inset: raw counts
    (b) editors  main panel: normalised shares | inset: raw counts
In the normalised version each edition is divided by its own total, so
that its thirteen category values add up to one. This removes the size
of the edition and keeps its thematic composition.

Input:
    One text file per edition, in the same folder as this script, named
    '<code>.txt' or '<code>_*.txt', with one edit per line:
        title user <unix_timestamp> ['category1', 'category2', ...]
    Edits made by bots are already excluded from these files.
    The count matrices are cached in fig4_static_edits.csv and
    fig4_static_editors.csv; use --recompute to rebuild them.

Output:
    figuras/fig4_v2.pdf
    resultados_fig4_cuotas.txt  (category shares per edition)

Usage:
    python 3_fig4.py
    python 3_fig4.py --recompute
"""

import os
import re
import sys
import gc
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =================================================================
# Configuration
# =================================================================

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "figuras"

# Same edition order as in the other scripts, so that dendrograms can be
# compared directly.
LANGUAGES = ['ar', 'de', 'hu', 'zh', 'pt', 'es', 'fr', 'vi', 'ru', 'ja', 'it']

# The 13 thematic categories analysed.
CATEGORIES = ['art', 'events', 'games', 'geography', 'health', 'history',
              'mathematics', 'nature', 'philosophy', 'politics',
              'religion', 'rights', 'sports']

# Category labels that appear in Spanish or Portuguese in the raw data,
# mapped to their English name.
TRANSLATION_DICT = {
    # Spanish
    'arte': 'art', 'deporte': 'sports', 'derecho': 'rights', 'eventos': 'events',
    'filosofia': 'philosophy', 'geografia': 'geography', 'historia': 'history',
    'juegos': 'games', 'matematica': 'mathematics', 'naturaleza': 'nature',
    'politica': 'politics', 'salud': 'health',
    # Portuguese
    'desportos': 'sports', 'direito': 'rights', 'jogos': 'games',
    'natureza': 'nature', 'religiao': 'religion', 'saude': 'health',
}

# Line parser. It is the same one used by the temporal analysis, so that
# the static and temporal representations count exactly the same lines.
LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\d+)\s+\[(.*)\]\s*$")
CATS_RE = re.compile(r"'([^']+)'")

# Cache files for the two count matrices.
CACHE = {"edits": DATA_DIR / "fig4_static_edits.csv",
         "editors": DATA_DIR / "fig4_static_editors.csv"}

# Totals reported in Table 1 of the manuscript, used for sanity checks.
TABLE1_EDITS = {'ar': 7674946, 'de': 39689683, 'es': 55266094, 'fr': 58325570,
                'hu': 6666339, 'it': 22200807, 'ja': 24584471, 'pt': 37853332,
                'ru': 14199597, 'vi': 3657770, 'zh': 12618302}
TABLE1_EDITORS = {'ar': 23641, 'de': 357575, 'es': 2682096, 'fr': 203042,
                  'hu': 148067, 'it': 97162, 'ja': 126639, 'pt': 1475229,
                  'ru': 75929, 'vi': 12259, 'zh': 76555}

# Figure appearance.
FIGSIZE = (14, 6)
MAIN_LEAF_FONTSIZE = 11
INSET_WIDTH, INSET_HEIGHT = "26%", "22%"
INSET_LOC = "upper left"
INSET_ALPHA = 0.85
INSET_LEAF_FONTSIZE = 6
ANNOTATE_HEIGHTS = True


# =================================================================
# Data loading
# =================================================================

def find_file(data_dir, code):
    """
    Locate the data file of a language edition.

    Returns '<code>.txt' if it exists, otherwise the first '<code>_*.txt'
    found in alphabetical order.
    """
    exact = os.path.join(data_dir, f"{code}.txt")
    if os.path.exists(exact):
        return exact
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith(f"{code}_") and fname.endswith(".txt"):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No file found for '{code}' in {data_dir}")


def count_edition(path, lang, progress_every=2_000_000):
    """
    Count the edits and distinct editors per category of one edition.

    Args:
        path: path to the edition's data file.
        lang: language code, used only in the progress messages.
        progress_every: print progress every this many lines (0 to disable).

    Returns:
        Tuple (n_edits, n_editors), two dicts mapping each category to
        its number of edits and of distinct editors.
    """
    n_edits = {c: 0 for c in CATEGORIES}
    users = {c: set() for c in CATEGORIES}
    n_lines = n_ok = n_skip = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    ...{n_lines:,} lines")

            # Parse the line; skip it if malformed.
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                n_skip += 1
                continue
            _title, user, _ts, cats_raw = m.groups()

            # Keep only the analysed categories, translated to English.
            cats = [TRANSLATION_DICT.get(c, c) for c in CATS_RE.findall(cats_raw)]
            cats = [c for c in cats if c in n_edits]
            if not cats:
                n_skip += 1
                continue

            # Count the edit once per category and record its editor.
            for c in cats:
                n_edits[c] += 1
                users[c].add(user)
            n_ok += 1

    n_editors = {c: len(users[c]) for c in CATEGORIES}

    print(f"    [{lang}] lines={n_lines:,} valid={n_ok:,} skipped={n_skip:,}")

    # Free the memory used by the sets of users before the next edition.
    users.clear()
    gc.collect()
    return n_edits, n_editors


def build_matrices(recompute=False):
    """
    Build the 11 x 13 matrices of edits and editors.

    The matrices are read from the cache if it exists, unless recompute
    is True; otherwise they are computed from the raw files and cached.

    Returns:
        Dict with two DataFrames, 'edits' and 'editors' (rows: editions,
        columns: categories).
    """
    if not recompute and all(p.exists() for p in CACHE.values()):
        print("Using cache:", ", ".join(p.name for p in CACHE.values()))
        return {m: pd.read_csv(CACHE[m], index_col="language").reindex(LANGUAGES)
                for m in CACHE}

    rows = {"edits": {}, "editors": {}}
    for lang in LANGUAGES:
        path = find_file(str(DATA_DIR), lang)
        print(f"[{lang}] {os.path.basename(path)}")
        edits, editors = count_edition(path, lang)
        rows["edits"][lang], rows["editors"][lang] = edits, editors

    dfs = {}
    for m in ("edits", "editors"):
        df = pd.DataFrame(rows[m]).T[CATEGORIES].reindex(LANGUAGES)
        df.index.name = "language"
        df.to_csv(CACHE[m])
        dfs[m] = df
        print(f"Cache written: {CACHE[m].name}")
    return dfs


# =================================================================
# Sanity checks
# =================================================================

def sanity_checks(dfs):
    """
    Print consistency checks of the count matrices.

    1. Total edits per edition compared with Table 1. The total here is
       larger, since a page in N categories is counted N times.
    2. The largest number of editors in any category must not exceed the
       total number of editors of the edition in Table 1.
    3. If tensor_cat_edits.csv (temporal analysis) exists, summing its
       168 hours must reproduce the static matrix of edits exactly.
    """
    print("\n--- Total edits compared with Table 1 ---")
    for lang in LANGUAGES:
        tot = int(dfs["edits"].loc[lang].sum())
        t1 = TABLE1_EDITS[lang]
        print(f"  {lang}: {tot:>12,}  Table 1: {t1:>12,}  "
              f"ratio {tot / max(t1, 1):.2f}")

    print("\n--- Editors: maximum per category <= total ---")
    for lang in LANGUAGES:
        mx = int(dfs["editors"].loc[lang].max())
        t1 = TABLE1_EDITORS[lang]
        status = 'OK' if mx <= t1 else 'CHECK: max > total'
        print(f"  {lang}: max={mx:>9,}  Table 1={t1:>9,}  [{status}]")

    tensor_file = DATA_DIR / "tensor_cat_edits.csv"
    if not tensor_file.exists():
        print("\n(tensor_cat_edits.csv not found: cross-check skipped)")
        return
    print("\n--- Cross-check with the temporal tensor (edits only) ---")
    tensor = pd.read_csv(tensor_file, index_col=[0, 1])
    summed = tensor.sum(axis=1).unstack(level=1).reindex(index=LANGUAGES,
                                                         columns=CATEGORIES)
    diff = (summed - dfs["edits"]).abs()
    rel = diff.values.sum() / max(dfs["edits"].values.sum(), 1)
    if rel < 1e-9:
        print("  Exact match. OK")
    else:
        print(f"  Mismatch (relative difference {rel:.3e}).")
        print(diff.stack().sort_values(ascending=False).head(5).to_string())


# =================================================================
# Figure
# =================================================================

def shares(df):
    """Divide each edition by its total, so that its row adds up to one."""
    total = df.sum(axis=1).replace(0, 1)
    return df.div(total, axis=0)


def annotate_heights(ax, d):
    """Write the height of each merge on the dendrogram."""
    for ic, dc, col in zip(d["icoord"], d["dcoord"], d["color_list"]):
        x, y = 0.5 * sum(ic[1:3]), dc[1]
        if y > 0:
            ax.plot(x, y, "o", ms=3, c=col)
            ax.annotate(f"{y:.3g}", (x, y), xytext=(0, -8),
                        textcoords="offset points", va="top", ha="center",
                        fontsize=6)


def draw_panel(ax, df, title):
    """
    Draw one panel: dendrogram of the normalised shares in the main axes
    and dendrogram of the raw counts in an inset.
    """
    labels = df.index.tolist()
    Z_shares = linkage(shares(df).values, method="ward")
    Z_raw = linkage(df.values.astype(float), method="ward")

    # Main panel: normalised data.
    d = dendrogram(Z_shares, labels=labels, ax=ax, distance_sort="descending",
                   leaf_rotation=90, leaf_font_size=MAIN_LEAF_FONTSIZE)
    if ANNOTATE_HEIGHTS:
        annotate_heights(ax, d)
    ax.set_ylabel("Distance", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")

    # Inset: raw counts.
    ins = inset_axes(ax, width=INSET_WIDTH, height=INSET_HEIGHT,
                     loc=INSET_LOC, borderpad=1.2)
    dendrogram(Z_raw, labels=labels, ax=ins, distance_sort="descending",
               leaf_rotation=90, leaf_font_size=INSET_LEAF_FONTSIZE)
    ins.tick_params(labelsize=INSET_LEAF_FONTSIZE)
    ins.patch.set_alpha(INSET_ALPHA)


# =================================================================
# Report
# =================================================================

def write_report(dfs):
    """Save and print the category shares of every edition."""
    lines = ["FIGURE 4 - category shares per edition "
             "(fraction of total activity)", ""]
    for metric, df in dfs.items():
        q = shares(df)
        lines.append(f"=== {metric} ===")
        lines.append(q.round(4).to_string())
        lines.append("  mean per category:")
        lines.append("  " + q.mean().round(4).to_string().replace("\n", "\n  "))
        lines.append("")
    text = "\n".join(lines)
    (DATA_DIR / "resultados_fig4_cuotas.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# =================================================================
# Main
# =================================================================

def main():
    dfs = build_matrices(recompute="--recompute" in sys.argv[1:])
    sanity_checks(dfs)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_panel(axes[0], dfs["edits"], "(a) Edits")
    draw_panel(axes[1], dfs["editors"], "(b) Editors")
    # subplots_adjust instead of tight_layout, which is not compatible
    # with inset axes.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12, wspace=0.22)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig4_v2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    write_report(dfs)


if __name__ == "__main__":
    main()
