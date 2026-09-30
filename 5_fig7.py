"""
Language pairs in the static and in the temporal clustering (Fig. 7).

For each metric (edits, editors), the static clustering (category shares,
Fig. 4) is compared with the two temporal representations of Figs. 5-6,
all without dimensionality reduction:

    static vs timing only           the comparison of interest: the
                                    timing-only representation carries no
                                    information on thematic composition,
                                    so any difference between the two can
                                    only come from the temporal dimension.
    static vs composition + timing  reference: the category shares are
                                    recovered by summing the weekly
                                    profiles of this representation, so
                                    the two are expected to agree closely.

For every clustering, all 55 possible pairs of editions are evaluated. A
pair "forms" when the two editions join each other directly in a merge,
before either of them has joined any other edition. For each pair the
figure shows whether it forms and the height at which the two editions
first meet, relative to the root of the dendrogram.

The script is self-contained: the static vectors and the temporal tensor
are built in a single pass over the raw files, so that both sides count
exactly the same lines.

Counting:
    edits   : each edit is counted once for every category its page
              belongs to (Section 2.1).
    editors : distinct editors per (category, hour) for the temporal
              tensor, and per category over the whole period for the
              static vector. The latter is not the sum of the former,
              since that would count an editor once for every hour in
              which they were active.

Input:
    One text file per edition, in the same folder as this script, named
    '<code>.txt' or '<code>_*.txt', with one edit per line:
        title user <unix_timestamp> ['category1', 'category2', ...]
    Edits made by bots are already excluded from these files.
    The results of the parsing are cached in fig7_tensor_<metric>.csv and
    fig7_static_<metric>.csv; use --recompute to rebuild them.

Output:
    figuras/fig7_static_vs_temporal.pdf
    resultados_fig7_static_vs_temporal.txt
    resultados_fig7_static_vs_temporal.csv

Usage:
    python 5_fig7.py
    python 5_fig7.py --recompute
"""

import os
import re
import sys
import gc
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage


# =================================================================
# Parameters
# =================================================================

METRICS = ["edits", "editors"]

# Pairs discussed in the main text; their labels are shown in bold.
MANUSCRIPT_PAIRS = [("ru", "zh"), ("es", "pt"), ("vi", "ar"), ("de", "fr")]

# Row order of the figure: pairs listed here come first, in this order;
# any other pair follows in alphabetical order.
PAIR_ORDER = ["pt-es", "zh-ru", "fr-it", "de-hu", "de-fr", "ar-vi",
              "ar-ja", "ar-pt", "zh-ja", "ar-es", "ru-ja", "ar-ru",
              "hu-fr", "zh-vi", "es-it"]

# Cell colours: light grey when the pair does not form, blue when it does.
BINARY_COLORS = ListedColormap(["#f2f2f2", "#2c7fb8"])

# Comparisons shown in the figure: (title, temporal normalisation).
COMPARISONS = [
    ("static (shares) vs timing only", "timing"),
    ("static (shares) vs composition + timing", "composition"),
]


# =================================================================
# Data configuration
# =================================================================

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "figuras"

LANGUAGES = ['ar', 'de', 'hu', 'zh', 'pt', 'es', 'fr', 'vi', 'ru', 'ja', 'it']

# Local time zone of each edition, as IANA identifiers (Table 1).
# Using IANA names means daylight saving time is applied automatically.
TIMEZONE_DICT = {
    'ar': 'Asia/Riyadh', 'de': 'Europe/Berlin', 'es': 'Europe/Madrid',
    'fr': 'Europe/Paris', 'hu': 'Europe/Budapest', 'it': 'Europe/Rome',
    'ja': 'Asia/Tokyo', 'pt': 'Europe/Lisbon',
    'ru': 'Europe/Moscow', 'vi': 'Asia/Ho_Chi_Minh', 'zh': 'Asia/Shanghai',
}

# The 13 thematic categories analysed, and their row index.
CATEGORIES = ['art', 'events', 'games', 'geography', 'health', 'history',
              'mathematics', 'nature', 'philosophy', 'politics',
              'religion', 'rights', 'sports']
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

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

HOURS_PER_WEEK = 168
DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
HOUR_LABELS = [f"{d} {h:02d}h" for d in DAY_NAMES for h in range(24)]

# Regular expressions to parse one line of input and its category list.
LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\d+)\s+\[(.*)\]\s*$")
CATS_RE = re.compile(r"'([^']+)'")

ALL_PAIRS = list(combinations(LANGUAGES, 2))


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


def process_edition(path, lang, metric, progress_every=2_000_000):
    """
    Read one edition and build its temporal and static counts.

    Args:
        path: path to the edition's data file.
        lang: language code.
        metric: 'edits' or 'editors'.
        progress_every: print progress every this many lines (0 to disable).

    Returns:
        Tuple (tensor, static): integer arrays of shape (13, 168) and (13,).
    """
    tz = ZoneInfo(TIMEZONE_DICT[lang])
    n_cats = len(CATEGORIES)
    T = np.zeros((n_cats, HOURS_PER_WEEK), dtype=np.int64)
    if metric == "editors":
        # One set of users per (category, hour) for the temporal tensor,
        # and one per category for the static vector.
        users_by_hour = [[set() for _ in range(HOURS_PER_WEEK)] for _ in range(n_cats)]
        users_by_cat = [set() for _ in range(n_cats)]
    n_lines = n_ok = n_skip = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    ...{n_lines:,} lines")

            # Parse the line; skip it if malformed or without analysed
            # categories.
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                n_skip += 1
                continue
            _title, user, ts_str, cats_raw = m.groups()
            try:
                ts = int(ts_str)
            except ValueError:
                n_skip += 1
                continue
            cats = [TRANSLATION_DICT.get(c, c) for c in CATS_RE.findall(cats_raw)]
            idxs = [CAT_INDEX[c] for c in cats if c in CAT_INDEX]
            if not idxs:
                n_skip += 1
                continue

            # Hour of the week in local time (0 = Monday 00h).
            dt_local = datetime.fromtimestamp(ts, tz=dt_timezone.utc).astimezone(tz)
            wh = dt_local.weekday() * 24 + dt_local.hour

            for i in idxs:
                if metric == "edits":
                    T[i, wh] += 1
                else:
                    users_by_hour[i][wh].add(user)
                    users_by_cat[i].add(user)
            n_ok += 1

    if metric == "edits":
        static = T.sum(axis=1)
    else:
        T = np.array([[len(s) for s in row] for row in users_by_hour],
                     dtype=np.int64)
        static = np.array([len(s) for s in users_by_cat], dtype=np.int64)
        # Free the memory used by the sets before the next edition.
        del users_by_hour, users_by_cat
        gc.collect()

    print(f"    [{lang}] lines={n_lines:,} valid={n_ok:,} skipped={n_skip:,}")
    return T, static


def build(metric, recompute=False):
    """
    Build the temporal tensor and the static matrix of one metric.

    The results are read from the cache if it exists, unless recompute is
    True; otherwise they are computed from the raw files and cached.

    Returns:
        Tuple (tensor of shape (11, 13, 168), DataFrame of shape (11, 13)).
    """
    cache_t = DATA_DIR / f"fig7_tensor_{metric}.csv"
    cache_s = DATA_DIR / f"fig7_static_{metric}.csv"
    if cache_t.exists() and cache_s.exists() and not recompute:
        print(f"[{metric}] using cache: {cache_t.name}, {cache_s.name}")
        table = pd.read_csv(cache_t, index_col=[0, 1])
        T = np.stack([table.loc[lang].reindex(CATEGORIES).values
                      for lang in LANGUAGES])
        static = pd.read_csv(cache_s, index_col="language").reindex(LANGUAGES)
        return T.astype(float), static

    tensors, statics = {}, {}
    for lang in LANGUAGES:
        path = find_file(str(DATA_DIR), lang)
        print(f"[{metric}] {lang} ({TIMEZONE_DICT[lang]}): "
              f"{os.path.basename(path)}")
        tensors[lang], statics[lang] = process_edition(path, lang, metric)

    table = pd.concat(
        {lang: pd.DataFrame(tensors[lang], index=CATEGORIES, columns=HOUR_LABELS)
         for lang in LANGUAGES}, names=["language", "category"])
    table.to_csv(cache_t)
    static = pd.DataFrame({lang: pd.Series(statics[lang], index=CATEGORIES)
                           for lang in LANGUAGES}).T
    static.index.name = "language"
    static.to_csv(cache_s)
    print(f"[{metric}] cache written: {cache_t.name}, {cache_s.name}")
    return np.stack([tensors[lang] for lang in LANGUAGES]).astype(float), static


def check_edits(T, static):
    """For edits, summing the 168 hours must reproduce the static matrix."""
    diff = np.abs(T.sum(axis=2) - static.values).sum()
    rel = diff / max(static.values.sum(), 1)
    if rel < 1e-9:
        print("CHECK edits: static matrix = sum of the tensor. OK")
    else:
        print(f"CHECK edits: mismatch (relative difference {rel:.3e}).")


# =================================================================
# Normalisations
# =================================================================

def block_mask(T, min_count=0):
    """
    Mark the non-empty (edition, category) blocks. The same mask is applied
    to both normalisations, so that they use the same set of cells.
    """
    total = T.sum(axis=2)
    return (total > 0) & (total >= min_count)


def normalise(T, mode, mask):
    """
    Normalise the tensor and flatten it for clustering.

    Args:
        T: array of shape (11, 13, 168).
        mode: 'composition' (each edition divided by its total activity) or
              'timing' (each block divided by its sum over the 168 hours).
        mask: boolean array of shape (11, 13) from block_mask.

    Returns:
        Array of shape (11, 2184).
    """
    X = np.where(mask[:, :, None], T, 0.0)
    if mode == "composition":
        total = X.sum(axis=(1, 2), keepdims=True)
    elif mode == "timing":
        total = X.sum(axis=2, keepdims=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    total = np.where(total == 0, 1.0, total)
    return (X / total).reshape(X.shape[0], -1)


def shares(df):
    """Divide each edition by its total, so that its 13 values add up to one."""
    total = df.sum(axis=1).replace(0, 1)
    return df.div(total, axis=0)


def ward(X):
    """Hierarchical clustering with Ward's linkage and Euclidean distance."""
    return linkage(X, method="ward")


# =================================================================
# Pairs in a dendrogram
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


def direct_pairs(Z):
    """
    Evaluate all 55 pairs in a dendrogram.

    Returns:
        Tuple (pairs, heights): the set of pairs that form, and a dict with
        the relative merge height of every pair.
    """
    pairs, heights = set(), {}
    for pair in ALL_PAIRS:
        h, direct = merge_height(Z, *pair)
        heights[pair] = h
        if direct:
            pairs.add(pair)
    return pairs, heights


# =================================================================
# Analysis
# =================================================================

def analyse(recompute=False):
    """Compare the static and temporal pairs for every metric and comparison."""
    results = {}
    for metric in METRICS:
        T, static = build(metric, recompute=recompute)
        if metric == "edits":
            check_edits(T, static)
        mask = block_mask(T)

        Z_static = ward(shares(static).values)
        static_pairs, static_heights = direct_pairs(Z_static)

        comparisons = {}
        for title, mode in COMPARISONS:
            temporal_pairs, temporal_heights = direct_pairs(ward(normalise(T, mode, mask)))
            comparisons[title] = dict(
                static=static_pairs, temporal=temporal_pairs,
                h_static=static_heights, h_temporal=temporal_heights)
        results[metric] = comparisons
    return results


# =================================================================
# Figure
# =================================================================

def common_rows(results):
    """
    Rows of the figure: the pairs that form in any panel, in a single order
    shared by all panels, so that row i is the same pair everywhere.
    """
    union = set()
    for comparisons in results.values():
        for d in comparisons.values():
            union |= d["static"] | d["temporal"]

    def sort_key(pair):
        name, reverse = f"{pair[0]}-{pair[1]}", f"{pair[1]}-{pair[0]}"
        for i, listed in enumerate(PAIR_ORDER):
            if listed in (name, reverse):
                return (0, i)
        return (1, name)

    return sorted(union, key=sort_key)


def plot_figure(results):
    """Draw one panel per metric and comparison, and save the figure."""
    rows = common_rows(results)
    fig, axes = plt.subplots(len(METRICS), len(COMPARISONS),
                             figsize=(5.0 * len(COMPARISONS),
                                      1.2 + 0.5 * len(rows) * len(METRICS)),
                             squeeze=False)
    for r, metric in enumerate(METRICS):
        for c, (title, _mode) in enumerate(COMPARISONS):
            ax = axes[r][c]
            d = results[metric][title]

            # Cell colour: filled if the pair forms in that analysis.
            B = np.zeros((max(len(rows), 1), 2))
            for i, pair in enumerate(rows):
                B[i, 0] = 1.0 if pair in d["static"] else 0.0
                B[i, 1] = 1.0 if pair in d["temporal"] else 0.0
            ax.imshow(B, vmin=0, vmax=1, cmap=BINARY_COLORS, aspect="auto")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["static", "temporal"], fontsize=8)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{a}-{b}" for a, b in rows], fontsize=8)

            # White lines between cells.
            ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", length=0)

            # Relative merge height written in each cell; pairs discussed in
            # the main text in bold.
            for i, pair in enumerate(rows):
                for j, key in enumerate(("h_static", "h_temporal")):
                    v = d[key].get(pair, np.nan)
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if B[i, j] else "#555555")
                if pair in MANUSCRIPT_PAIRS or pair[::-1] in MANUSCRIPT_PAIRS:
                    ax.get_yticklabels()[i].set_fontweight("bold")

            ax.set_title(f"({chr(97 + r * len(COMPARISONS) + c)}) "
                         f"{metric} — {title}", fontsize=9)

    fig.legend(handles=[Patch(facecolor="#2c7fb8", label="pair"),
                        Patch(facecolor="#f2f2f2", edgecolor="#cccccc",
                              label="no pair")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig7_static_vs_temporal.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# =================================================================
# Report
# =================================================================

def write_report(results):
    """Save and print the pairs of each comparison, as text and as CSV."""
    lines, records = ["FIGURE 7 - static versus temporal clustering (pairs)", ""], []
    fmt = lambda s: [f"{a}-{b}" for a, b in sorted(s)]
    for metric in METRICS:
        lines.append(f"================ {metric} ================")
        for title, _mode in COMPARISONS:
            d = results[metric][title]
            lines.append(f"--- {title} ---")
            lines.append(f"  pairs in the static clustering: {len(d['static'])} | "
                         f"in the temporal clustering: {len(d['temporal'])}")
            lines.append(f"  in both       : {fmt(d['static'] & d['temporal'])}")
            lines.append(f"  static only   : {fmt(d['static'] - d['temporal'])}")
            lines.append(f"  temporal only : {fmt(d['temporal'] - d['static'])}")
            lines.append("")
            for pair in sorted(d["static"] | d["temporal"]):
                records.append(dict(
                    metric=metric, comparison=title,
                    pair=f"{pair[0]}-{pair[1]}",
                    in_static=pair in d["static"],
                    in_temporal=pair in d["temporal"],
                    height_static=d["h_static"][pair],
                    height_temporal=d["h_temporal"][pair]))
    base = DATA_DIR / "resultados_fig7_static_vs_temporal"
    pd.DataFrame(records).to_csv(base.with_suffix(".csv"), index=False)
    text = "\n".join(lines)
    base.with_suffix(".txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# =================================================================
# Main
# =================================================================

def main():
    results = analyse(recompute="--recompute" in sys.argv[1:])
    write_report(results)
    plot_figure(results)


if __name__ == "__main__":
    main()
