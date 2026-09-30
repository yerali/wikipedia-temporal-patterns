"""
Temporal tensor and the two normalisations used in the temporal analysis.

This module builds, for each metric (edits or editors), the tensor
    11 editions x 13 categories x 168 hours of the week
over the whole data span of each edition, and defines the two
normalisations used in Figs. 5 and 6 and in the Supplementary
Information:

    "composition": each edition is divided by its total activity (its
                   2,184 cells add up to one). The size of the edition is
                   removed and the relative weight of the categories is
                   kept, so the representation reflects thematic
                   composition together with timing.
    "timing":      each (edition, category) block is divided by its own
                   sum over the 168 hours. The size of the edition and the
                   weight of each category are removed, so only the shape
                   of the weekly cycle remains.

The module is imported by the figure scripts. Run on its own, it also
compares the dendrograms obtained with the two normalisations:
    - agreement between them (adjusted Rand index at k = 3 and k = 5, and
      cophenetic correlation), against a null model in which the 168 hours
      are permuted within each (edition, category) block, which destroys
      timing and keeps composition;
    - share of the distance due to composition and to timing under the
      "composition" normalisation.

Input:
    One text file per edition, in the same folder as this script, named
    '<code>.txt' or '<code>_*.txt', with one edit per line:
        title user <unix_timestamp> ['category1', 'category2', ...]
    Each edit is counted once for every category its page belongs to.
    The tensors are cached in tensor_cat_edits.csv and
    tensor_cat_editors.csv; use --recompute to rebuild them.

Output (when run on its own):
    figuras/dendrogramas_dos_normalizaciones.pdf
    resultados_dos_normalizaciones.txt

Usage:
    python dendrogramas_dos_normalizaciones.py
    python dendrogramas_dos_normalizaciones.py --recompute
"""

import os
import re
import sys
import gc
from pathlib import Path
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet


# =================================================================
# Parameters
# =================================================================

# Minimum total count for an (edition, category) block to be included.
# 0 keeps every non-empty block.
MIN_COUNT = 0

KS = [3, 5]              # dendrogram cuts used for the adjusted Rand index
NULL_REPLICAS = 200      # number of permutations of the null model
METRICS = ["edits", "editors"]

# Normalisation modes and the title of their panel.
NORMALISATIONS = [
    ("normalised by total activity\n(composition + timing)", "composition"),
    ("each category normalised to unit sum\n(timing only)", "timing"),
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


def parse_line(line):
    """
    Parse one line of input.

    Returns:
        Tuple (user, timestamp, categories), keeping only the analysed
        categories, or None if the line is malformed or has none of them.
    """
    m = LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    _title, user, ts_str, cats_raw = m.groups()
    try:
        ts = int(ts_str)
    except ValueError:
        return None
    cats = [TRANSLATION_DICT.get(c, c) for c in CATS_RE.findall(cats_raw)]
    cats = [c for c in cats if c in CAT_INDEX]
    if not cats:
        return None
    return user, ts, cats


def week_hour(ts, tz):
    """Hour of the week in local time (0 = Monday 00h, 167 = Sunday 23h)."""
    dt_local = datetime.fromtimestamp(ts, tz=dt_timezone.utc).astimezone(tz)
    return dt_local.weekday() * 24 + dt_local.hour


def build_edits(path, lang, progress_every=2_000_000):
    """
    Count the edits of one edition per category and hour of the week.

    Each edit is counted once in every category its page belongs to.

    Returns:
        Integer array of shape (13, 168).
    """
    tz = ZoneInfo(TIMEZONE_DICT[lang])
    M = np.zeros((len(CATEGORIES), HOURS_PER_WEEK), dtype=np.int64)
    n_lines = n_ok = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    ...{n_lines:,} lines")
            parsed = parse_line(line)
            if parsed is None:
                continue
            _user, ts, cats = parsed
            wh = week_hour(ts, tz)
            for c in cats:
                M[CAT_INDEX[c], wh] += 1
            n_ok += 1
    print(f"    lines={n_lines:,} valid={n_ok:,} total={M.sum():,}")
    return M


def build_editors(path, lang, progress_every=2_000_000):
    """
    Count the distinct editors of one edition per category and hour of
    the week.

    A set of user names is kept for each of the 13 x 168 cells, so that
    each editor is counted only once per cell.

    Returns:
        Integer array of shape (13, 168).
    """
    tz = ZoneInfo(TIMEZONE_DICT[lang])
    seen = [[set() for _ in range(HOURS_PER_WEEK)] for _ in CATEGORIES]
    n_lines = n_ok = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    ...{n_lines:,} lines")
            parsed = parse_line(line)
            if parsed is None:
                continue
            user, ts, cats = parsed
            wh = week_hour(ts, tz)
            for c in cats:
                seen[CAT_INDEX[c]][wh].add(user)
            n_ok += 1
    M = np.array([[len(s) for s in row] for row in seen], dtype=np.int64)
    print(f"    lines={n_lines:,} valid={n_ok:,} total={M.sum():,}")
    # Free the memory used by the sets before the next edition.
    del seen
    gc.collect()
    return M


def load_tensor(metric, recompute=False):
    """
    Return the tensor of one metric over the whole data span.

    The tensor is read from the cache if it exists, unless recompute is
    True; otherwise it is built from the raw files and cached.

    Args:
        metric: 'edits' or 'editors'.
        recompute: rebuild the tensor from the raw files.

    Returns:
        Float array of shape (11, 13, 168).
    """
    cache = DATA_DIR / f"tensor_cat_{metric}.csv"
    if cache.exists() and not recompute:
        print(f"[{metric}] using cache: {cache.name}")
        table = pd.read_csv(cache, index_col=[0, 1])
        T = np.stack([table.loc[lang].reindex(CATEGORIES).values
                      for lang in LANGUAGES])
        return T.astype(float)

    builder = build_edits if metric == "edits" else build_editors
    blocks = {}
    for lang in LANGUAGES:
        path = find_file(DATA_DIR, lang)
        print(f"[{metric}] {lang} ({TIMEZONE_DICT[lang]}): {os.path.basename(path)}")
        blocks[lang] = builder(path, lang)

    # Save as a table with index (language, category) and 168 columns.
    table = pd.concat(
        {lang: pd.DataFrame(blocks[lang], index=CATEGORIES, columns=HOUR_LABELS)
         for lang in LANGUAGES},
        names=["language", "category"])
    table.to_csv(cache)
    print(f"[{metric}] cache written: {cache.name}")
    return np.stack([blocks[lang] for lang in LANGUAGES]).astype(float)


# =================================================================
# Normalisations
# =================================================================

def block_mask(T, min_count=MIN_COUNT):
    """
    Mark the (edition, category) blocks included in the analysis.

    A block is included if it is not empty and reaches min_count. The same
    mask is applied to both normalisations, so that they use exactly the
    same set of cells.

    Returns:
        Boolean array of shape (11, 13).
    """
    total = T.sum(axis=2)
    return (total > 0) & (total >= min_count)


def normalise(T, mode, mask):
    """
    Normalise the tensor and flatten it for clustering.

    Args:
        T: array of shape (11, 13, 168).
        mode: 'composition' (divide each edition by its total activity) or
              'timing' (divide each block by its sum over the 168 hours).
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
    total[total == 0] = 1.0          # empty blocks stay at zero
    X = X / total
    return X.reshape(X.shape[0], -1)


def ward(X):
    """Hierarchical clustering with Ward's linkage and Euclidean distance."""
    return linkage(X, method="ward")


# =================================================================
# Agreement between the two dendrograms
# =================================================================

AGREEMENT_KEYS = [f"ARI_k{k}" for k in KS] + ["cophenetic"]


def agreement(Z1, Z2):
    """
    Agreement between two dendrograms: adjusted Rand index of the
    partitions at each k in KS, and correlation of cophenetic distances.
    """
    result = {f"ARI_k{k}": adjusted_rand_score(fcluster(Z1, k, criterion="maxclust"),
                                               fcluster(Z2, k, criterion="maxclust"))
              for k in KS}
    result["cophenetic"] = float(np.corrcoef(cophenet(Z1), cophenet(Z2))[0, 1])
    return result


def null_agreement(T, mask, n_rep=NULL_REPLICAS, seed=0):
    """
    Agreement expected between the two normalisations without any
    temporal signal.

    In each replica the 168 hours are permuted within every (edition,
    category) block, which destroys timing and keeps composition.

    Returns:
        Dict mapping each agreement measure to an array of n_rep values.
    """
    rng = np.random.default_rng(seed)
    values = {k: [] for k in AGREEMENT_KEYS}
    for _ in range(n_rep):
        P = np.empty_like(T)
        for i in range(T.shape[0]):
            for c in range(T.shape[1]):
                P[i, c] = rng.permutation(T[i, c])
        a = agreement(ward(normalise(P, "composition", mask)),
                      ward(normalise(P, "timing", mask)))
        for k in AGREEMENT_KEYS:
            values[k].append(a[k])
    return {k: np.array(v) for k, v in values.items()}


# =================================================================
# Composition versus timing
# =================================================================

def composition_timing_split(T, mask):
    """
    Split the distance of the 'composition' normalisation into the part
    due to composition and the part due to timing.

    Each cell is written as share[edition, category] x profile[edition,
    category, hour]. The distance is then computed twice: replacing every
    profile by the mean profile (composition only) and replacing every
    share by the mean share (timing only).

    Returns:
        Tuple (composition_distance, timing_distance, shares), where shares
        has shape (11, 13).
    """
    X = np.where(mask[:, :, None], T, 0.0)
    total = X.sum(axis=(1, 2), keepdims=True)
    total[total == 0] = 1.0
    Xn = X / total
    shares = Xn.sum(axis=2)
    profiles = np.divide(Xn, shares[:, :, None],
                         out=np.zeros_like(Xn), where=shares[:, :, None] > 0)

    composition_only = shares[:, :, None] * profiles.mean(axis=0)[None, :, :]
    timing_only = shares.mean(axis=0)[None, :, None] * profiles

    def total_distance(A):
        """Sum of squared Euclidean distances over all pairs of editions."""
        A = A.reshape(A.shape[0], -1)
        return float(np.sum((A[:, None, :] - A[None, :, :]) ** 2))

    return total_distance(composition_only), total_distance(timing_only), shares


# =================================================================
# Figure
# =================================================================

def plot_dendrograms(results):
    """Draw one dendrogram per metric and normalisation, and save it."""
    fig, axes = plt.subplots(len(METRICS), len(NORMALISATIONS),
                             figsize=(6.2 * len(NORMALISATIONS),
                                      4.6 * len(METRICS)), squeeze=False)
    for r, metric in enumerate(METRICS):
        for c, (title, mode) in enumerate(NORMALISATIONS):
            ax = axes[r][c]
            dendrogram(results[metric]["Z"][mode], labels=LANGUAGES, ax=ax,
                       distance_sort="descending", leaf_rotation=90,
                       leaf_font_size=10)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{metric}\nDistance", fontsize=10)
    fig.suptitle("Ward clustering of 11 editions x 13 categories x 168 hours",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "dendrogramas_dos_normalizaciones.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# =================================================================
# Analysis and report
# =================================================================

def analyse(metric, recompute=False):
    """Run the comparison of the two normalisations for one metric."""
    T = load_tensor(metric, recompute=recompute)
    mask = block_mask(T)
    Z = {mode: ward(normalise(T, mode, mask)) for _, mode in NORMALISATIONS}
    obs = agreement(Z["composition"], Z["timing"])
    print(f"[{metric}] null model ({NULL_REPLICAS} permutations)...")
    null = null_agreement(T, mask)
    comp, timing, shares = composition_timing_split(T, mask)
    return dict(n_excluded=int((~mask).sum()), Z=Z, agreement=obs, null=null,
                comp=comp, timing=timing, shares=shares)


def write_report(results, path):
    """Save and print the results of the comparison."""
    lines = ["TWO NORMALISATIONS - whole data span, 11 x 13 x 168",
             f"MIN_COUNT = {MIN_COUNT}", ""]
    for metric in METRICS:
        r = results[metric]
        lines.append(f"=== {metric} ===")
        lines.append(f"  excluded or empty (edition, category) blocks: "
                     f"{r['n_excluded']} of {len(LANGUAGES) * len(CATEGORIES)}")

        # Mean share of each category, in decreasing order.
        lines.append("  -- mean share of each category --")
        mean_share = r["shares"].mean(axis=0)
        for i in np.argsort(-mean_share):
            lines.append(f"     {CATEGORIES[i]:<12} {mean_share[i]:.4f}")

        # Share of the distance due to composition and to timing.
        total = r["comp"] + r["timing"]
        lines.append("  -- distance split under the 'composition' normalisation --")
        lines.append(f"     composition: {r['comp']:.3e}  ({100 * r['comp'] / total:.1f}%)")
        lines.append(f"     timing     : {r['timing']:.3e}  ({100 * r['timing'] / total:.1f}%)")

        # Observed agreement against the null model.
        lines.append("  -- agreement between the two normalisations --")
        for k in AGREEMENT_KEYS:
            obs = r["agreement"][k]
            null = r["null"][k]
            p = float(np.mean(null >= obs))
            lines.append(f"     {k:<12} observed {obs:.3f} | "
                         f"null {null.mean():.3f} +- {null.std():.3f} | p={p:.3f}")
        lines.append("")
    text = "\n".join(lines)
    Path(path).write_text(text, encoding="utf-8")
    print("\n" + text)


# =================================================================
# Main
# =================================================================

def main():
    recompute = "--recompute" in sys.argv[1:]
    results = {m: analyse(m, recompute=recompute) for m in METRICS}
    write_report(results, DATA_DIR / "resultados_dos_normalizaciones.txt")
    plot_dendrograms(results)


if __name__ == "__main__":
    main()
