"""
Weekly editing activity per thematic category for eleven Wikipedia
language editions.

For each edition, every edit is converted to local time, assigned to one
of the 168 hours of the week and to its thematic categories, and counted
per year. The figure shows, for each category, the mean over years of the
number of edits (Fig. 1) or of distinct editors (Fig. 2) at each hour of
the week.

Input:
    One text file per edition in the data folder, named '<code>.txt' or
    '<code>_*.txt' (e.g. 'es.txt', 'ar_wp.txt'), with one edit per line:
        title user <unix_timestamp> ['category1', 'category2', ...]

Output:
    figuras/fig_all_edits.pdf or figuras/fig_all_editors.pdf

Usage:
    python 1_fig1_2_all_languages.py edits
    python 1_fig1_2_all_languages.py editors
    python 1_fig1_2_all_languages.py edits /path/to/data
"""

import sys
import re
import os
import gc
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# Configuration
# =================================================================

# Order in which the editions appear in the figure (row by row).
PANEL_ORDER = ['ar', 'de', 'hu', 'zh', 'pt', 'es', 'fr', 'vi', 'ru', 'ja', 'it']

# Local time zone of each edition, as IANA identifiers (Table 1).
# Using IANA names means daylight saving time is applied automatically.
TIMEZONE_DICT = {
    'ar': 'Asia/Riyadh',
    'de': 'Europe/Berlin',
    'es': 'Europe/Madrid',
    'fr': 'Europe/Paris',
    'hu': 'Europe/Budapest',
    'it': 'Europe/Rome',
    'ja': 'Asia/Tokyo',
    'pt': 'Europe/Lisbon',
    'ru': 'Europe/Moscow',
    'vi': 'Asia/Ho_Chi_Minh',
    'zh': 'Asia/Shanghai',
}

# The 13 thematic categories analysed, and their column index.
CATEGORY_ORDER = [
    'art', 'events', 'games', 'geography', 'health', 'history', 'mathematics',
    'nature', 'philosophy', 'politics', 'religion', 'rights', 'sports',
]
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORY_ORDER)}

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

# Colour of each category in the figure.
COLOR_DICT = {
    'art': '#1f77b4', 'events': '#d62728', 'games': '#7f7f7f',
    'geography': '#9FE2BF', 'health': '#F5E11B', 'history': '#e377c2',
    'mathematics': '#bcbd22', 'nature': '#17becf', 'philosophy': '#9467bd',
    'politics': '#873600', 'religion': '#000000', 'rights': '#2ca02c',
    'sports': '#ff7f0e',
}

# Regular expressions to parse one line of input and its category list.
LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\d+)\s+\[(.*)\]\s*$")
CATS_RE = re.compile(r"'([^']+)'")

HOURS_PER_WEEK = 168


# =================================================================
# Data loading
# =================================================================

def find_file(data_dir, code):
    """
    Locate the data file of a language edition.

    Args:
        data_dir: folder containing the data files.
        code: two-letter language code (e.g. 'es').

    Returns:
        Path to '<code>.txt' if it exists, otherwise to the first
        '<code>_*.txt' found in alphabetical order.
    """
    exact = os.path.join(data_dir, f"{code}.txt")
    if os.path.exists(exact):
        return exact
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith(f"{code}_") and fname.endswith(".txt"):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No file found for '{code}' in {data_dir}")


def new_year_grid(metric, n_cats):
    """
    Create an empty (168, n_cats) accumulator for one year.

    For 'edits' it holds integer counts; for 'editors' it holds a set of
    user names per cell, so that each editor is counted only once.
    """
    if metric == 'edits':
        return np.zeros((HOURS_PER_WEEK, n_cats), dtype=np.int64)
    grid = np.empty((HOURS_PER_WEEK, n_cats), dtype=object)
    for r in range(HOURS_PER_WEEK):
        for c in range(n_cats):
            grid[r, c] = set()
    return grid


def build_yearly_matrix(path, timezone_name, metric, progress_every=2_000_000):
    """
    Read one edition and count its activity per year, hour of the week
    and category.

    Args:
        path: path to the edition's data file.
        timezone_name: IANA time zone used to convert timestamps.
        metric: 'edits' (number of edits) or 'editors' (distinct editors).
        progress_every: print progress every this many lines (0 to disable).

    Returns:
        Array of shape (n_years, 168, n_categories).
    """
    tz = ZoneInfo(timezone_name)
    n_cats = len(CATEGORY_ORDER)
    year_data = {}
    n_lines = n_ok = n_skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                print(f"    ...{n_lines:,} lines")

            # Parse the line; skip it if malformed or without categories.
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                n_skipped += 1
                continue
            _title, user, ts_str, cats_raw = m.groups()
            try:
                ts = int(ts_str)
            except ValueError:
                n_skipped += 1
                continue
            cats = CATS_RE.findall(cats_raw)
            if not cats:
                n_skipped += 1
                continue

            # Convert the UTC timestamp to local time and find the hour of
            # the week (0 = Monday 00h, 167 = Sunday 23h).
            dt_local = datetime.fromtimestamp(ts, tz=dt_timezone.utc).astimezone(tz)
            week_hour = dt_local.weekday() * 24 + dt_local.hour
            year = dt_local.year

            if year not in year_data:
                year_data[year] = new_year_grid(metric, n_cats)

            # Add the edit to every analysed category it belongs to.
            cell = year_data[year]
            hit = False
            for cat_raw in cats:
                cat = TRANSLATION_DICT.get(cat_raw, cat_raw)
                idx = CATEGORY_INDEX.get(cat)
                if idx is None:
                    continue
                if metric == 'edits':
                    cell[week_hour, idx] += 1
                else:
                    cell[week_hour, idx].add(user)
                hit = True
            if hit:
                n_ok += 1
            else:
                n_skipped += 1

    if not year_data:
        raise ValueError(f"No valid lines in {path}")

    min_year, max_year = min(year_data), max(year_data)
    n_years = max_year - min_year + 1
    print(f"    lines={n_lines:,} ok={n_ok:,} skipped={n_skipped:,} "
          f"| years={n_years} ({min_year}-{max_year})")

    # Stack the yearly grids into a single array; for 'editors', replace
    # each set of users by its size.
    all_years = np.zeros((n_years, HOURS_PER_WEEK, n_cats), dtype=np.float64)
    for yr, cell in year_data.items():
        if metric == 'edits':
            all_years[yr - min_year] = cell
        else:
            counts = np.zeros((HOURS_PER_WEEK, n_cats), dtype=np.int64)
            for r in range(HOURS_PER_WEEK):
                for c in range(n_cats):
                    counts[r, c] = len(cell[r, c])
            all_years[yr - min_year] = counts

    # Free the memory used by the sets before the next edition.
    del year_data
    gc.collect()
    return all_years


# =================================================================
# Statistics
# =================================================================

def mean_sem(all_years):
    """
    Mean and standard error of the mean across years.

    Args:
        all_years: array of shape (n_years, 168, n_categories).

    Returns:
        Tuple (mean, sem), each of shape (168, n_categories).
    """
    n = all_years.shape[0]
    mean = all_years.mean(axis=0)
    sem = all_years.std(axis=0) / np.sqrt(n) if n > 0 else all_years.std(axis=0)
    return mean, sem


# =================================================================
# Plotting
# =================================================================

def plot_edition(ax, code, mean, sem, ylabel):
    """
    Draw the weekly profile of every category of one edition.

    A +1 SEM band is shown only for the three most active categories,
    to keep the panel readable.
    """
    hours = np.arange(HOURS_PER_WEEK)
    totals = mean.sum(axis=0)
    top3 = {CATEGORY_ORDER[i] for i in np.argsort(totals)[-3:]}

    for i, category in enumerate(CATEGORY_ORDER):
        color = COLOR_DICT.get(category, 'black')
        line_mean = mean[:, i]
        if category in top3:
            ax.fill_between(hours, line_mean, line_mean + sem[:, i],
                            color=color, alpha=0.2, linewidth=0)
        ax.plot(hours, line_mean, linestyle='-', color=color, label=category)

    # Axis labels and one tick every six hours.
    ax.set_title(code)
    ax.set_xlabel('Hour of the week')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_xticks(np.arange(0, HOURS_PER_WEEK, 6))
    ax.set_xticklabels(
        [f'{d} {h:02d}h' for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
         for h in range(0, 24, 6)],
        rotation=90,
    )


def add_category_legend(ax):
    """Use an empty panel to show the colour legend of the categories."""
    ax.axis('off')
    handles = [plt.Line2D([0], [0], color=COLOR_DICT[c], lw=2) for c in CATEGORY_ORDER]
    ax.legend(handles, CATEGORY_ORDER, loc='center', title='Categories', ncol=2)


# =================================================================
# Main
# =================================================================

def main():
    # Read the metric and, optionally, the data folder from the command line.
    args = sys.argv[1:]
    metric = args[0] if len(args) >= 1 else None
    data_dir = args[1] if len(args) >= 2 else "."
    if metric not in ("edits", "editors"):
        print("Usage: python 1_fig1_2_all_languages.py [edits|editors] [data folder]")
        sys.exit(1)

    ylabel = "Number of edits" if metric == 'edits' else "Number of editors"

    # Figure layout: 6 x 2 panels on an A4 page, with font sizes adapted
    # to that scale.
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 9,
        'axes.labelsize': 7,
        'xtick.labelsize': 4.5,
        'ytick.labelsize': 5.5,
        'legend.fontsize': 6.5,
        'legend.title_fontsize': 7.5,
        'lines.linewidth': 0.8,
    })
    fig, axes = plt.subplots(6, 2, figsize=(8.27, 11.69))

    # One panel per edition. Editions are processed one at a time to keep
    # memory usage low.
    for panel_i, code in enumerate(PANEL_ORDER):
        path = find_file(data_dir, code)
        print(f"[{code}] {os.path.basename(path)}")
        all_years = build_yearly_matrix(path, TIMEZONE_DICT[code], metric)
        mean, sem = mean_sem(all_years)
        plot_edition(axes.flat[panel_i], code, mean, sem, ylabel)
        del all_years
        gc.collect()

    # The twelfth panel holds the legend.
    add_category_legend(axes.flat[11])

    # Save the figure as PDF.
    plt.tight_layout()
    os.makedirs("figuras", exist_ok=True)
    out_path = os.path.join("figuras", f"fig_all_{metric}.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"\nFigure saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
