"""
Hourly editing activity by language edition (Fig. 3).

For each of the eleven Wikipedia editions, every edit is converted to
local time and counted by hour of the day (0-23), aggregating all
categories and the whole data span. Each edit is counted once, even if
its page belongs to several categories. Spanish (red) and Portuguese
(blue) are highlighted; the remaining editions are drawn in black.

Input:
    One text file per edition, in the same folder as this script, named
    '<code>.txt' or '<code>_*.txt' (e.g. 'es.txt', 'pt_wiki.txt'), with
    one edit per line:
        title user <unix_timestamp> ['category1', 'category2', ...]

Output:
    figuras/fig3_hourly_activity_by_language.pdf

Usage:
    python 2_fig3_hourly_activity_by_language.py
"""

import os
import re
from zoneinfo import ZoneInfo
from datetime import datetime, timezone as dt_timezone

import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# Configuration
# =================================================================

# Editions included in the figure.
LANGUAGES = ['ar', 'de', 'hu', 'zh', 'pt', 'es', 'fr', 'vi', 'ru', 'ja', 'it']

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

# Highlighted editions and their colours; all others use DEFAULT_COLOR.
HIGHLIGHT = {'es': 'red', 'pt': 'blue'}
DEFAULT_COLOR = 'black'

# Only the timestamp (third field) is needed; categories are not parsed.
LINE_RE = re.compile(r"^\S+\s+\S+\s+(\d+)")

# Accepts file names such as 'pt.txt', 'pt_wiki.txt' or 'ar_wp.txt'.
FILE_RE_TEMPLATE = r"^{code}[._]"


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
        Path to the first '.txt' file whose name starts with '<code>.'
        or '<code>_', in alphabetical order.
    """
    pattern = re.compile(FILE_RE_TEMPLATE.format(code=re.escape(code)))
    for fname in sorted(os.listdir(data_dir)):
        if pattern.match(fname) and fname.endswith(".txt"):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No .txt file found for '{code}' in {data_dir}")


def count_by_hour_of_day(path, tz_name):
    """
    Count the edits of one edition by local hour of the day.

    The file is read line by line, so memory use does not depend on its
    size.

    Args:
        path: path to the edition's data file.
        tz_name: IANA time zone used to convert timestamps.

    Returns:
        Integer array of length 24 with the number of edits per hour.
    """
    tz = ZoneInfo(tz_name)
    counts = np.zeros(24, dtype=np.int64)
    skipped = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Extract the timestamp; skip malformed lines.
            m = LINE_RE.match(line)
            if not m:
                skipped += 1
                continue
            # Convert the UTC timestamp to local time and count it.
            ts = int(m.group(1))
            dt_local = datetime.fromtimestamp(ts, tz=dt_timezone.utc).astimezone(tz)
            counts[dt_local.hour] += 1
            total += 1

    if skipped:
        print(f"    Warning: {skipped} lines could not be parsed and were skipped.")
    print(f"    {os.path.basename(path)}: {total} edits counted.")
    return counts


# =================================================================
# Plotting
# =================================================================

def plot_hourly_activity(all_counts, output_dir="figuras"):
    """
    Plot one curve per edition and save the figure as PDF.

    Each curve is labelled at its right end instead of using a legend.

    Args:
        all_counts: dict mapping language code to its 24 hourly counts.
        output_dir: folder where the figure is saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    hours = np.arange(24)

    plt.figure(figsize=(10, 6))
    for code in LANGUAGES:
        color = HIGHLIGHT.get(code, DEFAULT_COLOR)
        counts = all_counts[code]
        is_highlight = code in HIGHLIGHT
        # Highlighted editions are drawn thicker and on top.
        plt.plot(
            hours, counts, marker='o', markersize=3, linestyle='-',
            color=color, linewidth=1.6 if is_highlight else 1.0,
            zorder=3 if is_highlight else 2,
        )
        # Language label at the end of the curve.
        plt.text(hours[-1] + 0.3, counts[-1], code, color=color,
                 fontsize=9, va='center')

    plt.title("Hourly Activity by Language")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Edits")
    plt.xlim(0, 26)  # leaves room on the right for the labels
    plt.xticks(np.arange(0, 21, 5))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig3_hourly_activity_by_language.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"\nFigure saved to: {os.path.abspath(out_path)}")


# =================================================================
# Main
# =================================================================

def main():
    # The data files are expected in the same folder as this script.
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Reading data from: {data_dir}\n")

    # Count the hourly activity of each edition.
    all_counts = {}
    for code in LANGUAGES:
        print(f"[{code}] processing...")
        path = find_file(data_dir, code)
        all_counts[code] = count_by_hour_of_day(path, TIMEZONE_DICT[code])

    plot_hourly_activity(all_counts)


if __name__ == "__main__":
    main()
