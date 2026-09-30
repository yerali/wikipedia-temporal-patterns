# Temporal patterns of preferences through Wikipedia editing in different languages

Scripts that reproduce the figures of the article and of its Supplementary
Information.

## Input data

The data files of the eleven language editions must already be in the same
folder as the scripts, one file per edition, named `<code>.txt` or
`<code>_*.txt` (for example `es.txt`), where `<code>` is one of
`ar de es fr hu it ja pt ru vi zh`.

Each line of a file is one edit:

```
title user <unix_timestamp> ['category1', 'category2', ...]
```

## Requirements

The figures were generated with Python 3.12.3 and the following package
versions:

- numpy 2.5.2
- scipy 1.18.1
- pandas 3.0.5
- matplotlib 3.11.1
- scikit-learn 1.9.0
- tensorflow 2.21.0 (Supplementary Information only, for the autoencoder)
- umap-learn 0.5.12 (Supplementary Information only)

They can be installed with:

```
pip install -r requirements.txt
```

If tensorflow or umap-learn is not installed, the corresponding method is
skipped and the rest of the analysis runs normally. The results of the
autoencoder, t-SNE and UMAP may differ slightly with other package
versions, even though all random seeds are fixed.

## Scripts

| Script | Output |
|---|---|
| `1_fig1_2_all_languages.py` | Figs. 1 and 2 |
| `2_fig3_hourly_activity_by_language.py` | Fig. 3 |
| `3_fig4.py` | Fig. 4 |
| `4_fig5_6.py` | Figs. 5 and 6 |
| `5_fig7.py` | Fig. 7 |
| `6_figSI.py` | Fig. S1 (Supplementary Information) |

Two further files are modules imported by the scripts above and do not need
to be run on their own:

- `dendrogramas_dos_normalizaciones.py`: builds the temporal tensor
  (11 editions x 13 categories x 168 hours) and defines its two
  normalisations. Used by `4_fig5_6.py` and `6_figSI.py`.
- `dimensionality_reduction.py`: the dimensionality reduction methods
  compared in the Supplementary Information. Used by `6_figSI.py`.

## Usage

Run each script from the folder that contains the scripts and the data:

```
python 1_fig1_2_all_languages.py edits
python 1_fig1_2_all_languages.py editors
python 2_fig3_hourly_activity_by_language.py
python 3_fig4.py
python 4_fig5_6.py
python 5_fig7.py
python 6_figSI.py
```

The scripts are independent and can be run in any order. Figures are saved
as PDF in the `figuras/` subfolder.

Reading the data files is the slowest step, so `3_fig4.py`, `4_fig5_6.py`,
`5_fig7.py` and `6_figSI.py` cache the counts in CSV files in the same
folder and reuse them in later runs. To rebuild the caches from the data
files, add `--recompute` (for `3_fig4.py`, `4_fig5_6.py` and `5_fig7.py`),
or delete the CSV files.

Counting distinct editors keeps sets of user names in memory; for the
largest editions this may require several GB of RAM.
