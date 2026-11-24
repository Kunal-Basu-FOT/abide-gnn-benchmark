# ABIDE functional connectome benchmark

Code for comparing an MLP with five graph neural networks on ABIDE functional
connectivity features. The saved reference runs use AAL and CC200 atlases,
top-10% graph edges, and stratified 10-fold cross-validation. The corresponding
notebook outputs are summarized in [`results/reference/benchmark_summary.csv`](results/reference/benchmark_summary.csv).

## Reference results

These are pooled out-of-fold AUROC values from the saved **early-stopping**
notebooks. They have been transcribed and checked against those notebook outputs;
the full training runs have not been rerun for this release.

| Model | AAL | CC200 |
|---|---:|---:|
| MLP | 0.7067 | 0.7476 |
| GCN | 0.6166 | 0.6497 |
| ChebNet | 0.6174 | 0.6559 |
| GIN | 0.5849 | 0.5834 |
| GraphSAGE | 0.6284 | 0.6696 |
| GAT | 0.6229 | 0.6786 |

Accuracy, F1, AUPRC, balanced accuracy, sensitivity and specificity are in the
CSV. The MLP is an important baseline: in these runs it outperformed every GNN
on AUROC. This benchmark does not establish cross-site generalization. The
package contains a separate leave-one-site-out implementation, but the table
above reports stratified folds pooled across sites.

## Data

Raw ABIDE images, subject-level features, and phenotypic records are not
distributed here. Obtain the [ABIDE Preprocessed Connectomes Project](https://preprocessed-connectomes-project.org/abide/)
data under its terms, then construct the AAL and/or CC200 merged connectivity
tables. The loader expects one row per subject, columns `SITE_ID`, `DX_GROUP`,
`AGE_AT_SCAN`, `SEX`, `FIQ`, and connectivity columns prefixed `FC_`. The
reference notebooks used 848 subjects; the code does not force that count.
`DX_GROUP` is mapped to ASD=1 and typical development=0 when coded as 1/2.

## Run

Python 3.11 or 3.12 is recommended. Install PyTorch and PyTorch Geometric for
your CPU/CUDA setup using their official instructions, then install the remaining
dependencies with `python -m pip install -r requirements.txt`, then install
this package with `python -m pip install -e .`.

```bash
python scripts/run_benchmark.py \
  --aal-csv /path/to/ABIDE_AAL_Merged_FC_Phenotypic.csv \
  --cc200-csv /path/to/ABIDE_CC200_Merged_FC_Phenotypic.csv \
  --output-dir results/generated
```

Use `--models MLP,GCN` or `--atlases CC200` for a smaller run. The default is
the six-model, two-atlas baseline; tuning and explainability are opt-in. The
input tables must use the same atlas ordering as the original preprocessing.
All output paths are relative to the chosen output directory.

## Repository map

- `abide_gnnbench/`: model definitions, fold-wise preprocessing, training,
  metrics, graph construction, and optional explainability.
- `scripts/run_benchmark.py`: portable command-line entry point.
- `scripts/extract_archived_results.py`: extracts the published table from the
  archived notebook ZIP when that ZIP is available.
- `results/reference/`: compact results and source provenance.

The graph code fits standardization inside each outer training fold. Model
selection and any site-specific conclusions need a separate audit before being
treated as confirmatory. The reference table is an archival result, not an
estimate from a newly executed run of this repository.
