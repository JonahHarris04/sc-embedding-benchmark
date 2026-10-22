# Do single-cell foundation models beat classical embeddings for cell-type annotation?

A reproducible benchmark comparing **PCA**, **scVI**, and zero-shot **scGPT** embeddings on human lung
scRNA-seq (healthy + lung adenocarcinoma) from the CZ CELLxGENE Census.

**Question:** If I annotate cells from new donors by transferring labels from a reference, which
embedding transfers best, and does the advantage hold for *rare* cell types?

## Design
- **Data:** donor-balanced subsample of lung cells (10x 3' v3, primary data only).
- **Split by donor, not by cell.** Held-out donors are never seen during label transfer, which mimics
  annotating a new patient and avoids inflated scores from near-duplicate cells.
- **Metrics:** kNN label-transfer accuracy and macro F1 on held-out donors, per-cell-type F1,
  and cell-type silhouette.

## Run
```bash
conda env create -f environment.yaml && conda activate sc-embed-bench
snakemake --cores 8                               # full pipeline
snakemake --cores 8 --config run_scgpt=false      # no GPU? skip scGPT
```

## Results
_Add `results/figures/f1_by_cell_type.png`, the metrics table, and 3–5 sentences on what you found._

## Limitations
_E.g. kNN is one probe of embedding quality; scGPT is used zero-shot, not fine-tuned; one tissue only._
