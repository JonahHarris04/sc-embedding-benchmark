"""QC, normalization, HVG selection, PCA, and a fixed donor-level train/test split.

The split lives in adata.obs["split"] so every embedding is evaluated on the
exact same held-out donors.
"""
import argparse

import numpy as np
import scanpy as sc
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
rng = np.random.default_rng(cfg["seed"])

adata = sc.read_h5ad(args.inp)

# ---- QC ----
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
n0 = adata.n_obs
adata = adata[
    (adata.obs.n_genes_by_counts >= cfg["min_genes"])
    & (adata.obs.pct_counts_mt <= cfg["max_pct_mito"])
].copy()
sc.pp.filter_genes(adata, min_cells=3)
print(f"QC kept {adata.n_obs:,}/{n0:,} cells")

# Drop cell types too rare to evaluate fairly
counts = adata.obs.cell_type.value_counts()
keep = counts[counts >= cfg["min_cells_per_type"]].index
adata = adata[adata.obs.cell_type.isin(keep)].copy()
adata.obs["cell_type"] = adata.obs.cell_type.astype(str).astype("category")
print(f"{adata.obs.cell_type.nunique()} cell types after filtering")

# ---- Keep raw counts for scVI / scGPT, then normalize for PCA ----
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(
    adata, n_top_genes=cfg["n_top_genes"], flavor="seurat_v3",
    layer="counts", batch_key="donor_id",
)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=cfg["n_pcs"], mask_var="highly_variable")

# ---- Donor-level split (never split cells from one donor across sets) ----
donors = adata.obs.donor_id.unique()
n_test = max(1, int(round(cfg["test_donor_fraction"] * len(donors))))
test_donors = set(rng.choice(donors, size=n_test, replace=False))
adata.obs["split"] = np.where(adata.obs.donor_id.isin(test_donors), "test", "train")
print(adata.obs.groupby("split").agg(cells=("donor_id", "size"), donors=("donor_id", "nunique")))

adata.write_h5ad(args.out)
