"""Benchmark embeddings by how well they transfer cell-type labels to unseen donors.

For each embedding:
  * kNN label transfer: fit on train donors, predict held-out donors
    -> accuracy, macro F1 (rare types count as much as common ones), per-type F1
  * cell-type silhouette on held-out cells (do types form separate clusters?)
  * UMAP colored by cell type
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--adata", required=True)
p.add_argument("--embeddings", nargs="+", required=True)
p.add_argument("--outdir", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
rng = np.random.default_rng(cfg["seed"])

out = Path(args.outdir)
(out / "figures").mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(args.adata, backed="r")
obs = adata.obs[["cell_type", "donor_id", "split"]].copy()
y = obs.cell_type.astype(str).values
train = (obs.split == "train").values
test = ~train
labels = sorted(np.unique(y[test]))

rows, per_type = [], {}
embs = {Path(f).stem: np.load(f) for f in args.embeddings}

for name, X in embs.items():
    assert X.shape[0] == len(obs), f"{name}: {X.shape[0]} rows vs {len(obs)} cells"
    Xs = StandardScaler().fit(X[train]).transform(X)

    knn = KNeighborsClassifier(n_neighbors=cfg["knn_k"], n_jobs=-1).fit(Xs[train], y[train])
    pred = knn.predict(Xs[test])

    # Silhouette is O(n^2); subsample held-out cells
    idx = np.flatnonzero(test)
    idx = rng.choice(idx, size=min(5000, len(idx)), replace=False)
    sil = silhouette_score(Xs[idx], y[idx])

    rows.append({
        "embedding": name,
        "dims": X.shape[1],
        "accuracy": accuracy_score(y[test], pred),
        "macro_f1": f1_score(y[test], pred, average="macro"),
        "celltype_silhouette": sil,
    })
    per_type[name] = f1_score(y[test], pred, labels=labels, average=None, zero_division=0)
    print(rows[-1])

metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
metrics.to_csv(out / "metrics.csv", index=False)
print(metrics.to_string(index=False))

# ---- Per-cell-type F1, sorted by abundance so you can see where rare types fail ----
abundance = pd.Series(y[test]).value_counts()
f1_df = pd.DataFrame(per_type, index=labels).loc[abundance.index]
f1_df.index = [f"{ct} (n={abundance[ct]})" for ct in f1_df.index]
fig, ax = plt.subplots(figsize=(1.2 * len(embs) + 4, 0.3 * len(f1_df) + 1.5))
sns.heatmap(f1_df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
            cbar_kws={"label": "F1 on held-out donors"}, ax=ax)
ax.set_title("Label transfer F1 by cell type (most → least abundant)")
fig.tight_layout()
fig.savefig(out / "figures" / "f1_by_cell_type.png", dpi=200)
plt.close(fig)

# ---- UMAPs on a shared subsample ----
sub = np.sort(rng.choice(len(obs), size=min(20000, len(obs)), replace=False))
fig, axes = plt.subplots(1, len(embs), figsize=(6 * len(embs), 5.5), squeeze=False)
for ax, (name, X) in zip(axes[0], embs.items()):
    a = sc.AnnData(X=np.zeros((len(sub), 1), dtype=np.float32), obs=obs.iloc[sub].copy())
    a.obsm["emb"] = X[sub]
    sc.pp.neighbors(a, use_rep="emb", random_state=cfg["seed"])
    sc.tl.umap(a, random_state=cfg["seed"])
    sc.pl.umap(a, color="cell_type", ax=ax, show=False, legend_loc=None,
               frameon=False, title=name.upper(), size=4)
fig.tight_layout()
fig.savefig(out / "figures" / "umaps.png", dpi=200)
plt.close(fig)
