"""Train scVI on raw counts of the HVGs, with donor as the batch covariate.

scVI is trained without cell-type labels, so using all cells (train + test
donors) leaks no label information into the evaluation.
"""
import argparse

import numpy as np
import scanpy as sc
import scvi
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
scvi.settings.seed = cfg["seed"]

adata = sc.read_h5ad(args.inp)
adata = adata[:, adata.var.highly_variable].copy()

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="donor_id")
model = scvi.model.SCVI(adata, n_latent=cfg["scvi_latent"])
model.train(max_epochs=cfg["scvi_epochs"], early_stopping=True)

np.save(args.out, model.get_latent_representation().astype(np.float32))
