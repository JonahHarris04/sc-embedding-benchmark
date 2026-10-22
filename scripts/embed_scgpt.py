"""Zero-shot cell embeddings from a pretrained scGPT checkpoint (no fine-tuning).

scGPT does its own normalization and matches genes to its vocabulary by symbol,
so we hand it raw counts across all genes rather than the HVG subset.
Download the "whole-human" checkpoint from the scGPT GitHub README into
config["scgpt_model_dir"] (it should contain args.json, vocab.json, best_model.pt).
"""
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scgpt as scg
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))

adata = sc.read_h5ad(args.inp)
adata.X = adata.layers["counts"].copy()
adata.var["gene_name"] = adata.var_names

emb = scg.tasks.embed_data(
    adata,
    model_dir=Path(cfg["scgpt_model_dir"]),
    gene_col="gene_name",
    batch_size=64,
    return_new_adata=True,
)
np.save(args.out, np.asarray(emb.X, dtype=np.float32))
