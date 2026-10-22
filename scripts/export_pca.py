"""Export the PCA computed in preprocess.py as a plain .npy embedding."""
import argparse

import numpy as np
import scanpy as sc

p = argparse.ArgumentParser()
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

adata = sc.read_h5ad(args.inp, backed="r")
np.save(args.out, np.asarray(adata.obsm["X_pca"], dtype=np.float32))
