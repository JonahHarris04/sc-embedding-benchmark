"""Pull a donor-balanced lung subset from the CELLxGENE Census.

Fetches one donor at a time and caches each donor's (subsampled) cells under
data/cache/<census_version>/, so an interrupted run resumes where it stopped.

Why per donor: the Census stores cells in blocks, and a donor's cells sit next
to each other. Reading whole donors touches only their blocks; reading a random
sample of cells scattered across all donors touches nearly every block, which
means downloading most of the dataset to keep a small fraction of it.
"""
import argparse
import re
import time
from pathlib import Path

import anndata as ad
import cellxgene_census
import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
rng = np.random.default_rng(cfg["seed"])

cache = Path(args.out).parent / "cache" / str(cfg["census_version"])
cache.mkdir(parents=True, exist_ok=True)

disease_list = ", ".join(f"'{d}'" for d in cfg["diseases"])
obs_filter = (
    f"tissue_general == '{cfg['tissue']}' "
    f"and disease in [{disease_list}] "
    f"and assay == \"{cfg['assay']}\" "
    "and is_primary_data == True"
)
obs_cols = ["soma_joinid", "donor_id", "cell_type", "disease", "dataset_id"]
var_cols = ["feature_id", "feature_name"]


def cache_path(donor: str) -> Path:
    return cache / (re.sub(r"[^A-Za-z0-9._-]", "_", donor) + ".h5ad")


with cellxgene_census.open_soma(census_version=cfg["census_version"]) as census:
    # 1) Metadata only (cheap): which cells match, and from which donors
    obs = (
        census["census_data"]["homo_sapiens"]
        .obs.read(value_filter=obs_filter, column_names=obs_cols)
        .concat()
        .to_pandas()
    )
    for c in ["donor_id", "disease", "cell_type", "dataset_id"]:
        obs[c] = obs[c].astype(str)
    print(f"{len(obs):,} matching cells from {obs.donor_id.nunique()} donors", flush=True)

    # 2) Pick donors, split evenly across disease groups so both are represented
    donor_disease = obs.groupby("donor_id")["disease"].agg(lambda s: s.mode().iat[0])
    per_group = max(1, cfg["max_donors"] // len(cfg["diseases"]))
    chosen = []
    for d in cfg["diseases"]:
        pool = np.sort(donor_disease[donor_disease == d].index.values)
        picked = rng.choice(pool, size=min(per_group, len(pool)), replace=False)
        chosen += sorted(picked)
        print(f"  {d}: {len(picked)} of {len(pool)} donors", flush=True)

    # 3) Fetch each donor's cells, subsample locally, cache to disk
    t_all = time.time()
    for i, donor in enumerate(chosen, 1):
        f = cache_path(donor)
        if f.exists():
            print(f"[{i}/{len(chosen)}] {donor}: cached, skipping", flush=True)
            continue
        coords = np.sort(obs.loc[obs.donor_id == donor, "soma_joinid"].values)
        t = time.time()
        a = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=coords,
            obs_column_names=obs_cols,
            var_column_names=var_cols,
        )
        n_fetched = a.n_obs
        if a.n_obs > cfg["max_cells_per_donor"]:
            donor_rng = np.random.default_rng([cfg["seed"], i])
            keep = np.sort(donor_rng.choice(a.n_obs, cfg["max_cells_per_donor"], replace=False))
            a = a[keep].copy()
        tmp = f.with_suffix(".tmp.h5ad")
        a.write_h5ad(tmp)
        tmp.rename(f)  # only a complete file ever counts as cached
        print(
            f"[{i}/{len(chosen)}] {donor}: fetched {n_fetched:,} cells, kept {a.n_obs:,} "
            f"({time.time() - t:.0f}s, {(time.time() - t_all) / 60:.1f} min total)",
            flush=True,
        )

# 4) Combine the cached donors into one file
adata = ad.concat([ad.read_h5ad(cache_path(d)) for d in chosen], merge="same")
adata.var_names = adata.var["feature_name"].astype(str)
adata.var_names_make_unique()
adata.obs_names_make_unique()
print(adata, flush=True)
adata.write_h5ad(args.out)
