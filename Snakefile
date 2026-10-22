configfile: "config.yaml"

EMBEDDINGS = ["pca", "scvi"] + (["scgpt"] if config["run_scgpt"] else [])


rule all:
    input:
        "results/metrics.csv",
        "results/figures/f1_by_cell_type.png",
        "results/figures/umaps.png",


rule download:
    output: "data/raw.h5ad"
    params: cfg="config.yaml"
    shell: "python scripts/download.py --config {params.cfg} --out {output}"


rule preprocess:
    input: "data/raw.h5ad"
    output: "data/processed.h5ad"
    params: cfg="config.yaml"
    shell: "python scripts/preprocess.py --config {params.cfg} --inp {input} --out {output}"


rule embed_pca:
    # PCA is computed during preprocessing; this rule just exports it
    input: "data/processed.h5ad"
    output: "embeddings/pca.npy"
    shell: "python scripts/export_pca.py --inp {input} --out {output}"


rule embed_scvi:
    input: "data/processed.h5ad"
    output: "embeddings/scvi.npy"
    params: cfg="config.yaml"
    resources: gpu=1
    shell: "python scripts/embed_scvi.py --config {params.cfg} --inp {input} --out {output}"


rule embed_scgpt:
    input: "data/processed.h5ad"
    output: "embeddings/scgpt.npy"
    params: cfg="config.yaml"
    resources: gpu=1
    shell: "python scripts/embed_scgpt.py --config {params.cfg} --inp {input} --out {output}"


rule evaluate:
    input:
        adata="data/processed.h5ad",
        embs=expand("embeddings/{e}.npy", e=EMBEDDINGS),
    output:
        "results/metrics.csv",
        "results/figures/f1_by_cell_type.png",
        "results/figures/umaps.png",
    params: cfg="config.yaml"
    shell:
        "python scripts/evaluate.py --config {params.cfg} --adata {input.adata} "
        "--embeddings {input.embs} --outdir results"
