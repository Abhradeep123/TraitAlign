# TraitAlign

<p align="center">
  <b>Personality-Aware Graph Neural Network Recommender System</b><br>
  Aligning Behavioural Proxies with User Psychological Traits for Improved Collaborative Filtering
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.4%2B-orange" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Overview

**TraitAlign** is a personality-aware recommendation framework that bridges the gap between *observed interaction behaviour* and *latent psychological traits*. Rather than treating side-information as a simple feature concatenation, TraitAlign introduces a novel alignment objective that explicitly encourages the GNN-learned user embedding space to reflect the structure of personality-derived behavioural proxies.

The key insight is that **how** a user interacts — their diversity, novelty-seeking, temporal stability, exploration tendency, cross-category breadth, and loyalty — is a more faithful reflection of their underlying personality than raw demographic features alone. TraitAlign fuses these behavioural proxies with language-model embeddings of side-information (occupation text, country, movie semantics) through a multi-task alignment loss on top of standard Bayesian Personalised Ranking (BPR).

---

## Highlights

- 🧠 **Personality-aware alignment loss** — contrastive objective that aligns GNN embedding similarity with behavioural-proxy similarity
- 🔗 **Heterogeneous GNN backbone** — supports GCN, GAT, and GraphSAGE on a bipartite user-item graph
- 📐 **Dataset-agnostic proxy engine** — computes six behavioural dimensions from any interaction table using unsupervised item clustering
- 🤖 **MiniLM side-information encoding** — `all-MiniLM-L6-v2` text embeddings for occupation labels, country names, and item metadata
- ⚡ **Optuna HPO** — automated hyperparameter optimisation with TPE sampler
- 📊 **Full-ranking evaluation** — HR, NDCG, MRR at k ∈ {10, 20} with training/validation masking

---

## Datasets

TraitAlign is evaluated on three public benchmarks:

| Dataset | Domain | Personality Signal |
|---|---|---|
| **Personality-18** | Movies (ML-25M) | Big Five (OCEAN) scores |
| **MovieLens-1M** | Movies | Occupation-derived personality proxies |
| **LastFM-9000** | Music | Country / demographic proxies |

Raw data is downloaded automatically on first run (see [Quick Start](#quick-start)).

---

## Repository Structure

```
TraitAlign/
├── traitalign/
│   ├── config.py          # Path dataclasses and global constants
│   ├── downloaders.py     # Idempotent dataset download helpers
│   ├── preprocessing.py   # Behavioural-proxy engine & MiniLM preprocessing
│   ├── embeddings.py      # Item embedding generation pipeline
│   ├── data_loading.py    # Dataset loaders & train/val/test splitting
│   ├── models.py          # GNN architectures & TraitAlign model
│   ├── training.py        # Training loops (TraitAlign + baseline)
│   ├── evaluation.py      # Full-ranking metrics (HR / NDCG / MRR)
│   ├── utils.py           # Seed utilities
│   ├── __init__.py
│   └── __main__.py        # Experiment entry point
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

**Step 1 — Install PyTorch** (choose the command that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
# CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only
pip install torch torchvision torchaudio
```

**Step 2 — Install PyTorch Geometric:**

```bash
pip install torch-geometric
```

**Step 3 — Install all remaining dependencies via `requirements.txt`:**

```bash
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` pins minimum versions for NumPy, SciPy, Pandas, scikit-learn, Sentence-Transformers, Optuna, tqdm, and kagglehub. PyTorch and PyTorch Geometric are intentionally excluded from it because they require a CUDA-version-specific install URL — always install them first using the commands above.

> [!TIP]
> For faster graph operations you can optionally install the PyG C++ acceleration extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`). Pre-built wheels matching your PyTorch + CUDA version are available at the [PyG wheels page](https://data.pyg.org/whl/). These are listed (commented out) at the bottom of `requirements.txt`.

### 2. Configure Kaggle Credentials

Datasets sourced from Kaggle require an API key. Place your `kaggle.json` under `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows), or set the environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3. Run the Full Experiment

From the project root:

```bash
python -m traitalign
```

This will automatically:
1. Download all required raw datasets
2. Run preprocessing and behavioural-proxy computation
3. Generate MiniLM item/side-information embeddings
4. Launch Optuna hyperparameter optimisation for GCN, GAT, and GraphSAGE backbones
5. Evaluate the best TraitAlign model against the MTL baseline
6. Print a summary results table

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAITALIGN_SEED` | `42` | Global random seed |
| `TRAITALIGN_DATASET` | `personality18` | Dataset to use (`personality18`, `movielens`, `lastfm`) |
| `TRAITALIGN_DATA_DIR` | `Datasets/traitalign-personality18` | Path to preprocessed data |
| `TRAITALIGN_DRY_RUN` | `1` | Set to `0` for full training (100 epochs, 20 Optuna trials) |

**Example — full training run:**

```bash
TRAITALIGN_DRY_RUN=0 TRAITALIGN_SEED=123 python -m traitalign
```

**SLURM / HPC:**

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
export TRAITALIGN_DRY_RUN=0
export TRAITALIGN_SEED=42
python -m traitalign
```

---

## Evaluation Protocol

- **Split**: per-user random 80 / 10 / 10 edge-level split; users with fewer than 5 interactions are train-only
- **Metrics**: HR@10, HR@20, NDCG@10, NDCG@20, MRR@10, MRR@20
- **Masking**: training (and validation) items are masked at test time to prevent leakage
- **Backbones evaluated**: GCN, GAT, GraphSAGE

---

## Results (Dry Run / Smoke Test)

Running `python -m traitalign` (dry run, 2 epochs) produces a per-backbone summary table:

```
=========================================================================================
Model                | HR@10   | HR@20   | NDCG@10  | NDCG@20  | MRR@10  | MRR@20
-----------------------------------------------------------------------------------------
Baseline (gcn)       | ...     | ...     | ...      | ...      | ...     | ...
TraitAlign (gcn)     | ...     | ...     | ...      | ...      | ...     | ...
=========================================================================================
```

Full results on the three benchmarks are reported in the accompanying paper.


## License

This project is licensed under the [MIT License](LICENSE).  
© 2026 Abhradeep Dutta || Himanshu Mandal