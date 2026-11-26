"""
TraitAlign experiment runner.

Usage
-----
    # From the project root (d:\\Demo):
    python -m traitalign                     # default: personality18, seed=42
    TRAITALIGN_SEED=123 python -m traitalign # custom seed
    TRAITALIGN_DRY_RUN=1 python -m traitalign # quick smoke-test (2 epochs)

All configuration is read from environment variables (same as the original
script) so existing SLURM job-scripts work unchanged.
"""
from __future__ import annotations

import json
import os
import warnings

import torch

warnings.filterwarnings("ignore")

# CUDA debug flags (set early, before any kernel launch)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def main() -> None:
    import optuna

    from .data_loading import load_traitalign_dataset
    from .downloaders import download_all
    from .evaluation import evaluate_full
    from .models import TraitAlignWrapper
    from .training import run_optuna_trial, _full_forward
    from .utils import set_seed

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    # ---- 1. Download raw datasets (idempotent) --------------------------
    download_all()

    # ---- 2. Preprocessing (only needed once; handled by preprocessor) ---
    from .preprocessing import TraitAlignPreprocessor
    preprocessor = TraitAlignPreprocessor(output_dir="Datasets")
    preprocessor.run_all()

    # ---- 3. Generate embeddings (only needed once) ----------------------
    if not os.path.exists("P_i.npy"):
        from .embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        generator.run()
    else:
        print("P_i.npy already exists, skipping embedding generation.")

    # ---- 4. Experiment configuration ------------------------------------
    SEED = int(os.environ.get("TRAITALIGN_SEED", 42))
    set_seed(SEED)

    DATASET_NAME = os.environ.get("TRAITALIGN_DATASET", "personality18")
    DATA_DIR = os.environ.get("TRAITALIGN_DATA_DIR", r"Datasets/traitalign-personality18")
    DRY_RUN = os.environ.get("TRAITALIGN_DRY_RUN", "0").lower() in ("1", "true", "yes")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🚀 STARTING EXPERIMENT (seed={SEED}, gnn=gat) 🚀")
    dataset = load_traitalign_dataset(DATASET_NAME, DATA_DIR)
    print("✅ Real data loaded successfully!")

    # ---- Optuna HPO -----------------------------------------------------
    n_trials = 2 if DRY_RUN else 20
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: run_optuna_trial(
            trial, dataset, dry_run=DRY_RUN, device=device, seed=SEED,
        ),
        n_trials=n_trials,
    )

    print(f"\n🎉 TRAITALIGN TUNING COMPLETE 🎉")
    print(f"Best TraitAlign Validation NDCG@10: {study.best_value:.4f}")
    print(f"Best TraitAlign model saved to: best_full_gat_model_seed{SEED}.pth")

    # Save best hyper-parameters
    hparams = {
        "seed": SEED,
        "gnn_type": "gat",
        "dataset": DATASET_NAME,
        "best_value_ndcg10": study.best_value,
        "best_params": study.best_params,
    }
    hparams_path = f"best_hparams_full_gat_seed{SEED}.json"
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"Best hyperparameters saved to: {hparams_path}")

    # ---- Evaluate best TraitAlign model ---------------------------------
    best_model = TraitAlignWrapper(
        num_users=dataset["num_users"],
        num_items=dataset["num_items"],
        embed_dim=64,
        minilm_dim=dataset["minilm_vec"].shape[1],
        scalar_dim=dataset["scalar_vec"].shape[1],
        out_dim=dataset["true_proxies"].shape[1],
    ).to(device)
    best_model.load_state_dict(
        torch.load(f"best_full_gat_model_seed{SEED}.pth", map_location=device),
    )
    best_model.eval()
    with torch.no_grad():
        out_eval = _full_forward(best_model, dataset["data"], device)

    traitalign_metrics = evaluate_full(
        out_eval["user"], out_eval["item"],
        dataset["train_edge_index"].to(device),
        dataset["val_dict"], dataset["test_dict"],
        k_list=(10, 20),
    )

    # ---- Summary table --------------------------------------------------
    def _fmt(m: dict) -> str:
        return (
            f"{m['HR@10']:.4f}  | {m['HR@20']:.4f}  | "
            f"{m['NDCG@10']:.4f}  | {m['NDCG@20']:.4f}  | "
            f"{m['MRR@10']:.4f}  | {m['MRR@20']:.4f}"
        )

    print(f"\n🔥 EXPERIMENT FULL SUMMARY (seed={SEED}) 🔥")
    print("=" * 91)
    print(f"{'Model':<20} | {'HR@10':<7} | {'HR@20':<7} | {'NDCG@10':<7} | {'NDCG@20':<7} | {'MRR@10':<7} | {'MRR@20':<7}")
    print("-" * 91)
    print(f"{'TraitAlign (gat)':<20} | " + _fmt(traitalign_metrics))
    print("=" * 91)


if __name__ == "__main__":
    main()
