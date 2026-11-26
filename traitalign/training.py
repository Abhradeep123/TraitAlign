"""
Training loop: Optuna-based hyperparameter tuning for TraitAlign (GAT).
"""
from __future__ import annotations

import copy
import time

import torch
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm

from .evaluation import val_ndcg_full
from .models import TraitAlignWrapper


# Module-level best-NDCG tracker per seed (reset per experiment if needed)
BEST_NDCG_BY_SEED: dict[int, float] = {}


# =========================================================================
# Shared helpers
# =========================================================================

def _build_train_loader(data, batch_size: int = 2048) -> LinkNeighborLoader:
    edge_label_index = data["user", "rates", "item"].edge_index
    edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float)
    return LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=1.0,
        edge_label_index=(("user", "rates", "item"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def _safe_edge_dicts(batch):
    """Clamp edge indices to valid ranges and extract edge weights."""
    safe_edge_index_dict = {}
    edge_weight_dict = {}
    for k, e_idx in batch.edge_index_dict.items():
        src_type, _, dst_type = k
        safe = e_idx.clone()
        safe[0] = torch.clamp(safe[0], min=0, max=max(0, batch[src_type].n_id.size(0) - 1))
        safe[1] = torch.clamp(safe[1], min=0, max=max(0, batch[dst_type].n_id.size(0) - 1))
        safe_edge_index_dict[k] = safe
        if hasattr(batch[k], "edge_weight"):
            edge_weight_dict[k] = batch[k].edge_weight
    return safe_edge_index_dict, edge_weight_dict or None


def _extract_bpr_edges(batch):
    """Return pos/neg edge indices and the count for BPR pairing."""
    label_index = batch["user", "rates", "item"].edge_label_index
    label = batch["user", "rates", "item"].edge_label
    pos = label_index[:, label == 1.0]
    neg = label_index[:, label == 0.0]
    n = min(pos.size(1), neg.size(1))
    return pos[:, :n], neg[:, :n], n


def _collect_full_edge_weights(data, device):
    """Gather per-edge-type weights from data stores."""
    ew: dict[tuple, torch.Tensor] = {}
    for k in data.edge_index_dict:
        if hasattr(data[k], "edge_weight"):
            ew[k] = data[k].edge_weight.to(device)
    return ew or None


def _full_forward(model, data, device):
    """Run a full-graph forward pass (eval mode, no grad)."""
    x_dict_all = {
        "user": data["user"].n_id.to(device),
        "item": data["item"].n_id.to(device),
    }
    safe_ei = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    ew = _collect_full_edge_weights(data, device)
    return model(x_dict_all, safe_ei, ew)


# =========================================================================
# TraitAlign Optuna trial
# =========================================================================

def run_optuna_trial(
    trial,
    dataset_dict: dict,
    dry_run: bool = True,
    device: str = "cuda",
    seed: int = 42,
) -> float:
    global BEST_NDCG_BY_SEED
    torch.cuda.empty_cache()

    lambda_beh_cont = trial.suggest_float("lambda_beh_cont", 1e-4, 1.0, log=True)
    lambda_align = trial.suggest_float("lambda_align", 1e-4, 1.0, log=True)
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-2, log=True)

    data = dataset_dict["data"]
    train_loader = _build_train_loader(data)

    minilm_full = dataset_dict["minilm_vec"].to(device)
    scalar_full = dataset_dict["scalar_vec"].to(device)
    proxies_full = dataset_dict["true_proxies"].to(device)

    model = TraitAlignWrapper(
        num_users=dataset_dict["num_users"],
        num_items=dataset_dict["num_items"],
        embed_dim=64,
        minilm_dim=dataset_dict["minilm_vec"].shape[1],
        scalar_dim=dataset_dict["scalar_vec"].shape[1],
        out_dim=dataset_dict["true_proxies"].shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 2 if dry_run else 100
    best_val_ndcg = -1.0
    best_model_state = None
    patience = 0
    t0 = time.time()

    print(f"\n>>> Starting Optuna Trial {trial.number} (Epochs: {epochs}, Batches: {len(train_loader)}) <<<\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_dict_batch = {"user": batch["user"].n_id, "item": batch["item"].n_id}
            safe_ei, ew = _safe_edge_dicts(batch)
            out = model(x_dict_batch, safe_ei, ew)

            pos_edges, neg_edges, n_bpr = _extract_bpr_edges(batch)
            if n_bpr == 0:
                continue

            max_u = max(0, out["user"].size(0) - 1)
            max_i = max(0, out["item"].size(0) - 1)
            u_loc = torch.clamp(pos_edges[0], min=0, max=max_u)
            pi_loc = torch.clamp(pos_edges[1], min=0, max=max_i)
            ni_loc = torch.clamp(neg_edges[1], min=0, max=max_i)

            user_feat = out["user"][u_loc]
            pos_feat = out["item"][pi_loc]
            neg_feat = out["item"][ni_loc]

            g_users = torch.clamp(batch["user"].n_id[u_loc], min=0, max=dataset_dict["num_users"] - 1)
            g_pos = torch.clamp(batch["item"].n_id[pi_loc], min=0, max=dataset_dict["num_items"] - 1)
            g_neg = torch.clamp(batch["item"].n_id[ni_loc], min=0, max=dataset_dict["num_items"] - 1)

            bpr, reg = model.bpr_loss(user_feat, pos_feat, neg_feat, g_users, g_pos, g_neg)
            l_beh, l_align = model.compute_traitalign_losses(
                user_feat, minilm_full[g_users], scalar_full[g_users], proxies_full[g_users],
            )
            loss = bpr + lambda_beh_cont * l_beh + lambda_align * l_align + lambda_reg * reg

            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN/Inf loss detected! Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Periodic validation
        if epoch % 5 == 0 or dry_run or epoch == 1:
            model.eval()
            with torch.no_grad():
                out_eval = _full_forward(model, data, device)

            cur_ndcg = val_ndcg_full(
                out_eval["user"], out_eval["item"],
                dataset_dict["train_edge_index"].to(device),
                dataset_dict["val_dict"], k=10,
            )
            print(f"  Epoch [{epoch}/{epochs}] | Loss: {epoch_loss:.4f} | Val NDCG@10: {cur_ndcg:.4f}")

            if cur_ndcg > best_val_ndcg:
                best_val_ndcg = cur_ndcg
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

            if patience >= 10:
                print("  [Early Stopping] Patience reached.")
                break

    print(f"Trial Finished | Time: {time.time() - t0:.1f}s | Best Val NDCG@10: {best_val_ndcg:.4f}")

    if best_val_ndcg > BEST_NDCG_BY_SEED.get(seed, -1.0):
        BEST_NDCG_BY_SEED[seed] = best_val_ndcg
        torch.save(best_model_state, f"best_full_gat_model_seed{seed}.pth")

    return best_val_ndcg
