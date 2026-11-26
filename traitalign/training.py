"""
Training loops: Optuna-based hyperparameter tuning (TraitAlign) and the
paper's MTL baseline.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm

from .evaluation import evaluate_full, val_ndcg_full
from .models import BaselineMTLWrapper, TraitAlignWrapper


# Module-level best-NDCG tracker (reset per experiment if needed)
BEST_NDCG_BY_TYPE: dict[str, float] = {}


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


def _full_forward(model, data, device, *, needs_side_info: bool = False,
                  minilm_vec=None, scalar_vec=None):
    """Run a full-graph forward pass (eval mode, no grad)."""
    x_dict_all = {
        "user": data["user"].n_id.to(device),
        "item": data["item"].n_id.to(device),
    }
    safe_ei = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    ew = _collect_full_edge_weights(data, device)

    if needs_side_info:
        return model(x_dict_all, safe_ei, minilm_vec, scalar_vec, ew)
    return model(x_dict_all, safe_ei, ew)


# =========================================================================
# TraitAlign Optuna trial
# =========================================================================

def run_optuna_trial(
    trial,
    dataset_dict: dict,
    gnn_type: str,
    dry_run: bool = True,
    device: str = "cuda",
    seed: int = 42,
) -> float:
    global BEST_NDCG_BY_TYPE
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
        gnn_type=gnn_type,
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

    if best_val_ndcg > BEST_NDCG_BY_TYPE.get(gnn_type, -1.0):
        BEST_NDCG_BY_TYPE[gnn_type] = best_val_ndcg
        torch.save(best_model_state, f"best_full_{gnn_type}_model_seed{seed}.pth")

    return best_val_ndcg


# =========================================================================
# Baseline (MTL, no alignment loss)
# =========================================================================

def run_baseline_model(
    dataset_dict: dict,
    gnn_type: str,
    dataset_name: str = "",
    dry_run: bool = True,
    device: str = "cuda",
) -> tuple[dict, None]:
    print(f"\n==========================================")
    print(f"🚀 RUNNING FEATURE AWARE PAPER BASELINE ({gnn_type.upper()}) 🚀")
    print(f"==========================================")

    torch.cuda.empty_cache()
    data = dataset_dict["data"]
    train_loader = _build_train_loader(data)

    # Normalised side-info
    raw_minilm = dataset_dict["minilm_vec"].to(device)
    if raw_minilm.shape[1] > 0:
        minilm_full = (raw_minilm - raw_minilm.mean(0)) / (raw_minilm.std(0) + 1e-6)
    else:
        minilm_full = raw_minilm

    raw_scalar = dataset_dict["scalar_vec"].to(device)
    scalar_full = (raw_scalar - raw_scalar.mean(0)) / (raw_scalar.std(0) + 1e-6)
    proxies_full = dataset_dict["true_proxies"].to(device)

    embed_dim = 5 if "personality" in dataset_name.lower() else 64
    model = BaselineMTLWrapper(
        num_users=dataset_dict["num_users"],
        num_items=dataset_dict["num_items"],
        gnn_type=gnn_type,
        embed_dim=embed_dim,
        minilm_dim=dataset_dict["minilm_vec"].shape[1],
        scalar_dim=dataset_dict["scalar_vec"].shape[1],
        out_dim=dataset_dict["true_proxies"].shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lambda_aux = 0.0  # baseline = no alignment / similarity-matching losses

    epochs = 2 if dry_run else 100
    best_val_ndcg = -1.0
    best_model_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Baseline Epoch {epoch}/{epochs}", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_batch = {"user": batch["user"].n_id, "item": batch["item"].n_id}
            safe_ei, ew = _safe_edge_dicts(batch)

            g_users_batch = torch.clamp(
                batch["user"].n_id, min=0, max=dataset_dict["num_users"] - 1,
            )
            b_minilm = minilm_full[g_users_batch]
            b_scalar = scalar_full[g_users_batch]

            out = model(x_batch, safe_ei, b_minilm, b_scalar, ew)

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

            # BPR
            pos_scores = (user_feat * pos_feat).sum(dim=1)
            neg_scores = (user_feat * neg_feat).sum(dim=1)
            loss_link = -F.logsigmoid(pos_scores - neg_scores).mean()

            # MTL auxiliary
            p_hat = model.aux_head(out["user"])
            p_true = proxies_full[g_users_batch]
            variance = p_true.var()
            loss_aux = F.mse_loss(p_hat, p_true) / variance if variance > 0 else F.mse_loss(p_hat, p_true)

            loss = loss_link + lambda_aux * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Periodic validation
        if epoch % 5 == 0 or dry_run or epoch == 1:
            model.eval()
            with torch.no_grad():
                out_eval = _full_forward(
                    model, data, device,
                    needs_side_info=True,
                    minilm_vec=minilm_full, scalar_vec=scalar_full,
                )

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

    print(f"Baseline Finished | Time: {time.time() - t0:.1f}s | Best Val NDCG@10: {best_val_ndcg:.4f}")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out_eval = _full_forward(
            model, data, device,
            needs_side_info=True,
            minilm_vec=minilm_full, scalar_vec=scalar_full,
        )

    return evaluate_full(
        out_eval["user"], out_eval["item"],
        dataset_dict["train_edge_index"].to(device),
        dataset_dict["val_dict"], dataset_dict["test_dict"],
        k_list=(10, 20),
    ), None
