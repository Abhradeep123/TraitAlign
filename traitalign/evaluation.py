"""
Full-ranking evaluation metrics: HR, NDCG, MRR at configurable k.
"""
from __future__ import annotations

import math

import torch


def _score_matrix(user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
    return torch.matmul(user_embs, item_embs.T)


def _mask_items(scores_mat: torch.Tensor, item_dict: dict) -> None:
    """In-place: set ``scores_mat[u, item] = -inf`` for every (u, item) in *item_dict*."""
    for u, items in item_dict.items():
        if items:
            idx = torch.as_tensor(items, dtype=torch.long, device=scores_mat.device)
            scores_mat[u, idx] = float("-inf")


def val_ndcg_full(
    user_embs: torch.Tensor,
    item_embs: torch.Tensor,
    train_edge_index: torch.Tensor,
    val_dict: dict,
    k: int = 10,
) -> float:
    """Full-ranking NDCG@k on the held-out *validation* split (train items masked)."""
    with torch.no_grad():
        scores_mat = _score_matrix(user_embs, item_embs)
        scores_mat[train_edge_index[0], train_edge_index[1]] = float("-inf")

        val_users = [u for u in val_dict if len(val_dict[u]) > 0]
        if not val_users:
            return 0.0

        topk = scores_mat[val_users].topk(k, dim=1).indices.cpu().numpy()

        ndcg_sum = 0.0
        for row, u in enumerate(val_users):
            gt = set(val_dict[u])
            hits = [1 if topk[row, j] in gt else 0 for j in range(k)]
            dcg = sum(hits[r] / math.log2(r + 2) for r in range(k))
            idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gt), k)))
            if idcg > 0:
                ndcg_sum += dcg / idcg
        return ndcg_sum / len(val_users)


def evaluate_full(
    user_embs: torch.Tensor,
    item_embs: torch.Tensor,
    train_edge_index: torch.Tensor,
    val_dict: dict,
    test_dict: dict,
    k_list: tuple[int, ...] = (10, 20),
) -> dict[str, float]:
    """Full-catalog ranking evaluation on the held-out *test* split.

    Train *and* val items are masked so they cannot be mistaken for correct
    test hits.  Computes HR / NDCG / MRR @ k for each k in *k_list*.
    """
    metrics = {f"{m}@{k}": 0.0 for m in ("HR", "NDCG", "MRR") for k in k_list}

    with torch.no_grad():
        scores_mat = _score_matrix(user_embs, item_embs)
        scores_mat[train_edge_index[0], train_edge_index[1]] = float("-inf")
        _mask_items(scores_mat, val_dict)

        test_users = [u for u in test_dict if len(test_dict[u]) > 0]
        if not test_users:
            return metrics

        max_k = max(k_list)
        topk = scores_mat[test_users].topk(max_k, dim=1).indices.cpu().numpy()

        for row, u in enumerate(test_users):
            gt = set(test_dict[u])
            ranked = topk[row]
            hits_at = [1.0 if ranked[j] in gt else 0.0 for j in range(max_k)]
            for k in k_list:
                n_hit = sum(hits_at[:k])
                dcg = sum(hits_at[j] / math.log2(j + 2) for j in range(k))
                idcg = sum(1.0 / math.log2(j + 2) for j in range(min(len(gt), k)))
                mrr = 0.0
                for j in range(k):
                    if hits_at[j]:
                        mrr = 1.0 / (j + 1)
                        break
                metrics[f"HR@{k}"] += 1.0 if n_hit > 0 else 0.0
                metrics[f"NDCG@{k}"] += (dcg / idcg) if idcg > 0 else 0.0
                metrics[f"MRR@{k}"] += mrr

        n_eval = len(test_users)
        for key in metrics:
            metrics[key] /= n_eval

    return metrics
