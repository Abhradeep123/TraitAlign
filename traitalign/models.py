"""
PyTorch model architectures for the TraitAlign recommender.

Includes:
    StableHeteroGNN   – 2-layer heterogeneous GAT encoder with residual connections.
    TraitAlignMLP     – Side-information → behavioural-proxy prediction head.
    TraitAlignWrapper – Full TraitAlign model (embedding + GAT + MLP).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


# =========================================================================
# Heterogeneous GAT backbone
# =========================================================================

class StableHeteroGNN(nn.Module):
    """2-layer heterogeneous GAT encoder with residual skip connections."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        heads = 4 if hidden_channels % 4 == 0 else 1
        out_dim = hidden_channels // heads
        self.conv1 = HeteroConv({
            ("user", "rates", "item"): GATConv((-1, -1), out_dim, heads=heads, dropout=0.0, add_self_loops=False),
            ("item", "rev_rates", "user"): GATConv((-1, -1), out_dim, heads=heads, dropout=0.0, add_self_loops=False),
        }, aggr="sum")
        self.conv2 = HeteroConv({
            ("user", "rates", "item"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=0.0, concat=False, add_self_loops=False),
            ("item", "rev_rates", "user"): GATConv((-1, -1), hidden_channels, heads=heads, dropout=0.0, concat=False, add_self_loops=False),
        }, aggr="sum")

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        x1 = self.conv1(x_dict, edge_index_dict)
        x1 = {k: x1[k] + x_dict[k] for k in x1}  # skip connection
        x1 = {k: F.elu(v) for k, v in x1.items()}
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {k: x2[k] + x1[k] for k in x2}
        return x2


# =========================================================================
# TraitAlign MLP head
# =========================================================================

class TraitAlignMLP(nn.Module):
    def __init__(
        self,
        minilm_dim: int = 384,
        scalar_dim: int = 2,
        compressed_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 6,
    ) -> None:
        super().__init__()
        self.minilm_dim = minilm_dim
        if minilm_dim > 0:
            self.minilm_compressor = nn.Linear(minilm_dim, compressed_dim)
            combined_dim = compressed_dim + scalar_dim
        else:
            combined_dim = scalar_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, minilm_vec, scalar_vec):
        if self.minilm_dim > 0:
            compressed = self.minilm_compressor(minilm_vec)
            combined = torch.cat([compressed, scalar_vec], dim=1)
        else:
            combined = scalar_vec
        return self.net(combined)


# =========================================================================
# Full TraitAlign wrapper
# =========================================================================

class TraitAlignWrapper(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        minilm_dim: int = 384,
        scalar_dim: int = 2,
        out_dim: int = 6,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.gnn = StableHeteroGNN(hidden_channels=embed_dim)
        self.mlp = TraitAlignMLP(
            minilm_dim=minilm_dim, scalar_dim=scalar_dim, out_dim=out_dim,
        )

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        u_idx = torch.clamp(x_dict["user"], min=0, max=self.user_emb.num_embeddings - 1)
        i_idx = torch.clamp(x_dict["item"], min=0, max=self.item_emb.num_embeddings - 1)
        x_dict_emb = {
            "user": self.user_emb(u_idx),
            "item": self.item_emb(i_idx),
        }
        return self.gnn(x_dict_emb, edge_index_dict, edge_weight_dict)

    def bpr_loss(self, user_feat, pos_item_feat, neg_item_feat, users, pos_items, neg_items):
        pos_scores = (user_feat * pos_item_feat).sum(dim=1)
        neg_scores = (user_feat * neg_item_feat).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (
            (self.user_emb.weight[users] ** 2).sum(dim=1).mean()
            + (self.item_emb.weight[pos_items] ** 2).sum(dim=1).mean()
            + (self.item_emb.weight[neg_items] ** 2).sum(dim=1).mean()
        )
        return bpr, reg_loss

    def compute_traitalign_losses(self, user_feat, minilm_vec, scalar_vec, true_proxies):
        pred_proxies = self.mlp(minilm_vec, scalar_vec)
        l_align = F.mse_loss(pred_proxies, true_proxies)

        u_norm = F.normalize(user_feat, p=2, dim=1, eps=1e-6)
        sim_emb = torch.matmul(u_norm, u_norm.T)
        p_norm = F.normalize(true_proxies, p=2, dim=1, eps=1e-6)
        sim_beh = torch.matmul(p_norm, p_norm.T)

        l_beh_cont = F.mse_loss(sim_emb, sim_beh)
        return l_beh_cont, l_align
