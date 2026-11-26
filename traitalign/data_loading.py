"""
Dataset loading, user-level train/val/test splitting, and PyG HeteroData
construction for all three TraitAlign datasets (personality18, movielens, lastfm).
"""
from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from .config import MIN_INTERACTIONS
from .preprocessing import GlobalBehavioralProxies


# =========================================================================
# Per-user random 80 / 10 / 10 edge-level split
# =========================================================================

def user_split_80_10_10(
    src_ids: list,
    dst_ids: list,
    num_users: int,
    seed: int = 42,
    min_interactions: int = MIN_INTERACTIONS,
) -> tuple[list[int], dict, dict]:
    """Per-user random 80/10/10 split over *edge positions*.

    Users with fewer than ``min_interactions`` total interactions get all
    their edges placed in train (they don't contribute val/test instances).

    Returns
    -------
    train_pos : sorted list of original edge-index positions
    val_dict  : ``user_idx -> [held-out item ids]``
    test_dict : ``user_idx -> [held-out item ids]``
    """
    rng = np.random.default_rng(seed)

    user_edge_idx: dict[int, list[int]] = defaultdict(list)
    for idx, u in enumerate(src_ids):
        user_edge_idx[u].append(idx)

    train_pos: list[int] = []
    val_dict: dict[int, list[int]] = {}
    test_dict: dict[int, list[int]] = {}

    for u in range(num_users):
        idxs = user_edge_idx.get(u, [])
        n = len(idxs)
        if n == 0:
            continue
        idxs = list(idxs)
        rng.shuffle(idxs)

        if n < min_interactions:
            train_pos.extend(idxs)
            continue

        n_test = max(1, int(round(0.1 * n)))
        n_val = max(1, int(round(0.1 * n)))
        test_idx = idxs[:n_test]
        val_idx = idxs[n_test : n_test + n_val]
        tr_idx = idxs[n_test + n_val :]
        if len(tr_idx) == 0:  # always keep ≥1 train interaction
            tr_idx = [val_idx.pop()] if val_idx else [test_idx.pop()]

        test_dict[u] = [dst_ids[i] for i in test_idx]
        val_dict[u] = [dst_ids[i] for i in val_idx]
        train_pos.extend(tr_idx)

    train_pos = sorted(train_pos)

    n_val_int = sum(len(v) for v in val_dict.values())
    n_test_int = sum(len(v) for v in test_dict.values())
    print(
        f"  [split] users={num_users} train={len(train_pos)} "
        f"val={n_val_int} test={n_test_int} "
        f"(interactions, min_interactions={min_interactions})"
    )

    return train_pos, val_dict, test_dict


# =========================================================================
# Degree-normalised edge weights
# =========================================================================

def _precompute_edge_weights(
    train_edge_index: torch.Tensor, num_users: int, num_items: int,
) -> torch.Tensor:
    """Symmetric degree normalisation: ``D_u^{-1/2} · D_i^{-1/2}``."""
    deg_u = torch.bincount(train_edge_index[0], minlength=num_users).float()
    deg_i = torch.bincount(train_edge_index[1], minlength=num_items).float()
    deg_u_inv = deg_u.pow(-0.5)
    deg_u_inv.masked_fill_(deg_u_inv == float("inf"), 0)
    deg_i_inv = deg_i.pow(-0.5)
    deg_i_inv.masked_fill_(deg_i_inv == float("inf"), 0)
    return deg_u_inv[train_edge_index[0]] * deg_i_inv[train_edge_index[1]]


# =========================================================================
# Build a PyG HeteroData from a train split
# =========================================================================

def _build_heterodata(
    train_edge_index: torch.Tensor, num_users: int, num_items: int,
) -> HeteroData:
    data = HeteroData()
    data["user"].num_nodes = num_users
    data["item"].num_nodes = num_items
    data["user"].n_id = torch.arange(num_users)
    data["item"].n_id = torch.arange(num_items)
    data["user", "rates", "item"].edge_index = train_edge_index
    data["user", "rates", "item"].edge_weight = _precompute_edge_weights(
        train_edge_index, num_users, num_items,
    )
    data = T.ToUndirected()(data)
    return data


# =========================================================================
# Personality-18 loader
# =========================================================================

def _load_personality18() -> dict:
    personality_cols = [
        "openness", "agreeableness", "emotional_stability",
        "conscientiousness", "extraversion",
    ]

    # Resolve paths (Kaggle vs. local)
    pers_path = "/kaggle/input/datasets/arslanali4343/top-personality-dataset/2018-personality-data.csv"
    if not os.path.exists(pers_path):
        pers_path = r"Datasets/top-personality-dataset/2018-personality-data.csv"
    pers_df = pd.read_csv(pers_path)
    pers_df.columns = pers_df.columns.str.strip()

    ratings_path = "/kaggle/input/datasets/arslanali4343/top-personality-dataset/2018_ratings.csv"
    if not os.path.exists(ratings_path):
        ratings_path = r"Datasets/top-personality-dataset/2018_ratings.csv"
    ratings_df = pd.read_csv(ratings_path)
    ratings_df.columns = ratings_df.columns.str.strip()

    movies_path = "/kaggle/input/datasets/garymk/movielens-25m-dataset/ml-25m/movies.csv"
    if not os.path.exists(movies_path):
        movies_path = r"Datasets/movielens-25m/movies.csv"
    movies_df = pd.read_csv(movies_path)

    print("Loading MiniLM Embeddings for Items...")
    p_i_path = "/kaggle/input/datasets/kkaushik06/movielens-minilm-embeddings/P_i.npy"
    if not os.path.exists(p_i_path):
        p_i_path = r"P_i.npy"
    P_i = np.load(p_i_path)
    movie_emb_dict = {int(row[0]): row[1:] for row in P_i}
    del P_i
    gc.collect()

    ratings_df = ratings_df.merge(
        movies_df[["movieId", "genres"]], left_on="movie_id", right_on="movieId", how="left",
    )
    ratings_df["genres"] = ratings_df["genres"].fillna("Unknown")
    ratings_df["tstamp"] = pd.to_datetime(ratings_df["tstamp"])
    ratings_df = ratings_df.sort_values(["useri", "tstamp"])

    unique_users = ratings_df["useri"].unique()
    unique_movies = ratings_df["movie_id"].unique()
    user_mapping = {uid: i for i, uid in enumerate(unique_users)}
    movie_mapping = {mid: i for i, mid in enumerate(unique_movies)}

    ratings_df["user_idx"] = ratings_df["useri"].map(user_mapping)
    ratings_df["movie_idx"] = ratings_df["movie_id"].map(movie_mapping)
    pers_df = pers_df[pers_df["userid"].isin(unique_users)].copy()
    pers_df["user_idx"] = pers_df["userid"].map(user_mapping)

    num_users = len(user_mapping)
    num_items = len(movie_mapping)

    personality_tensor = torch.zeros((num_users, 5), dtype=torch.float)
    for _, row in pers_df.iterrows():
        idx = row["user_idx"]
        traits = [row[c] for c in personality_cols]
        personality_tensor[idx] = torch.tensor(traits, dtype=torch.float)

    del movies_df
    gc.collect()

    # Behavioural proxies via the shared engine
    print(
        "Constructing Behavioral Profiles (shared GlobalBehavioralProxies engine "
        "-- identical formulas to MovieLens-1M / LastFM, plus loyalty)..."
    )
    item_embeddings_by_idx = {}
    for movie_id, emb in movie_emb_dict.items():
        idx = movie_mapping.get(movie_id)
        if idx is not None:
            item_embeddings_by_idx[idx] = emb
    del movie_emb_dict
    gc.collect()

    aligned_item_ids = sorted(item_embeddings_by_idx.keys())
    aligned_embeddings = np.array(
        [item_embeddings_by_idx[i] for i in aligned_item_ids], dtype=float,
    )

    proxy_engine = GlobalBehavioralProxies(n_clusters=20)
    proxy_engine.fit_item_structure(aligned_item_ids, aligned_embeddings)
    proxy_engine.fit_popularity(ratings_df, item_col="movie_idx")
    proxies_df = proxy_engine.compute(
        ratings_df, user_col="user_idx", item_col="movie_idx",
        timestamp_col="tstamp", weight_col=None,
        deconfound=True, n_degree_bins=10,
    )

    behavioral_profiles = np.zeros(
        (num_users, len(GlobalBehavioralProxies.PROXY_COLUMNS)),
    )
    proxies_by_user = proxies_df.set_index("user_idx")
    for u_idx in proxies_by_user.index:
        behavioral_profiles[int(u_idx)] = proxies_by_user.loc[
            u_idx, GlobalBehavioralProxies.PROXY_COLUMNS
        ].values
    true_proxies = torch.tensor(behavioral_profiles, dtype=torch.float32)

    # Split & graph
    print("Constructing PyG HeteroData...")
    src_all = ratings_df["user_idx"].tolist()
    dst_all = ratings_df["movie_idx"].tolist()
    train_pos, val_dict, test_dict = user_split_80_10_10(
        src_all, dst_all, num_users, seed=42,
    )
    train_df = ratings_df.iloc[train_pos]
    train_edge_index = torch.tensor(
        [train_df["user_idx"].values, train_df["movie_idx"].values], dtype=torch.long,
    )
    data = _build_heterodata(train_edge_index, num_users, num_items)

    return {
        "num_users": num_users,
        "num_items": num_items,
        "data": data,
        "train_edge_index": train_edge_index,
        "val_dict": val_dict,
        "test_dict": test_dict,
        "scalar_vec": personality_tensor,
        "minilm_vec": torch.empty((num_users, 0)),
        "true_proxies": true_proxies,
    }


# =========================================================================
# MovieLens-1M / LastFM loader (shared code path)
# =========================================================================

def _load_tabular_dataset(dataset_name: str, data_dir: str) -> dict:
    if dataset_name == "movielens":
        prefix, emb_file = "ml1m", "occupation_embeddings.npy"
        idx_col, item_col = "OccupationID", "MovieID"
    else:
        prefix, emb_file = "lastfm", "country_embeddings.npy"
        idx_col, item_col = "CountryIdx", "ArtistMBID"

    meta_df = pd.read_csv(f"{data_dir}/{prefix}_user_metadata.csv")
    proxies_df = pd.read_csv(f"{data_dir}/{prefix}_proxies.csv")
    interactions_df = pd.read_csv(f"{data_dir}/{prefix}_interactions.csv")

    merged_users = (
        pd.merge(meta_df, proxies_df, on="UserID", how="inner")
        .sort_values("UserID")
        .reset_index(drop=True)
    )
    num_users = len(merged_users)

    user_mapping = {u: i for i, u in enumerate(merged_users["UserID"])}
    interactions_df = interactions_df[interactions_df["UserID"].isin(user_mapping)]
    interactions_df["UserID_idx"] = interactions_df["UserID"].map(user_mapping)

    unique_items = interactions_df[item_col].unique()
    item_mapping = {item: i for i, item in enumerate(unique_items)}
    interactions_df["Item_idx"] = interactions_df[item_col].map(item_mapping)
    num_items = len(unique_items)

    # Per-user random 80/10/10 split
    interactions_df = interactions_df.reset_index(drop=True)
    src_all = interactions_df["UserID_idx"].tolist()
    dst_all = interactions_df["Item_idx"].tolist()
    train_pos, val_dict, test_dict = user_split_80_10_10(
        src_all, dst_all, num_users, seed=42,
    )
    train_df = interactions_df.iloc[train_pos]
    train_edge_index = torch.tensor(
        np.array([train_df["UserID_idx"].values, train_df["Item_idx"].values]),
        dtype=torch.long,
    )

    data = _build_heterodata(train_edge_index, num_users, num_items)

    scalar_vec = torch.tensor(
        merged_users[["Age_Norm", "Gender_Idx"]].values, dtype=torch.float32,
    )

    proxy_cols = [c for c in proxies_df.columns if c not in ("UserID", "genre_dist")]
    true_proxies = torch.tensor(merged_users[proxy_cols].values, dtype=torch.float32)
    true_proxies = (true_proxies - true_proxies.mean(dim=0)) / (true_proxies.std(dim=0) + 1e-6)

    unique_embs = np.load(f"{data_dir}/{prefix}_{emb_file}")
    minilm_vec = torch.tensor(
        unique_embs[merged_users[idx_col].values], dtype=torch.float32,
    )

    return {
        "num_users": num_users,
        "num_items": num_items,
        "data": data,
        "train_edge_index": train_edge_index,
        "val_dict": val_dict,
        "test_dict": test_dict,
        "scalar_vec": scalar_vec,
        "minilm_vec": minilm_vec,
        "true_proxies": true_proxies,
    }


# =========================================================================
# Public API
# =========================================================================

def load_traitalign_dataset(dataset_name: str, data_dir: str) -> dict:
    """Load any of the three supported datasets and return a ready-to-train dict."""
    print(f"Loading {dataset_name} dataset from {data_dir}...")
    if dataset_name == "personality18":
        return _load_personality18()
    return _load_tabular_dataset(dataset_name, data_dir)
