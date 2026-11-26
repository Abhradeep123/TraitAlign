"""
Behavioral-proxy computation and MiniLM embedding generation.

GlobalBehavioralProxies  — dataset-agnostic proxy engine (the five canonical
    TraitAlign dimensions plus a loyalty proxy).
TraitAlignPreprocessor   — end-to-end preprocessor that writes CSV/NPY
    artifacts for MovieLens-1M and LastFM-9000.
"""
from __future__ import annotations

import gc
import json
import os
import re
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .config import LastFMPaths, MovieLensPaths, OCCUPATION_MAP


# =========================================================================
# Shared behavioural-proxy engine
# =========================================================================

class GlobalBehavioralProxies:
    """Dataset-agnostic behavioral proxy engine.

    Computes six behavioural dimensions from an interaction table
    (user_id, item_id[, weight, timestamp]) plus per-item content embeddings:

        diversity, novelty, temporal_stability,
        exploration, cross_category, loyalty

    Pseudo-categories are derived by k-means clustering item embeddings so
    no genre/tag column is required for any dataset.
    """

    PROXY_COLUMNS = [
        "diversity", "novelty", "temporal_stability",
        "exploration", "cross_category", "loyalty",
    ]

    def __init__(
        self,
        n_clusters: int = 20,
        chunk_size: int = 50,
        min_interactions: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.chunk_size = chunk_size
        self.min_interactions = min_interactions
        self.random_state = random_state
        self._embedding_lookup: dict = {}
        self._item_cluster: dict = {}
        self._item_popularity: dict = {}

    # ------------------------------------------------------------------
    # Fit shared, per-dataset structures (call once per dataset)
    # ------------------------------------------------------------------

    def fit_item_structure(
        self, item_ids, item_embeddings: np.ndarray
    ) -> GlobalBehavioralProxies:
        """``item_ids`` aligned row-for-row with ``item_embeddings`` (n_items, dim)."""
        item_ids = list(item_ids)
        self._embedding_lookup = dict(zip(item_ids, item_embeddings))
        k = max(2, min(self.n_clusters, len(item_ids)))
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(item_embeddings)
        self._item_cluster = dict(zip(item_ids, labels))
        return self

    def fit_popularity(
        self,
        interactions: pd.DataFrame,
        item_col: str,
        weight_col: Optional[str] = None,
    ) -> GlobalBehavioralProxies:
        """Compute item popularity.

        ``weight_col=None`` treats each row as one unit of consumption.
        ``weight_col="Plays"`` (etc.) sums that column per item.
        """
        if weight_col is not None:
            counts = interactions.groupby(item_col)[weight_col].sum()
        else:
            counts = interactions.groupby(item_col).size()
        total = counts.sum()
        self._item_popularity = (counts / total).to_dict()
        return self

    # ------------------------------------------------------------------
    # Per-user proxy math
    # ------------------------------------------------------------------

    def _diversity(self, item_ids) -> float:
        embs = [self._embedding_lookup[i] for i in item_ids if i in self._embedding_lookup]
        if len(embs) < 2:
            return 0.0
        arr = np.asarray(embs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        sims = (arr / norms) @ (arr / norms).T
        iu = np.triu_indices(len(arr), k=1)
        return float((1.0 - sims[iu]).mean())

    def _novelty(self, item_ids) -> float:
        probs = [self._item_popularity.get(i, 1e-9) for i in item_ids]
        if not probs:
            return 0.0
        return float(np.mean([-np.log(p + 1e-9) for p in probs]))

    def _loyalty(self, items_with_weights) -> float:
        """Herfindahl–Hirschman concentration of consumption weight."""
        if not items_with_weights:
            return 0.0
        df = pd.DataFrame(items_with_weights, columns=["item", "w"])
        agg = df.groupby("item")["w"].sum()
        total = agg.sum()
        if total <= 0:
            return 0.0
        shares = agg / total
        return float((shares ** 2).sum())

    def _exploration_and_cross_category(self, item_ids):
        clusters = [self._item_cluster[i] for i in item_ids if i in self._item_cluster]
        if not clusters:
            return 0.0, 0.0
        counts = pd.Series(clusters).value_counts()
        top3 = set(counts.head(3).index)
        exploration = float(sum(1 for c in clusters if c not in top3) / len(clusters))
        probs = counts / counts.sum()
        cross_category = float(entropy(probs))
        return exploration, cross_category

    def _temporal_stability(self, item_ids_time_ordered) -> float:
        clusters = [
            self._item_cluster[i]
            for i in item_ids_time_ordered
            if i in self._item_cluster
        ]
        if len(clusters) < 2:
            return 0.0
        chunks = [
            clusters[i : i + self.chunk_size]
            for i in range(0, len(clusters), self.chunk_size)
        ]
        if len(chunks) < 2:
            return 0.0
        corrs = []
        for c1, c2 in zip(chunks[:-1], chunks[1:]):
            vc1 = pd.Series(c1).value_counts()
            vc2 = pd.Series(c2).value_counts()
            common = set(vc1.index) | set(vc2.index)
            if len(common) > 1:
                v1 = [vc1.get(c, 0) for c in common]
                v2 = [vc2.get(c, 0) for c in common]
                rho, _ = spearmanr(v1, v2)
                if not np.isnan(rho):
                    corrs.append(rho)
        return float(np.mean(corrs)) if corrs else 0.0

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        interactions: pd.DataFrame,
        user_col: str,
        item_col: str,
        timestamp_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        deconfound: bool = True,
        n_degree_bins: int = 10,
    ) -> pd.DataFrame:
        """One interaction table → one proxies DataFrame."""
        records = []
        for user_id, group in interactions.groupby(user_col):
            if len(group) < self.min_interactions:
                continue
            items = group[item_col].tolist()

            diversity = self._diversity(items)
            novelty = self._novelty(items)
            exploration, cross_category = self._exploration_and_cross_category(items)

            if weight_col and weight_col in group.columns:
                weights = group[weight_col].tolist()
            else:
                weights = [1.0] * len(items)
            loyalty = self._loyalty(list(zip(items, weights)))

            if timestamp_col and timestamp_col in group.columns:
                ordered_items = group.sort_values(timestamp_col)[item_col].tolist()
                temporal_stability = self._temporal_stability(ordered_items)
            else:
                temporal_stability = 0.0

            records.append({
                user_col: user_id,
                "diversity": diversity,
                "novelty": novelty,
                "temporal_stability": temporal_stability,
                "exploration": exploration,
                "cross_category": cross_category,
                "loyalty": loyalty,
                "_degree": len(group),
            })

        df = pd.DataFrame(records)

        # Joint min-max scaling to [0, 1]
        for col in self.PROXY_COLUMNS:
            c_min, c_max = df[col].min(), df[col].max()
            df[col] = (df[col] - c_min) / (c_max - c_min + 1e-8)

        if deconfound:
            df = self._deconfound(df, n_degree_bins)

        return df.drop(columns=["_degree"])

    @staticmethod
    def _deconfound(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Degree-stratified back-door adjustment."""
        bins = pd.qcut(df["_degree"], q=n_bins, labels=False, duplicates="drop")
        out = df.copy()
        for col in GlobalBehavioralProxies.PROXY_COLUMNS:
            decile_mean = out.groupby(bins)[col].transform("mean")
            out[col] = out[col] - decile_mean
        return out


# =========================================================================
# End-to-end preprocessor (writes CSV / NPY artifacts)
# =========================================================================

class TraitAlignPreprocessor:
    """Generates behavioral proxies and MiniLM side-information embeddings
    for the TraitAlign GNN pipeline."""

    def __init__(
        self,
        output_dir: str = ".",
        verbose: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.output_dir = output_dir
        self.verbose = verbose
        self.embedding_model_name = embedding_model
        self._model = None  # lazy-loaded SentenceTransformer
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _path(self, filename: str, subfolder: Optional[str] = None) -> str:
        target_dir = (
            os.path.join(self.output_dir, subfolder) if subfolder else self.output_dir
        )
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, filename)

    @property
    def model(self):
        """Lazily load (and cache) the SentenceTransformer embedding model."""
        if self._model is None:
            self._log(f"Loading embedding model '{self.embedding_model_name}'...")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    @staticmethod
    def _extract_year(title) -> int:
        match = re.search(r"\((\d{4})\)", str(title))
        return int(match.group(1)) if match else 1995

    @staticmethod
    def _shannon_entropy(series: pd.Series) -> float:
        counts = series.value_counts(normalize=True)
        return entropy(counts) if len(counts) > 0 else 0.0

    def _zip_files(
        self, zip_name: str, file_paths: list[str], subfolder: Optional[str] = None
    ) -> None:
        zip_path = self._path(zip_name, subfolder)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in file_paths:
                if os.path.exists(file):
                    zipf.write(file, os.path.basename(file))
        self._log(f"--> Created {zip_path} successfully!")

    @staticmethod
    def _min_max_normalize(
        df: pd.DataFrame, columns: list[str], prefix: str = "raw_"
    ) -> pd.DataFrame:
        for col in columns:
            col_min, col_max = df[col].min(), df[col].max()
            df[col.replace(prefix, "")] = (df[col] - col_min) / (col_max - col_min + 1e-8)
        return df

    # ------------------------------------------------------------------
    # MovieLens-1M
    # ------------------------------------------------------------------

    def run_movielens(
        self,
        users_path: str,
        movies_path: str,
        ratings_path: str,
        dataset_name: str = "traitalign-movielens1m",
        create_zip: bool = False,
        n_clusters: int = 20,
    ) -> Optional[dict]:
        self._log("\n==============================================")
        self._log("🎬 STARTING MOVIELENS 1M PREPROCESSING 🎬")
        self._log("==============================================")

        if not os.path.exists(users_path):
            self._log(f"Skipping MovieLens... could not find {users_path}")
            return None

        df_u = pd.read_csv(
            users_path, sep="::", engine="python", header=None,
            names=["UserID", "Gender", "Age", "OccupationID", "Zip"],
        )
        df_m = pd.read_csv(
            movies_path, sep="::", engine="python", header=None,
            names=["MovieID", "Title", "Genres"], encoding="latin-1",
        )
        df_r = pd.read_csv(
            ratings_path, sep="::", engine="python", header=None,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )

        self._log("Encoding item content (title + genres) with MiniLM...")
        item_text = (
            df_m["Title"].fillna("Unknown Title").astype(str)
            + ". "
            + df_m["Genres"].fillna("Unknown").astype(str).str.replace("|", ", ", regex=False)
            + "."
        )
        item_embeddings = self.model.encode(
            item_text.tolist(), show_progress_bar=self.verbose, convert_to_numpy=True,
        )

        self._log(
            "Calculating Global Behavioral Proxies "
            "(diversity, novelty, temporal_stability, exploration, cross_category)..."
        )
        proxy_engine = GlobalBehavioralProxies(n_clusters=n_clusters)
        proxy_engine.fit_item_structure(df_m["MovieID"].tolist(), item_embeddings)
        proxy_engine.fit_popularity(df_r, item_col="MovieID")
        final_proxies = proxy_engine.compute(
            df_r, user_col="UserID", item_col="MovieID",
            timestamp_col="Timestamp", weight_col=None,
        )

        df_u["Age_Norm"] = df_u["Age"] / 100.0
        df_u["Gender_Idx"] = (
            df_u["Gender"].map({"M": 0, "F": 1, "m": 0, "f": 1}).fillna(0).astype(int)
        )
        user_metadata = df_u[["UserID", "Gender_Idx", "Age_Norm", "OccupationID"]]

        self._log("Running MiniLM for Occupations...")
        occ_embeddings = (
            self.model.encode(
                [OCCUPATION_MAP[i] for i in range(21)], convert_to_tensor=True,
            )
            .cpu()
            .numpy()
        )

        proxies_file = self._path("ml1m_proxies.csv", dataset_name)
        metadata_file = self._path("ml1m_user_metadata.csv", dataset_name)
        interactions_file = self._path("ml1m_interactions.csv", dataset_name)
        embeddings_file = self._path("ml1m_occupation_embeddings.npy", dataset_name)

        final_proxies.to_csv(proxies_file, index=False)
        user_metadata.to_csv(metadata_file, index=False)
        df_r.to_csv(interactions_file, index=False)
        np.save(embeddings_file, occ_embeddings)

        result = {
            "proxies": proxies_file,
            "user_metadata": metadata_file,
            "interactions": interactions_file,
            "occupation_embeddings": embeddings_file,
        }

        if create_zip:
            self._zip_files(
                "movielens_traitalign_data.zip",
                [proxies_file, metadata_file, interactions_file, embeddings_file],
                dataset_name,
            )
            result["zip"] = self._path("movielens_traitalign_data.zip", dataset_name)

        self._log("✅ MovieLens Prep Complete!")
        return result

    # ------------------------------------------------------------------
    # LastFM-9000
    # ------------------------------------------------------------------

    def run_lastfm(
        self,
        users_path: str,
        plays_path: str,
        dataset_name: str = "traitalign-lastfm",
        create_zip: bool = False,
        n_clusters: int = 20,
    ) -> Optional[dict]:
        self._log("\n==============================================")
        self._log("🎵 STARTING LASTFM 360K PREPROCESSING 🎵")
        self._log("==============================================")

        if not os.path.exists(users_path):
            self._log(f"Skipping LastFM... could not find {users_path}")
            return None

        df_u = self._read_lastfm_users(users_path)
        df_u = df_u.dropna(subset=["Gender", "Age", "Country"])
        df_u["Age"] = pd.to_numeric(df_u["Age"], errors="coerce")
        df_u = df_u[(df_u["Age"] >= 10) & (df_u["Age"] <= 100)]

        sampled_users = df_u.sample(n=len(df_u), random_state=42)
        valid_user_ids = set(sampled_users["UserID"])

        self._log(f"Extracting interactions for {len(valid_user_ids)} users...")
        df_plays = self._read_lastfm_plays(plays_path, valid_user_ids)

        self._log("Encoding item content (artist name) with MiniLM...")
        unique_artists = df_plays[["ArtistMBID", "ArtistName"]].drop_duplicates(subset="ArtistMBID")
        artist_embeddings = self.model.encode(
            unique_artists["ArtistName"].fillna("Unknown Artist").astype(str).tolist(),
            show_progress_bar=self.verbose, convert_to_numpy=True,
        )

        self._log(
            "Calculating Global Behavioral Proxies "
            "(diversity, novelty, temporal_stability, exploration, cross_category)..."
        )
        proxy_engine = GlobalBehavioralProxies(n_clusters=n_clusters)
        proxy_engine.fit_item_structure(unique_artists["ArtistMBID"].tolist(), artist_embeddings)
        proxy_engine.fit_popularity(df_plays, item_col="ArtistMBID", weight_col="Plays")
        final_proxies = proxy_engine.compute(
            df_plays, user_col="UserID", item_col="ArtistMBID",
            timestamp_col=None, weight_col="Plays",
        )

        sampled_users = sampled_users.copy()
        sampled_users["Age_Norm"] = sampled_users["Age"] / 100.0
        sampled_users["Gender_Idx"] = (
            sampled_users["Gender"].map({"m": 0, "f": 1, "M": 0, "F": 1}).fillna(0).astype(int)
        )
        user_metadata = sampled_users[["UserID", "Gender_Idx", "Age_Norm", "Country"]]

        self._log("Running MiniLM for Countries...")
        unique_countries = sorted(sampled_users["Country"].unique().tolist())
        country_embeddings = (
            self.model.encode(unique_countries, convert_to_tensor=True).cpu().numpy()
        )
        country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}

        user_metadata = user_metadata.copy()
        user_metadata["CountryIdx"] = user_metadata["Country"].map(country_to_idx)
        user_metadata = user_metadata.drop(columns=["Country"])

        proxies_file = self._path("lastfm_proxies.csv", dataset_name)
        metadata_file = self._path("lastfm_user_metadata.csv", dataset_name)
        interactions_file = self._path("lastfm_interactions.csv", dataset_name)
        embeddings_file = self._path("lastfm_country_embeddings.npy", dataset_name)
        idx_file = self._path("lastfm_country_to_idx.json", dataset_name)

        final_proxies.to_csv(proxies_file, index=False)
        user_metadata.to_csv(metadata_file, index=False)
        df_plays[["UserID", "ArtistMBID", "Plays"]].to_csv(interactions_file, index=False)
        np.save(embeddings_file, country_embeddings)
        with open(idx_file, "w") as f:
            json.dump(country_to_idx, f)

        result = {
            "proxies": proxies_file,
            "user_metadata": metadata_file,
            "interactions": interactions_file,
            "country_embeddings": embeddings_file,
            "country_to_idx": idx_file,
        }

        if create_zip:
            self._zip_files(
                "lastfm_traitalign_data.zip",
                [proxies_file, metadata_file, interactions_file, embeddings_file, idx_file],
                dataset_name,
            )
            result["zip"] = self._path("lastfm_traitalign_data.zip", dataset_name)

        self._log("✅ LastFM Prep Complete!")
        del df_plays
        gc.collect()
        return result

    # ------------------------------------------------------------------
    # LastFM I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_lastfm_users(users_path: str) -> pd.DataFrame:
        df_u = pd.read_csv(users_path)

        gender_series = df_u[["f", "m"]].idxmax(axis=1)

        start_idx = df_u.columns.get_loc("Afghanistan")
        end_idx = df_u.columns.get_loc("Zimbabwe")
        country_cols = df_u.columns[start_idx : end_idx + 1]
        country_series = df_u[country_cols].idxmax(axis=1)

        age_cols = df_u.columns[end_idx + 1 :]
        age_series = df_u[age_cols].idxmax(axis=1).astype(int)

        return pd.DataFrame({
            "UserID": df_u["id"],
            "Gender": gender_series,
            "Age": age_series,
            "Country": country_series,
        })

    @staticmethod
    def _read_lastfm_plays(plays_path: str, valid_user_ids: set) -> pd.DataFrame:
        col_names = ["Unnamed: 0", "user_id", "artist_mbid", "artist_name", "plays"]
        filtered_chunks: list[pd.DataFrame] = []
        try:
            reader = pd.read_csv(plays_path, sep=",", header=None, names=col_names, chunksize=1_000_000)
            for chunk in reader:
                filtered_chunks.append(chunk[chunk["user_id"].isin(valid_user_ids)])
        except Exception:
            filtered_chunks = []
            reader = pd.read_csv(plays_path, sep="\t", header=None, names=col_names, chunksize=1_000_000)
            for chunk in reader:
                filtered_chunks.append(chunk[chunk["user_id"].isin(valid_user_ids)])

        df_plays = pd.concat(filtered_chunks, ignore_index=True)
        del filtered_chunks
        gc.collect()

        if "Unnamed: 0" in df_plays.columns:
            df_plays = df_plays.drop(columns=["Unnamed: 0"])

        df_plays = df_plays.rename(columns={
            "user_id": "UserID",
            "artist_mbid": "ArtistMBID",
            "artist_name": "ArtistName",
            "plays": "Plays",
        })
        df_plays["Plays"] = pd.to_numeric(df_plays["Plays"], errors="coerce")
        df_plays = df_plays.dropna(subset=["Plays"])
        return df_plays

    # ------------------------------------------------------------------
    # Run all preprocessors
    # ------------------------------------------------------------------

    def run_all(
        self,
        movielens_paths: Optional[MovieLensPaths] = None,
        lastfm_paths: Optional[LastFMPaths] = None,
    ) -> dict:
        if movielens_paths is None:
            movielens_paths = MovieLensPaths()
        if lastfm_paths is None:
            lastfm_paths = LastFMPaths()

        results = {
            "movielens": self.run_movielens(
                movielens_paths.users, movielens_paths.movies, movielens_paths.ratings,
            ),
            "lastfm": self.run_lastfm(lastfm_paths.users, lastfm_paths.plays),
        }
        self._log("\n🎉 All Processing Done!")
        return results
