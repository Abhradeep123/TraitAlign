"""
MiniLM movie-embedding generator (Title + Genres).

Produces a (num_movies, 385) array where column 0 is the movieId and
columns 1–384 are the ``all-MiniLM-L6-v2`` semantic embedding.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Encapsulates the full MiniLM movie-embedding generation pipeline."""

    def __init__(
        self,
        movies_path: str = os.path.join("Datasets", "movielens-25m", "movies.csv"),
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 256,
        output_path: str = "P_i.npy",
    ) -> None:
        self.movies_path = movies_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.movies_df: pd.DataFrame | None = None
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.P_i: np.ndarray | None = None

    # Step 1 ---------------------------------------------------------------

    def load_movies(self) -> pd.DataFrame:
        print("Loading MovieLens Data...")
        self.movies_df = pd.read_csv(self.movies_path)
        self.movies_df["clean_genres"] = self.movies_df["genres"].str.replace("|", ", ")
        self.movies_df["clean_genres"] = self.movies_df["clean_genres"].replace(
            "(no genres listed)", "",
        )
        self.movies_df["semantic_text"] = (
            self.movies_df["title"] + ". " + self.movies_df["clean_genres"] + "."
        )
        print(f"Total Movies to Encode: {len(self.movies_df)}")
        print("\n--- Sample Semantic Strings ---")
        for text in self.movies_df["semantic_text"].head(3):
            print(text)
        return self.movies_df

    # Step 2 ---------------------------------------------------------------

    def init_model(self) -> SentenceTransformer:
        print("Initializing MiniLM model on GPU...")
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    # Step 3 ---------------------------------------------------------------

    def generate_embeddings(self) -> np.ndarray:
        if self.movies_df is None:
            raise RuntimeError("Call load_movies() before generate_embeddings().")
        if self.model is None:
            raise RuntimeError("Call init_model() before generate_embeddings().")

        print("Generating Embeddings...")
        texts_to_encode = self.movies_df["semantic_text"].tolist()
        self.embeddings = self.model.encode(
            texts_to_encode, batch_size=self.batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        )
        print(f"Raw Embeddings Shape: {self.embeddings.shape}")
        return self.embeddings

    # Step 4 ---------------------------------------------------------------

    def build_output_array(self) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call generate_embeddings() before build_output_array().")

        print("Constructing final P_i.npy array...")
        movie_ids = self.movies_df["movieId"].values.reshape(-1, 1)
        self.P_i = np.hstack((movie_ids, self.embeddings))
        print(f"Final P_i Shape: {self.P_i.shape}")
        return self.P_i

    # Step 5 ---------------------------------------------------------------

    def save(self, output_path: str | None = None) -> str:
        if self.P_i is None:
            raise RuntimeError("Call build_output_array() before save().")
        path = output_path or self.output_path
        np.save(path, self.P_i)
        print(f"Saved successfully to {path}!")
        return path

    # Step 6 ---------------------------------------------------------------

    def diagnostic_check(self) -> float:
        if self.P_i is None:
            raise RuntimeError("Call build_output_array() before diagnostic_check().")

        print("================================================")
        print("         FINAL DIAGNOSTIC CHECK")
        print("================================================")
        emb_vectors = self.P_i[:, 1:]
        variance = np.var(emb_vectors)
        print(f"Variance of embeddings: {variance:.6f}")

        if np.isnan(variance):
            print("🚨 FAILURE: Embeddings contain NaNs!")
        elif variance == 0.0:
            print("🚨 FAILURE: Variance is 0.0! All embeddings are completely identical.")
        else:
            print("✅ SUCCESS! Embeddings are mathematically valid and ready for TraitAlign.")
        return variance

    # Convenience -----------------------------------------------------------

    def run(self, output_path: str | None = None) -> np.ndarray:
        """Execute the entire pipeline end to end."""
        self.load_movies()
        self.init_model()
        self.generate_embeddings()
        self.build_output_array()
        self.save(output_path)
        self.diagnostic_check()
        return self.P_i
