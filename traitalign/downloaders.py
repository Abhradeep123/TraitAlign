"""
Dataset download helpers.

Each class wraps the download + extraction logic for a single raw dataset
and is idempotent (skips work when the target directory already has files).
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile

import kagglehub


# ---------------------------------------------------------------------------
# MovieLens-25M (direct HTTP)
# ---------------------------------------------------------------------------

class MovieLensDownloader:
    def __init__(self) -> None:
        self.url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        self.base_save_dir = os.path.abspath("Datasets")
        self.dataset_name = "movielens-25m"
        self.save_dir = os.path.join(self.base_save_dir, self.dataset_name)
        self.zip_path = os.path.join(self.save_dir, "ml-25m.zip")

    def run(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        if any(f.endswith(".csv") for f in os.listdir(self.save_dir)):
            return

        if not os.path.exists(self.zip_path):
            print(f"\nDownloading {self.dataset_name}. Please wait, this might take a moment...")
            urllib.request.urlretrieve(self.url, self.zip_path)
            print("Download complete!")

        print("Extracting CSV files...")
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if filename:
                    print(f"Extracting: {filename}")
                    source = zip_ref.open(member)
                    target_path = os.path.join(self.save_dir, filename)
                    with open(target_path, "wb") as target:
                        target.write(source.read())
        print("Done!")


# ---------------------------------------------------------------------------
# Personality-18 (Kaggle)
# ---------------------------------------------------------------------------

class PersonalityDatasetDownloader:
    def __init__(self) -> None:
        self.kaggle_dataset_id = "arslanali4343/top-personality-dataset"
        self.base_save_dir = os.path.abspath("Datasets")
        self.dataset_name = "top-personality-dataset"
        self.save_dir = os.path.join(self.base_save_dir, self.dataset_name)

    def run(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        if any(f.endswith(".csv") for f in os.listdir(self.save_dir)):
            print(f"{self.dataset_name} already exists at {self.save_dir}, skipping download.")
            return

        print(f"\nDownloading {self.dataset_name} via kagglehub. Please wait, this might take a moment...")
        download_path = kagglehub.dataset_download(self.kaggle_dataset_id)
        print("Download complete!")

        print("Copying CSV files...")
        for root, _, files in os.walk(download_path):
            for filename in files:
                if filename.endswith(".csv"):
                    src = os.path.join(root, filename)
                    dst = os.path.join(self.save_dir, filename)
                    print(f"Copying: {filename}")
                    shutil.copy2(src, dst)
        print("Done!")


# ---------------------------------------------------------------------------
# MovieLens-1M (Kaggle)
# ---------------------------------------------------------------------------

class MovieLens1MDatasetDownloader:
    def __init__(self) -> None:
        self.kaggle_dataset_id = "odedgolden/movielens-1m-dataset"
        self.base_save_dir = os.path.abspath("Datasets/raw")
        self.dataset_name = "movielens-1m"
        self.save_dir = os.path.join(self.base_save_dir, self.dataset_name)
        self.target_extensions = (".dat", ".csv", ".txt")

    def run(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        existing = [f for f in os.listdir(self.save_dir) if f.endswith(self.target_extensions)]
        if existing:
            print(f"{self.dataset_name} already exists at {self.save_dir}, skipping download.")
            return

        print(f"\nDownloading {self.dataset_name} via kagglehub. Please wait, this might take a moment...")
        download_path = kagglehub.dataset_download(self.kaggle_dataset_id)
        print("Download complete!")

        print("Copying requested files...")
        for root, _, files in os.walk(download_path):
            for filename in files:
                if filename.endswith(self.target_extensions):
                    src = os.path.join(root, filename)
                    dst = os.path.join(self.save_dir, filename)
                    print(f"Copying: {filename}")
                    shutil.copy2(src, dst)
        print("Done!")


# ---------------------------------------------------------------------------
# LastFM-9000 (Kaggle)
# ---------------------------------------------------------------------------

class LastfmDatasetDownloader:
    def __init__(self) -> None:
        self.kaggle_dataset_id = "himanshu7d3/last-fm9000"
        self.base_save_dir = os.path.abspath("Datasets/raw")
        self.dataset_name = "lastfm"
        self.save_dir = os.path.join(self.base_save_dir, self.dataset_name)
        self.target_extensions = (".tsv", ".csv", ".txt", ".json", ".npy")

    def run(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        existing = [f for f in os.listdir(self.save_dir) if f.endswith(self.target_extensions)]
        if existing:
            print(f"{self.dataset_name} already exists at {self.save_dir}, skipping download.")
            return

        print(f"\nDownloading {self.dataset_name} via kagglehub. Please wait, this might take a moment...")
        download_path = kagglehub.dataset_download(self.kaggle_dataset_id)
        print("Download complete!")

        print("Copying requested files...")
        for root, _, files in os.walk(download_path):
            for filename in files:
                if filename.endswith(self.target_extensions):
                    src = os.path.join(root, filename)
                    dst = os.path.join(self.save_dir, filename)
                    print(f"Copying: {filename}")
                    shutil.copy2(src, dst)
        print("Done!")


# ---------------------------------------------------------------------------
# Convenience: download all four datasets
# ---------------------------------------------------------------------------

def download_all() -> None:
    """Download every raw dataset the pipeline needs (idempotent)."""
    LastfmDatasetDownloader().run()
    MovieLens1MDatasetDownloader().run()
    MovieLensDownloader().run()
    PersonalityDatasetDownloader().run()
