"""
Shared configuration: path dataclasses, constants, and lookup tables.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Path containers (one per raw dataset)
# ---------------------------------------------------------------------------

@dataclass
class MovieLensPaths:
    users: str = r"Datasets/raw/movielens-1m/users.dat"
    movies: str = r"Datasets/raw/movielens-1m/movies.dat"
    ratings: str = r"Datasets/raw/movielens-1m/ratings.dat"


@dataclass
class LastFMPaths:
    users: str = r"Datasets/raw/lastfm/user_features_9000.csv"
    plays: str = r"Datasets/raw/lastfm/lastfm_9000_users.csv"


# ---------------------------------------------------------------------------
# Training / evaluation constants
# ---------------------------------------------------------------------------

MIN_INTERACTIONS = 5  # users below this threshold are train-only (no val/test)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

OCCUPATION_MAP = {
    0: "other or not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
    19: "unemployed", 20: "writer",
}
