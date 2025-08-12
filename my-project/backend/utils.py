from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


class InfeasibleError(Exception):
    """Raised when the optimization problem is infeasible.

    Typically when total capacity is less than total demand.
    """


def validate_servers_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("servers.csv is empty")
    required = [
        "server_id",
        "location_x",
        "location_y",
        "setup_cost",
        "capacity_streams",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"servers.csv missing columns: {missing}")
    if df["server_id"].duplicated().any():
        raise ValueError("servers.csv has duplicate server_id values")
    for col in ["location_x", "location_y", "setup_cost", "capacity_streams"]:
        if df[col].isna().any():
            raise ValueError(f"servers.csv has NaN in {col}")
    if (df["setup_cost"] <= 0).any():
        raise ValueError("setup_cost must be > 0")
    if (df["capacity_streams"] <= 0).any():
        raise ValueError("capacity_streams must be > 0")


def validate_dmas_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("dmas.csv is empty")
    required = [
        "dma_id",
        "location_x",
        "location_y",
        "demand_streams",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dmas.csv missing columns: {missing}")
    if df["dma_id"].duplicated().any():
        raise ValueError("dmas.csv has duplicate dma_id values")
    for col in ["location_x", "location_y", "demand_streams"]:
        if df[col].isna().any():
            raise ValueError(f"dmas.csv has NaN in {col}")
    if (df["demand_streams"] < 0).any():
        raise ValueError("demand_streams must be >= 0")


def compute_k_range(total_demand: int, max_capacity: int, num_dmas: int) -> Tuple[int, int]:
    k_min = int(np.ceil(total_demand / max(1, max_capacity)))
    k_min = max(1, k_min)
    k_max = max(1, min(500, num_dmas))
    return k_min, k_max


