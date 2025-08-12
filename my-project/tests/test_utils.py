from __future__ import annotations

import pandas as pd

from backend.utils import compute_k_range, validate_dmas_dataframe, validate_servers_dataframe


def test_compute_k_range():
    kmin, kmax = compute_k_range(total_demand=25, max_capacity=10, num_dmas=7)
    assert kmin == 3
    assert kmax == 7


def test_validate_servers_dataframe():
    df = pd.DataFrame(
        {
            "server_id": ["S1", "S2"],
            "location_x": [0.0, 1.0],
            "location_y": [0.0, 1.0],
            "setup_cost": [10.0, 20.0],
            "capacity_streams": [5, 5],
        }
    )
    validate_servers_dataframe(df)


def test_validate_dmas_dataframe():
    df = pd.DataFrame(
        {
            "dma_id": ["D1", "D2"],
            "location_x": [0.0, 1.0],
            "location_y": [0.0, 1.0],
            "demand_streams": [0, 2],
        }
    )
    validate_dmas_dataframe(df)


