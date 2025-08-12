from __future__ import annotations

import io

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_optimize_small_sample():
    servers_df = pd.DataFrame(
        {
            "server_id": ["S1", "S2"],
            "location_x": [0.0, 10.0],
            "location_y": [0.0, 0.0],
            "setup_cost": [50.0, 60.0],
            "capacity_streams": [10, 10],
        }
    )
    dmas_df = pd.DataFrame(
        {
            "dma_id": ["D1", "D2", "D3"],
            "location_x": [1.0, 2.0, 9.0],
            "location_y": [0.0, 0.0, 0.0],
            "demand_streams": [3, 4, 2],
        }
    )

    servers_buf = io.BytesIO()
    servers_df.to_csv(servers_buf, index=False)
    servers_buf.seek(0)
    dmas_buf = io.BytesIO()
    dmas_df.to_csv(dmas_buf, index=False)
    dmas_buf.seek(0)

    files = {
        "servers": ("servers.csv", servers_buf, "text/csv"),
        "dmas": ("dmas.csv", dmas_buf, "text/csv"),
    }
    r = client.post("/optimize", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "activated_servers" in data
    assert "assignments" in data
    assert "costs" in data
    assert set(data["assignments"].keys()) == set(dmas_df["dma_id"])  # all dmas assigned


