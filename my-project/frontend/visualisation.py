from __future__ import annotations

from typing import Dict, Iterable, List

import altair as alt
import pandas as pd


def plot_locations(
    servers_df: pd.DataFrame, dmas_df: pd.DataFrame, assignments: Dict[str, str] | None = None
):
    servers_df = servers_df.copy()
    servers_df["type"] = "server"
    dmas_df = dmas_df.copy()
    dmas_df["type"] = "dma"

    servers_plot = alt.Chart(servers_df).mark_point(color="red", size=80).encode(
        x="location_x", y="location_y", tooltip=["server_id", "capacity_streams"]
    )
    dmas_plot = alt.Chart(dmas_df).mark_point(color="blue", size=40).encode(
        x="location_x", y="location_y", tooltip=["dma_id", "demand_streams"]
    )
    return servers_plot + dmas_plot


