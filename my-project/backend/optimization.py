from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
import pulp

from .utils import InfeasibleError, compute_k_range


def _extract_arrays(servers_df: pd.DataFrame, dmas_df: pd.DataFrame) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
]:
    server_locations = servers_df[["location_x", "location_y"]].to_numpy(dtype=np.float32)
    dma_locations = dmas_df[["location_x", "location_y"]].to_numpy(dtype=np.float32)
    setup_costs = servers_df["setup_cost"].to_numpy(dtype=np.float64)
    capacities = servers_df["capacity_streams"].to_numpy(dtype=np.int64)
    server_ids = servers_df["server_id"].astype(str).tolist()
    dma_ids = dmas_df["dma_id"].astype(str).tolist()
    return server_locations, dma_locations, setup_costs, capacities, server_ids, dma_ids


def _solve_reduced_cflp(
    setup_costs: np.ndarray,
    capacities: np.ndarray,
    aggregated_distances: np.ndarray,
    cluster_demands: np.ndarray,
    time_limit_seconds: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    num_servers, num_clusters = aggregated_distances.shape
    model = pulp.LpProblem("CFLP_Reduced", pulp.LpMinimize)

    y = [pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for j in range(num_servers)]
    x = [
        [pulp.LpVariable(f"x_{j}_{k}", lowBound=0, upBound=1, cat=pulp.LpBinary) for k in range(num_clusters)]
        for j in range(num_servers)
    ]

    model += (
        pulp.lpSum(setup_costs[j] * y[j] for j in range(num_servers))
        + pulp.lpSum(aggregated_distances[j, k] * x[j][k] for j in range(num_servers) for k in range(num_clusters))
    )

    for k in range(num_clusters):
        model += pulp.lpSum(x[j][k] for j in range(num_servers)) == 1

    for j in range(num_servers):
        model += pulp.lpSum(cluster_demands[k] * x[j][k] for k in range(num_clusters)) <= capacities[j] * y[j]

    # Linking constraints: a cluster can be assigned to a server only if that server is activated
    for j in range(num_servers):
        for k in range(num_clusters):
            model += x[j][k] <= y[j]

    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=max(1, int(time_limit_seconds))))

    # Treat only an optimal solution as valid. Any other status yields an infinite objective
    # so the caller will ignore this K configuration.
    if model.status != pulp.LpStatusOptimal:
        return float("inf"), np.zeros(num_servers, dtype=int), np.zeros((num_servers, num_clusters), dtype=int)

    y_sol = np.array([int(var.value() > 0.5) for var in y], dtype=int)
    x_sol = np.array([[int(var.value() > 0.5) for var in row] for row in x], dtype=int)
    obj = pulp.value(model.objective)
    return float(obj), y_sol, x_sol


def _greedy_assign(
    distance_matrix: np.ndarray,
    capacities: np.ndarray,
    demands: np.ndarray,
    server_ids: List[str],
    dma_ids: List[str],
) -> Tuple[Dict[int, int], np.ndarray]:
    assignment: Dict[int, int] = {}
    remaining_capacity = capacities.astype(int).copy()
    open_mask = np.zeros(len(capacities), dtype=int)
    server_order_by_dma = np.argsort(distance_matrix, axis=0)
    for dma_idx in range(len(dma_ids)):
        assigned = False
        demand_needed = int(demands[dma_idx])
        for server_idx in server_order_by_dma[:, dma_idx]:
            if remaining_capacity[server_idx] >= demand_needed:
                assignment[dma_idx] = int(server_idx)
                remaining_capacity[server_idx] -= demand_needed
                open_mask[server_idx] = 1
                assigned = True
                break
        if not assigned:
            raise InfeasibleError("Greedy fallback failed: insufficient capacity")
    return assignment, open_mask


def run_cflp_heuristic(servers_df: pd.DataFrame, dmas_df: pd.DataFrame) -> Dict[str, object]:
    server_locations, dma_locations, setup_costs, capacities, server_ids, dma_ids = _extract_arrays(
        servers_df, dmas_df
    )

    total_capacity = int(capacities.sum())
    total_demand = int(dmas_df["demand_streams"].sum())
    if total_capacity < total_demand:
        raise InfeasibleError("Insufficient server capacity relative to demand")

    distance_matrix = cdist(server_locations, dma_locations).astype(np.float32)

    k_min, k_max = compute_k_range(total_demand, int(capacities.max(initial=0)), len(dma_ids))
    # Sample up to 4 K values between k_min and k_max for responsiveness
    if k_min == k_max:
        k_values = [k_min]
    else:
        k_values = np.unique(np.linspace(k_min, k_max, num=min(4, k_max - k_min + 1), dtype=int)).tolist()

    best_total_cost = float("inf")
    best_assignment: Dict[int, int] = {}
    best_open_mask: np.ndarray | None = None

    demands = dmas_df["demand_streams"].to_numpy(dtype=np.int64)

    # Global time budget to avoid request timeouts (configurable via env)
    start_time = time.monotonic()
    global_time_budget = float(os.getenv("OPTIMIZER_TIME_BUDGET_SECONDS", "30"))
    early_exit_on_first_feasible = os.getenv("OPTIMIZER_EARLY_FEASIBLE", "1") == "1"
    mbkmeans_threshold = int(os.getenv("OPTIMIZER_MINIBATCH_THRESHOLD", "2000"))
    for idx_k, k in enumerate(k_values):
        if k <= 0:
            continue
        # Check remaining budget before clustering
        time_elapsed = time.monotonic() - start_time
        time_remaining = max(0.0, global_time_budget - time_elapsed)
        if time_remaining < 2.0:
            # Not enough time to proceed with clustering + MILP; fallback
            break

        # Choose clustering algorithm based on problem size
        if len(dma_ids) > mbkmeans_threshold:
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=42, batch_size=1024)
        else:
            kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
        labels = kmeans.fit_predict(dma_locations)

        cluster_demands = np.zeros(k, dtype=np.int64)
        for cluster_index in range(k):
            cluster_demands[cluster_index] = int(demands[labels == cluster_index].sum())

        # aggregated delivery cost of assigning an entire cluster to a server:
        # sum over DMAs in the cluster of distance_to_server (unweighted by demand)
        aggregated_distances = np.zeros((len(server_ids), k), dtype=np.float64)
        for cluster_index in range(k):
            mask = labels == cluster_index
            aggregated_distances[:, cluster_index] = distance_matrix[:, mask].sum(axis=1)

        # Allocate a fraction of remaining time to this solve
        time_elapsed = time.monotonic() - start_time
        time_remaining = max(0.0, global_time_budget - time_elapsed)
        # leave some time for post-processing
        per_solve_time = max(1.0, time_remaining / max(1, (len(k_values) - idx_k)))
        obj, y_sol, x_sol = _solve_reduced_cflp(
            setup_costs=setup_costs,
            capacities=capacities,
            aggregated_distances=aggregated_distances,
            cluster_demands=cluster_demands,
            time_limit_seconds=per_solve_time,
        )

        if not np.isfinite(obj):
            continue

        # Recover DMA-level assignment
        # For each cluster, the chosen server is argmax over x[:, cluster]
        chosen_server_by_cluster = np.argmax(x_sol, axis=0)
        assignment = {}
        for dma_idx, cluster_idx in enumerate(labels):
            server_idx = int(chosen_server_by_cluster[cluster_idx])
            assignment[dma_idx] = server_idx

        # Compute exact costs
        total_setup_cost = float(setup_costs[y_sol == 1].sum())
        # delivery cost: sum per dma of distance to its assigned server (unweighted)
        delivery_cost = 0.0
        for dma_idx, server_idx in assignment.items():
            delivery_cost += float(distance_matrix[server_idx, dma_idx])
        total_cost = total_setup_cost + delivery_cost

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_assignment = assignment
            best_open_mask = y_sol.copy()
            # Optionally exit early on first feasible solution for speed
            if early_exit_on_first_feasible:
                break

    if best_open_mask is None:
        # Fallback to a fast greedy assignment
        best_assignment, best_open_mask = _greedy_assign(
            distance_matrix=distance_matrix,
            capacities=capacities,
            demands=demands,
            server_ids=server_ids,
            dma_ids=dma_ids,
        )
        # Note: costs will be computed precisely below

    # Build response mapping ids
    activated_servers = [server_ids[i] for i, open_flag in enumerate(best_open_mask) if open_flag == 1]
    assignments: Dict[str, str] = {dma_ids[i]: server_ids[sidx] for i, sidx in best_assignment.items()}

    # Compute setup cost directly from the setup_costs array using the open mask
    total_setup_cost = float(setup_costs[best_open_mask == 1].sum())
    # Recompute delivery with IDs (unweighted)
    delivery_cost = 0.0
    for dma_index, server_index in best_assignment.items():
        delivery_cost += float(distance_matrix[server_index, dma_index])
    total_delivery_cost = float(delivery_cost)

    return {
        "activated_servers": activated_servers,
        "assignments": assignments,
        "costs": {
            "total_setup_cost": total_setup_cost,
            "total_delivery_cost": total_delivery_cost,
            "total_cost": total_setup_cost + total_delivery_cost,
        },
    }


def run_greedy_heuristic(servers_df: pd.DataFrame, dmas_df: pd.DataFrame) -> Dict[str, object]:
    server_locations = servers_df[["location_x", "location_y"]].to_numpy(dtype=np.float32)
    dma_locations = dmas_df[["location_x", "location_y"]].to_numpy(dtype=np.float32)
    capacities = servers_df["capacity_streams"].to_numpy(dtype=np.int64)
    setup_costs = servers_df["setup_cost"].to_numpy(dtype=np.float64)
    server_ids = servers_df["server_id"].astype(str).tolist()
    dma_ids = dmas_df["dma_id"].astype(str).tolist()
    demands = dmas_df["demand_streams"].to_numpy(dtype=np.int64)

    if int(capacities.sum()) < int(demands.sum()):
        raise InfeasibleError("Insufficient server capacity relative to demand")

    distance_matrix = cdist(server_locations, dma_locations).astype(np.float32)
    assignment, open_mask = _greedy_assign(
        distance_matrix=distance_matrix,
        capacities=capacities,
        demands=demands,
        server_ids=server_ids,
        dma_ids=dma_ids,
    )

    activated_servers = [server_ids[i] for i, open_flag in enumerate(open_mask) if open_flag == 1]
    assignments: Dict[str, str] = {dma_ids[i]: server_ids[sidx] for i, sidx in assignment.items()}
    total_setup_cost = float(setup_costs[open_mask == 1].sum())
    delivery_cost = 0.0
    for dma_index, server_index in assignment.items():
        delivery_cost += float(distance_matrix[server_index, dma_index])
    total_delivery_cost = float(delivery_cost)

    return {
        "activated_servers": activated_servers,
        "assignments": assignments,
        "costs": {
            "total_setup_cost": total_setup_cost,
            "total_delivery_cost": total_delivery_cost,
            "total_cost": total_setup_cost + total_delivery_cost,
        },
    }


