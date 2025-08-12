Validation_Rules.md - Input/Output Contracts
Overview
This document defines the input/output contracts for the Ad-Pod Stitching Server Optimization application. It specifies schemas, validation rules, and contracts to prevent errors and ensure data integrity.
Input Contracts (CSVs)
servers.csv

Columns: server_id (str, unique), location_x (float), location_y (float), setup_cost (float >0), capacity_streams (int >0).
Rules: No duplicates, no NaNs, numeric types enforced, setup_cost >0, capacity >0.
Validation: Pandas: df.duplicated().any(), pd.to_numeric(errors='raise'), df['setup_cost'].min() >0.

dmas.csv

Columns: dma_id (str, unique), location_x (float), location_y (float), demand_streams (int >=0).
Rules: No duplicates, no NaNs, numeric types, demand >=0.
Validation: Similar to servers, allow demand=0 (handle specially).

Output Contracts (JSON)

Schema:
{
"activated_servers": list[str] (unique server_ids),
"assignments": dict[str, str] (dma_id -> server_id, all DMAs covered),
"costs": {
"total_setup_cost": float (>=0),
"total_delivery_cost": float (>=0),
"total_cost": float (>=0)
}
}
Rules: Assignments cover all DMAs, activated_servers subset of inputs, costs sum correctly.
Validation: In tests, assert len(assignments) == num_dmas, sum(costs.values()) consistency.

Internal Contracts

Distance Matrix: Shape (num_servers, num_dmas), dtype=float32, all >=0.
K Range: K_min >=1, K_max <=500, sampled ints.
MILP: Objective matches aggregated costs, constraints enforced.

Enforce via Pydantic models and assertions in code.