## REPO Summary: Ad‑Pod Stitching Server Optimization (CFLP‑Heuristic)

This repository implements a production‑grade, full‑stack Python application that solves the Ad‑Pod Stitching Server Optimization problem via a clustering → reduced CFLP → exact MILP approach, with a greedy fallback. It includes a FastAPI backend, a Streamlit frontend, tests, sample data, deployment artifacts, and extensive docs.

### Repository Layout

```
my-project/
├─ backend/                # FastAPI API + optimization engine
│  ├─ app.py               # Endpoints, I/O handling, logging, CORS
│  ├─ optimization.py      # CFLP-heuristic and greedy fallback
│  ├─ models.py            # Pydantic response models
│  └─ utils.py             # Validation and helper utilities
├─ frontend/               # Streamlit UI
│  ├─ app.py               # Upload, invoke API, render results
│  └─ visualisation.py     # Altair plots (servers + DMAs)
├─ data/                   # Sample CSVs
│  ├─ servers.csv
│  └─ dmas.csv
├─ scripts/
│  └─ generate_sample_data.py  # Synthetic data generator
├─ tests/                  # Pytest suite
│  ├─ conftest.py
│  ├─ test_backend.py
│  ├─ test_frontend.py
│  └─ test_utils.py
├─ docs/                   # Context/engineering docs
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

### Primary Execution Components

- Backend service: `backend.app` (FastAPI)
  - Endpoints: `/health`, `/optimize` (CFLP‑heuristic), `/optimize_fast` (greedy)
  - Logging: `structlog` JSON logs; INFO/ERROR with exception context
  - CORS: allows `http://localhost:8501`

- Optimization engine: `backend.optimization`
  - Heuristic pipeline: KMeans/MiniBatchKMeans → reduced MILP (PuLP+CBC) → DMA assignment recovery
  - Greedy fallback for infeasible or time‑constrained situations

- Frontend: `frontend.app` (Streamlit)
  - Uploads `servers.csv` and `dmas.csv`
  - Calls backend endpoint (`/optimize` or `/optimize_fast`)
  - Displays activated servers, assignments, and cost metrics; provides download of JSON

### Data Schemas and Validation

- Servers CSV (`servers.csv`): columns `server_id, location_x, location_y, setup_cost, capacity_streams`
  - Validation (`backend.utils.validate_servers_dataframe`):
    - Non‑empty; all required columns present
    - Unique `server_id`; no NaNs in numeric columns
    - `setup_cost > 0`; `capacity_streams > 0`

- DMAs CSV (`dmas.csv`): columns `dma_id, location_x, location_y, demand_streams`
  - Validation (`backend.utils.validate_dmas_dataframe`):
    - Non‑empty; all required columns present
    - Unique `dma_id`; no NaNs; `demand_streams ≥ 0`

- API response (`backend.models.OptimizationResponse`):
  - `activated_servers: List[str]`
  - `assignments: Dict[str, str]` (every DMA assigned)
  - `costs: { total_setup_cost, total_delivery_cost, total_cost }` (all ≥ 0)

### End‑to‑End Execution Flow

1) User interaction (Streamlit `frontend/app.py`)
   - Inputs: upload `servers.csv`, `dmas.csv`; choose timeout and “fast” mode
   - On Optimize: POST multipart form to backend (`/optimize` or `/optimize_fast`)
   - Renders JSON response: activated servers, assignment table, costs; provides download

2) Request handling (FastAPI `backend/app.py`)
   - Reads uploaded files into memory and parses with Pandas
   - Validates CSVs using `validate_servers_dataframe` and `validate_dmas_dataframe`
   - Logs request size; delegates to solver:
     - `/optimize` → `run_cflp_heuristic`
     - `/optimize_fast` → `run_greedy_heuristic`
   - Returns `OptimizationResponse` or raises HTTP 400/500 with structured details

3) Optimization pipeline (`backend/optimization.py: run_cflp_heuristic`)
   - Extract arrays (`_extract_arrays`):
     - Server/dma locations (float32); setup_costs (float64); capacities (int64); ids (List[str])
   - Feasibility pre‑check: `sum(capacities) ≥ sum(demand_streams)` else `InfeasibleError`
   - Distance matrix: `cdist(server_locations, dma_locations)` as float32
   - Feasible K range (`backend.utils.compute_k_range`):
     - `k_min = ceil(total_demand / max_server_capacity)` (≥ 1)
     - `k_max = min(500, num_dmas)` (≥ 1)
   - Sample K values: up to 4 integers linearly spaced in [k_min, k_max]
   - Global time budget: `OPTIMIZER_TIME_BUDGET_SECONDS` (default 30s)
   - For each K (respecting time budget):
     - Clustering: `KMeans` or `MiniBatchKMeans` (threshold via `OPTIMIZER_MINIBATCH_THRESHOLD`, default 2000 DMAs), `random_state=42`
     - Aggregate per‑cluster demand (sum of DMA demands by label)
     - Aggregate distances: for every server j and cluster k, sum raw distances from server j to all DMAs in cluster k
     - Reduced CFLP MILP (`_solve_reduced_cflp`):
       - Vars: `y_j ∈ {0,1}` (open server j), `x_{j,k} ∈ {0,1}` (assign cluster k to server j)
       - Objective: minimize `Σ setup_cost_j*y_j + Σ agg_dist_{j,k}*x_{j,k}`
       - Constraints:
         - Assignment: `Σ_j x_{j,k} = 1` ∀k
         - Capacity: `Σ_k demand_k*x_{j,k} ≤ capacity_j*y_j` ∀j
         - Linking: `x_{j,k} ≤ y_j` ∀j,k
       - Solver: PuLP CBC with per‑solve time limit carved from remaining budget
       - Non‑optimal statuses are treated as infeasible for that K (ignored)
     - Recover DMA‑level assignment: map each DMA to the server chosen for its cluster
     - Compute costs:
       - `total_setup_cost = Σ setup_costs[open servers]`
       - `total_delivery_cost = Σ distance(server_assigned_to_dma, dma)` (unweighted by demand)
       - Track best by `total_cost`
     - Early exit on first feasible if `OPTIMIZER_EARLY_FEASIBLE=1` (default)
   - If no feasible K solution found: fallback `_greedy_assign`
   - Build response: activated server IDs, DMA→server map, and cost breakdown

4) Greedy fallback (`backend/optimization.py: run_greedy_heuristic` and `_greedy_assign`)
   - Preconditions and distance computation as above
   - For each DMA, try servers in ascending distance order; assign to first with sufficient remaining capacity
   - Opens a server upon first assignment; errors if any DMA cannot be assigned → `InfeasibleError`
   - Computes costs identically to heuristic pipeline

### Algorithms and Key Logic

- Distance calculation: Euclidean (`scipy.spatial.distance.cdist`) producing an `N_servers × N_dmas` float32 matrix
- Feasible K range: `k_min = ceil(total_demand / max_server_capacity)`, `k_max = min(500, num_dmas)`
- Clustering: `KMeans` for small/medium; `MiniBatchKMeans` beyond `OPTIMIZER_MINIBATCH_THRESHOLD`
- Reduced CFLP MILP: exact formulation on clusters to select open servers and cluster assignments within capacity
- Costs: delivery cost uses sum of raw (unweighted) distances; demands only constrain capacity (design choice documented in `README.md`)
- Time budgeting: total wall‑clock budget (default 30s) divided across K attempts; per‑attempt timeout passed to CBC
- Early termination: return first feasible solution when configured for responsiveness

### Configuration (Environment Variables)

- `OPTIMIZER_TIME_BUDGET_SECONDS` (default: `30`) → global time budget for `/optimize`
- `OPTIMIZER_EARLY_FEASIBLE` (default: `1`) → early exit upon first feasible K solution
- `OPTIMIZER_MINIBATCH_THRESHOLD` (default: `2000`) → switch to `MiniBatchKMeans` when DMAs exceed this size

### API Contract (Backend)

- `GET /health` → `{ "status": "ok" }`
- `POST /optimize` (multipart/form‑data)
  - files: `servers` (csv), `dmas` (csv)
  - 200 → `OptimizationResponse`
  - 400 → validation/infeasible (`InfeasibleError`, CSV schema errors)
  - 500 → unexpected error (guarded by try/except with structured logging)
- `POST /optimize_fast` → same I/O, uses greedy algorithm

### Frontend Behavior (Streamlit)

- Config: `BACKEND_URL = http://localhost:8000`
- Inputs: two CSV uploaders; timeout; fast‑mode toggle
- On optimize: posts to backend; renders:
  - Activated servers (list)
  - Assignments (dataframe of DMA→server)
  - Costs (Streamlit metrics: setup, delivery, total)
  - Download button for JSON result

### Logging, Errors, and Resilience

- Logging: `structlog` JSON with timestamp, level, and structured fields
- Validation errors (400): missing columns, NaNs, duplicates, non‑positive costs/capacities, negative demands
- Infeasible instances (400): total capacity < total demand; greedy fallback exhaustion
- Unexpected errors (500): caught, logged with stack trace; response body is sanitized

### Performance and Scalability

- Memory efficiency: distance matrices as float32; clustering selection by size
- MILP scaling: solved on clusters, not raw DMAs; per‑solve time limit enforced
- Heuristic K sampling: at most 4 K values for responsiveness; early‑feasible exit
- Fallback path: greedy ensures fast solution when MILP times out or is infeasible

### Tests (`tests/`)

- `test_backend.py`: `/health` endpoint; `/optimize` end‑to‑end on small synthetic instance; asserts assignment coverage and response shape
- `test_frontend.py`: module import health and presence of key variables
- `test_utils.py`: unit tests for `compute_k_range` and CSV validators

### Sample Data and Generator

- `data/servers.csv`, `data/dmas.csv`: large synthetic datasets included
- `scripts/generate_sample_data.py`:
  - Generates clustered servers and population‑weighted DMA locations
  - Validates feasibility (capacity vs demand) and prints summary
  - Saves `servers_sample.csv` and `dmas_sample.csv` under `data/`

### Deployment and Local Development

- Local (two terminals from `my-project/`):
  - Backend: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
  - Frontend: `streamlit run frontend/app.py --server.port 8501`

- Docker:
  - `Dockerfile`: installs `requirements.txt`, launches Uvicorn app on 8000
  - `docker-compose.yml`:
    - `backend` builds local image and exposes 8000
    - `frontend` uses `python:3.10-slim`, installs deps, runs Streamlit on 8501, depends on backend

### Notable Design Choices and Limitations

- Delivery cost is modeled as the sum of Euclidean distances per DMA (unweighted by demand). Capacity constraints incorporate demand; this is consistent throughout the pipeline and documented in `README.md`.
- Reduced MILP returns the exact optimal solution for the clustered instance under time limits; global optimality for the original (unclustered) instance is not guaranteed.
- Early‑feasible exit trades optimality for responsiveness; can be disabled via env var.

### How Everything Fits Together

- The Streamlit UI provides a simple operator interface to upload CSVs and see results.
- The FastAPI layer is a thin, validated façade that converts files into dataframes, logs requests, and delegates to the solver.
- The solver balances speed and solution quality via clustering and a reduced MILP, while enforcing feasibility via capacity constraints; it can gracefully degrade to a greedy method.
- Tests and docs ensure the repository is executable, maintainable, and production‑oriented.


