Implementation.md - Step-by-Step Build Instructions
Overview
This document provides a detailed, sequential guide for implementing the Ad-Pod Stitching Server Optimization application. Follow these steps to build an error-free, executable full-stack Python application. Each step includes prerequisites, actions, and verification tips. Use the CFLP-Heuristic for optimization.
Prerequisites

Python 3.10+ installed.
Create a virtual environment: python -m venv venv && source venv/bin/activate.
Install dependencies: pip install fastapi uvicorn streamlit pandas numpy scipy scikit-learn pulp requests altair.
Generate requirements.txt: pip freeze > requirements.txt.

Step 1: Set Up Project Structure

Create root directory: mkdir ad-pod-optimizer.
Add subfolders: mkdir backend frontend data tests scripts.
Refer to Project_Structure.md for exact file placements.

Step 2: Implement Backend (FastAPI)

Create backend/app.py, backend/optimization.py, backend/models.py, backend/utils.py.
In app.py: Define FastAPI app, add CORS middleware, create POST /optimize endpoint.
In optimization.py: Implement CFLP-Heuristic logic (data parsing, distance computation, K-loop with clustering and MILP).
In models.py: Use Pydantic for input/output schemas.
In utils.py: Add helpers for validation, distance calc.
Verify: Run uvicorn backend.app:app --reload, test endpoint with curl or Postman using sample CSVs.

Step 3: Implement Frontend (Streamlit)

Create frontend/app.py, frontend/visualisation.py.
In app.py: Set page config, add file uploaders, button to trigger API call, display results.
In visualisation.py: Define functions for plotting (scatter for locations, bar for costs).
Verify: Run streamlit run frontend/app.py, upload samples, check results display.

Step 4: Integrate Optimization Logic

In optimization.py:

Parse CSVs to DataFrames/arrays.
Validate (see Validation_Rules.md).
Compute distances with cdist.
Calculate K_min/K_max, sample K values.
For each K: Cluster with KMeans(random_state=42), aggregate, formulate/solve MILP with PuLP.
Select best, compute exact costs.


Handle errors: Raise exceptions for invalid data, infeasible problems.

Step 5: Add Tests

Create tests/test_backend.py, tests/test_frontend.py.
Use pytest: Test parsing, optimization, API responses, UI rendering.
Run: pytest.

Step 6: Add Sample Data Generation

Create scripts/generate_sample_data.py: Use NumPy random (seed=42) to generate CSVs in data/.

Step 7: Verify End-to-End

Generate samples.
Run backend and frontend.
Upload files, optimize, check JSON matches expected (e.g., sample output in prompt).

Step 8: Optimize for Performance

Use float32, on-the-fly computations.
Refer to Performance_SLA.md.

This guide ensures incremental, testable development leading to a production-ready application.