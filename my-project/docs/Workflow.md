Workflow.md - Development Process Automation
Overview
This document outlines the automated development workflow for building and maintaining the Ad-Pod Stitching Server Optimization application. It includes tools, scripts, and processes to ensure error-free code, consistent builds, and efficient iteration.
Workflow Stages
1. Setup

Command: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt.
Automation: Add setup.sh script for one-click env creation.

2. Development

Backend: uvicorn backend.app:app --reload --port 8000.
Frontend: streamlit run frontend/app.py.
Linting: black . && flake8 . (install black, flake8).
Type Checking: mypy . (install mypy).

3. Testing

Unit/Integration: pytest --cov . (install pytest, pytest-cov).
Coverage Target: >=80%.
Automation: Git hooks (pre-commit) for running tests/linters.

4. Data Generation

Run: python scripts/generate_sample_data.py.
Outputs to data/.

5. Build & Deploy

Docker Build: docker build -t ad-pod-optimizer ..
Compose: docker-compose up -d.
CI/CD: Use GitHub Actions YAML for auto-testing on push (template: lint → test → build).

6. Monitoring & Iteration

Logging: Use logging module in code, output to console/file.
Profiling: Use cProfile for performance bottlenecks.
Version Control: Git branches for features, merge via PRs.

Automation Tools

pre-commit: Install and configure for linting/tests.
Makefile: Add for common commands (e.g., make setup, make test).
CI Pipeline: Test on Python 3.10-3.12, ensure no warnings.

This workflow minimizes manual errors and enforces best practices.