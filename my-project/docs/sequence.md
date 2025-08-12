
1.Initialize Project Structure
Create project root directory with subfolders:
- backend/
- frontend/
- tests/
- data/
- docs/ (containing all .md files)
- infra/


2.Generate Core Implementation Files
Create backend/optimization.py implementing CFLP heuristic with:
- Chunked distance matrix calculation (float32)
- K-means clustering with random_state=42
- PuLP MILP solver with 25s timeout
- Fallback to greedy algorithm
- Solution validation


3.Build API Endpoint
Create backend/app.py with:
- FastAPI POST /optimize endpoint
- CSV validation using Pydantic models
- ThreadPoolExecutor for async processing
- Structured error responses (400/408/500)
- CORS middleware for Streamlit
- Prometheus instrumentation

4.Develop Streamlit UI
Create frontend/app.py with:
- CSV upload components
- Optimization progress display
- Assignment visualization (Altair/Matplotlib)
- Cost breakdown charts
- Error handling display
- @st.cache_data decorators


5.Implement Test Suite
Create tests/test_backend.py with:
- Property-based tests using Hypothesis
- Fuzz tests for invalid CSV inputs
- Performance benchmarks (pytest-benchmark)
- Memory usage tests
- API contract validation


6.Generate Infrastructure Files
Create infra/Dockerfile:
- Multi-stage build (builder → slim)
- Non-root user
- Health checks
- Resource limits

Create infra/docker-compose.yml:
- Backend service (port 8000)
- Frontend service (port 8501)
- Resource constraints



7.Create Support Scripts
Create scripts/generate_sample_data.py:
- Configurable dataset sizes
- Geographic distribution logic
- CSV validation rules
- Sample outputs for tests



8.Build CI/CD Pipeline
Create .github/workflows/ci.yml:
- pytest with coverage
- Security scanning (safety, bandit)
- Performance regression testing
- Docker build validation


9.Generate Documentation
Create README.md with:
- Installation instructions (venv/conda/docker)
- API usage examples
- Sample optimization requests/responses
- Troubleshooting guide




Execution Sequence
Start with core optimization engine
Cursor: Generate backend/optimization.py based on Implementation.md and Performance_SLA.md

Build API wrapper
Cursor: Create backend/app.py using API_Contract.md and Error_Handbook.md

Develop UI components
Cursor: Implement frontend/app.py following UI_UX_Spec.md

Create test harness
Cursor: Generate tests using Validation_Rules.md and Performance_SLA.md

Setup deployment artifacts
Cursor: Build infra files from Deployment_Playbook.md

Generate sample data
Cursor: Create scripts/generate_sample_data.py with configurable sizes

Assemble documentation
Cursor: Compile README.md from all .md files



Critical Quality Checks Before Execution
Cross-verify against Architecture.md

Confirm data flow matches design

Validate component interactions

Test against Performance_SLA.md

Run benchmarks with 50K DMAs

Verify <30s execution time

Monitor memory usage

Validate Error Handbook Coverage

Simulate all error codes

Test circuit breaker behavior

Verify log sanitization

Check API Contract Compliance

Validate request/response formats

Test rate limiting

Verify CORS headers



Recommended first command
Generate backend/optimization.py with:
- Type-hinted functions
- Numpy vectorization for distances
- KMeans clustering with fixed seed
- PuLP MILP formulation
- Memory-safe chunk processing
- Solution validation
- Timeout handling
- Fallback mechanism




Here's a step-by-step plan:
1. **Set up the project structure**:
   - Create the directories and files as specified in `Project_Structure.md`.
   - Ensure the `.cursorrules` and `.cursorignore` files are in the root.
2. **Generate the backend code**:
   - Focus on the FastAPI application as per `Architecture.md` and `API_Contract.md`.
   - Implement the optimization engine based on `Implementation.md` and `Performance_SLA.md`.
   - Include error handling as per `Error_Handbook.md` and validation as per `Validation_Rules.md`.
3. **Generate the frontend code**:
   - Create the Streamlit UI as per `UI_UX_Spec.md`.
   - Ensure it communicates with the backend using the API contract.
4. **Write tests**:
   - Generate tests for backend and frontend as per `Workflow.md` and `Validation_Rules.md`.
   - Include property-based tests, fuzzing, and performance tests.
5. **Create supporting files**:
   - Generate `Dockerfile` and `docker-compose.yml` as per `Deployment_Playbook.md`.
   - Create `requirements.txt` with all dependencies.
   - Write a comprehensive `README.md` for setup and usage.
6. **Run the sample**:
   - Use the data generator to create sample CSVs.
   - Run the application and capture the output for verification.