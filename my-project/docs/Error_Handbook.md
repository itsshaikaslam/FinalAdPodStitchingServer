Error_Handbook.md - Fault Tolerance System
Overview
This handbook details the fault tolerance system for the Ad-Pod Stitching Server Optimization application. It covers error types, handling strategies, and logging to ensure robust, error-free execution.
Error Categories & Handling
1. Input Validation Errors

Types: Missing columns, duplicate IDs, non-numeric values, negative demands/capacities.
Handling: In utils.py, use Pandas validation; raise ValueError with specific messages.
API Response: HTTP 400 Bad Request, JSON {"error": "Detailed message"}.

2. Infeasibility Errors

Types: Total capacity < total demand, no solution in MILP.
Handling: Check pre-optimization; if infeasible, raise custom InfeasibleError.
Response: HTTP 400, {"error": "Insufficient server capacity"}.

3. Runtime Errors

Types: Memory overflow (large distances), solver timeout.
Handling: Use try-except; for memory, use float32/on-the-fly calc; set solver timeout=25s, fallback to partial solution.
Response: HTTP 500 if critical, else best-effort results with warning.

4. Edge Cases

Zero-demand DMAs: Assign to nearest activated server or log and ignore in demand calc.
Single server/DMA: Bypass clustering if K=1.
Empty files: Validate early, raise error.

Logging & Diagnostics

Use Python logging: Config in app.py (level=INFO, file=app.log).
Log levels: DEBUG for computations, INFO for progress, ERROR for failures.
Tracebacks: Capture in exceptions, log full stack.

Recovery Strategies

Graceful Degradation: If one K fails, skip and continue loop.
Retries: For solver timeouts, retry once with increased time.
User Feedback: In frontend, display errors via st.error.

This system ensures the application fails safely and informatively.