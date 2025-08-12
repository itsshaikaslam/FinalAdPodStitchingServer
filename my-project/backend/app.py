from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

#mynew
import io
import logging
import structlog

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import OptimizationResponse
from .optimization import run_cflp_heuristic, run_greedy_heuristic
from .utils import (
    InfeasibleError,
    validate_dmas_dataframe,
    validate_servers_dataframe,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("adpod.backend")

app = FastAPI(title="Ad-Pod Stitching Server Optimizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_endpoint(
    servers: UploadFile = File(...),
    dmas: UploadFile = File(...),
):
    """Optimize server activation and DMA assignment.

    Accepts two CSV files and returns the optimization result as JSON.
    """
    try:
        servers_bytes = await servers.read()
        dmas_bytes = await dmas.read()

        try:
            servers_df = pd.read_csv(io.BytesIO(servers_bytes))
            dmas_df = pd.read_csv(io.BytesIO(dmas_bytes))
        except pd.errors.EmptyDataError as exc:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty") from exc

        validate_servers_dataframe(servers_df)
        validate_dmas_dataframe(dmas_df)

        logger.info("optimize_request", servers_rows=len(servers_df), dmas_rows=len(dmas_df))
        result = run_cflp_heuristic(servers_df=servers_df, dmas_df=dmas_df)

        logger.info(
            "optimize_response",
            num_activated=len(result["activated_servers"]),
            total_cost=result["costs"]["total_cost"],
        )
        return OptimizationResponse(**result)

    except InfeasibleError as exc:  # input infeasible
        logger.warning("Infeasible instance: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:  # validation errors
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # unexpected
        logger.exception("Unexpected error during optimization")
        raise HTTPException(status_code=500, detail="Unexpected error") from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize_fast", response_model=OptimizationResponse)
async def optimize_fast_endpoint(
    servers: UploadFile = File(...),
    dmas: UploadFile = File(...),
):
    try:
        servers_bytes = await servers.read()
        dmas_bytes = await dmas.read()

        try:
            servers_df = pd.read_csv(io.BytesIO(servers_bytes))
            dmas_df = pd.read_csv(io.BytesIO(dmas_bytes))
        except pd.errors.EmptyDataError as exc:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty") from exc

        validate_servers_dataframe(servers_df)
        validate_dmas_dataframe(dmas_df)

        logger.info("optimize_fast_request", servers_rows=len(servers_df), dmas_rows=len(dmas_df))
        result = run_greedy_heuristic(servers_df=servers_df, dmas_df=dmas_df)
        logger.info(
            "optimize_fast_response",
            num_activated=len(result["activated_servers"]),
            total_cost=result["costs"]["total_cost"],
        )
        return OptimizationResponse(**result)
    except InfeasibleError as exc:
        logger.warning("Infeasible instance: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # unexpected
        logger.exception("Unexpected error during fast optimization")
        raise HTTPException(status_code=500, detail="Unexpected error") from exc


