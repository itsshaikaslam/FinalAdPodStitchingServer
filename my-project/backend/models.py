from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Costs(BaseModel):
    total_setup_cost: float = Field(ge=0)
    total_delivery_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)


class OptimizationResponse(BaseModel):
    activated_servers: List[str]
    assignments: Dict[str, str]
    costs: Costs


