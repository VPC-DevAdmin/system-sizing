"""Versioned on-premises cost arithmetic with explicit measured and assumed inputs."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CostLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: int = Field(default=1, ge=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    priced_on: date
    source: str = Field(min_length=1, max_length=2048)
    acquisition: float = Field(ge=0)
    support_per_year: float = Field(ge=0)
    software_per_year: float = Field(ge=0)
    operations_per_year: float = Field(ge=0)
    amortization_years: float = Field(gt=0, le=20)
    electricity_per_kwh: float = Field(ge=0)
    cooling_multiplier: float = Field(ge=1, le=5)
    allocation_mode: Literal["whole_host", "allocated_share"] = "whole_host"
    allocated_share: float = Field(default=1, gt=0, le=1)
    setup_cost: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def whole_host_share(self):
        if self.allocation_mode == "whole_host" and self.allocated_share != 1:
            raise ValueError("Whole-host accounting must include the full host cost")
        return self

    def fixed_per_hour(self):
        annual = self.acquisition / self.amortization_years + self.support_per_year
        annual += self.software_per_year + self.operations_per_year
        return annual * self.allocated_share / (365.25 * 24)


class AccountedWork(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    campaign_id: str = Field(min_length=1)
    contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    demand_trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    accounting_duration_s: float = Field(gt=0)
    accepted_tasks: int = Field(gt=0)
    wall_energy_joules: float = Field(ge=0)
    energy_boundary: Literal["server_psu_input", "pdu_outlet"]
    includes_idle_and_reserve: Literal[True]
    quality_and_slo_passed: Literal[True]


def cost(work: AccountedWork, ledger: CostLedger):
    fixed = ledger.fixed_per_hour() * work.accounting_duration_s / 3600
    electricity = work.wall_energy_joules / 3600000 * ledger.electricity_per_kwh
    facility = electricity * ledger.cooling_multiplier
    total = fixed + facility
    return {
        "currency": ledger.currency,
        "fixed_cost": fixed,
        "electricity_and_cooling_cost": facility,
        "total_cost": total,
        "cost_per_accepted_task": total / work.accepted_tasks,
        "setup_cost_excluded_from_recurring_cost": ledger.setup_cost,
        "classification": "Calculated scenario using measured work and supplied prices",
        "capital_savings_claim": False,
    }


def compare_cost(a, b, ledger_a, ledger_b):
    if a.contract_sha256 != b.contract_sha256 or a.demand_trace_sha256 != b.demand_trace_sha256:
        raise ValueError("Cost comparison requires identical demand and acceptance contracts")
    if a.accepted_tasks != b.accepted_tasks:
        raise ValueError("Fixed-work comparison requires equal accepted task counts")
    if ledger_a.currency != ledger_b.currency:
        raise ValueError("Provide prices in a single currency")
    if ledger_a.allocation_mode != ledger_b.allocation_mode:
        raise ValueError("Accounting boundaries must match")
    baseline, treatment = cost(a, ledger_a), cost(b, ledger_b)
    av, bv = baseline["cost_per_accepted_task"], treatment["cost_per_accepted_task"]
    savings = av - bv
    additional_setup = max(0, ledger_b.setup_cost - ledger_a.setup_cost)
    return {
        "baseline": baseline,
        "treatment": treatment,
        "reduction_percent": 100 * savings / av if av else None,
        "break_even_tasks": additional_setup / savings if savings > 0 else None,
        "assumptions_a": ledger_a.model_dump(mode="json"),
        "assumptions_b": ledger_b.model_dump(mode="json"),
        "measured_a": a.model_dump(),
        "measured_b": b.model_dump(),
    }
