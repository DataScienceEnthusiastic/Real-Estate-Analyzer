"""Advanced decision analytics.

Adds the analytical depth a senior investor expects on top of the core
underwriting engine:

* **Break-even appreciation** — the appreciation rate at which the deal merely
  ties the risk-free SBI FD (and separately the Nifty 50 TRI).
* **Sensitivity grid** — equity IRR across a range of appreciation × rent
  escalation assumptions (a heat-map / contour).
* **Tornado** — how much each single driver moves the outcome, ranked.
* **Probability-weighted expected value** — a single ``expected`` answer from
  the Best / Base / Worst scenarios with user-defined likelihoods.
* **Return decomposition** — how much of the return is rent vs capital gain.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from src.finance_engine import run_scenario_underwriting
from src.models import (
    ExpectedValueResult,
    FinancingInput,
    InvestorProfile,
    PropertyInput,
    UnderwritingResult,
)

ScenarioParams = dict


def _base_scenario(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
) -> ScenarioParams:
    """Canonical Base-case scenario parameters (Benchmark appreciation, avg vacancy)."""
    return {
        "property_input": property_input,
        "financing_input": financing_input,
        "investor": investor,
        "macro": macro,
        "scenario_appreciation_rate": property_input.appreciation_rate,
        "scenario_vacancy_months": property_input.vacancy_months_avg,
        "scenario_loan_rate": financing_input.home_loan_interest_rate_pct,
        "scenario_exit_discount_pct": 0.0,
        "scenario_uc_delay_months": int(round(property_input.avg_uc_delay_months)),
    }


def _run(scenario: ScenarioParams) -> UnderwritingResult:
    return run_scenario_underwriting(**scenario)


def _irr_from(res: UnderwritingResult) -> float:
    return res.equity_irr_pct if res.equity_irr_pct is not None else float("nan")


def compute_break_even_appreciation(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
) -> dict[str, Optional[float]]:
    """
    Returns the appreciation rate (%) at which the deal's equity IRR equals the
    post-tax SBI FD return and the post-tax Nifty 50 TRI IRR. ``None`` if the crossing
    is outside the searched band.
    """
    base_res = _run(_base_scenario(property_input, financing_input, investor, macro))
    fd_posttax_pct = base_res.fd_benchmark.annual_return_pct
    nifty_pct = (
        base_res.nifty_benchmark.post_tax_irr_pct
        if base_res.nifty_benchmark.post_tax_irr_pct is not None
        else financing_input.nifty_benchmark_cagr_pct * 100.0
    )

    def irr_at(appreciation: float) -> float:
        scenario = _base_scenario(property_input, financing_input, investor, macro)
        scenario["scenario_appreciation_rate"] = float(appreciation)
        return _irr_from(_run(scenario))

    def find_break_even(hurdle_pct: float) -> Optional[float]:
        def f(a: float) -> float:
            val = irr_at(a)
            if np.isnan(val) or np.isinf(val):
                return -100.0 - hurdle_pct
            return val - hurdle_pct

        lo, hi = -0.05, 0.35
        f_lo = f(lo)
        f_hi = f(hi)
        if f_lo * f_hi >= 0:
            # If still same sign, try starting from 0.0
            lo = 0.0
            f_lo = f(lo)
            if f_lo * f_hi >= 0:
                return None
        try:
            root = brentq(f, lo, hi, xtol=1e-5, maxiter=200)
            return float(root * 100.0)
        except Exception:
            return None

    return {
        "fd_post_tax_pct": fd_posttax_pct,
        "nifty_pct": nifty_pct,
        "break_even_vs_fd_pct": find_break_even(fd_posttax_pct),
        "break_even_vs_nifty_pct": find_break_even(nifty_pct),
    }


def compute_sensitivity_grid(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
    appr_min: float = 0.0,
    appr_max: float = 0.14,
    appr_steps: int = 7,
    esc_min: float = 0.0,
    esc_max: float = 0.10,
    esc_steps: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (appreciation_axis_pct, escalation_axis_pct, irr_matrix) where
    ``irr_matrix[i, j]`` is the equity IRR for appreciation[i] × escalation[j].
    """
    appr_axis = np.linspace(appr_min, appr_max, appr_steps)
    esc_axis = np.linspace(esc_min, esc_max, esc_steps)
    irr_grid = np.full((appr_steps, esc_steps), np.nan)

    for i, appr in enumerate(appr_axis):
        for j, esc in enumerate(esc_axis):
            scenario = _base_scenario(property_input, financing_input, investor, macro)
            scenario["scenario_appreciation_rate"] = float(appr)
            scenario["financing_input"] = replace(
                financing_input, annual_rent_escalation_pct=float(esc)
            )
            irr_grid[i, j] = _irr_from(_run(scenario))

    return appr_axis * 100.0, esc_axis * 100.0, irr_grid


def compute_tornado_data(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
) -> list[dict[str, float]]:
    """
    One-at-a-time swing of the five main value drivers around the Base case.
    Returns a list of ``{"label": str, "low": float, "high": float}`` where
    ``low``/``high`` are the equity IRR (%) under the pessimistic/optimistic
    setting, sorted by impact.
    """
    base = _base_scenario(property_input, financing_input, investor, macro)
    base_irr = _irr_from(_run(base))

    def run(appr=None, escalation=None, rate=None, vacancy=None, discount=None) -> float:
        scenario = dict(base)
        if appr is not None:
            scenario["scenario_appreciation_rate"] = appr
        if rate is not None:
            scenario["scenario_loan_rate"] = rate
        if vacancy is not None:
            scenario["scenario_vacancy_months"] = vacancy
        if discount is not None:
            scenario["scenario_exit_discount_pct"] = discount
        if escalation is not None:
            scenario["financing_input"] = replace(
                financing_input, annual_rent_escalation_pct=escalation
            )
        return _irr_from(_run(scenario))

    items = [
        {
            "label": "Appreciation (±2 pp)",
            "low": run(appr=property_input.appreciation_rate - 0.02),
            "high": run(appr=property_input.appreciation_rate + 0.02),
        },
        {
            "label": "Rent escalation (±1 pp)",
            "low": run(escalation=max(0.0, financing_input.annual_rent_escalation_pct - 0.01)),
            "high": run(escalation=financing_input.annual_rent_escalation_pct + 0.01),
        },
        {
            "label": "Loan rate (±100 bps)",
            "low": run(rate=financing_input.home_loan_interest_rate_pct + 0.01),
            "high": run(rate=max(0.05, financing_input.home_loan_interest_rate_pct - 0.01)),
        },
        {
            "label": "Vacancy (best ↔ worst)",
            "low": run(vacancy=property_input.vacancy_months_worst),
            "high": run(vacancy=property_input.vacancy_months_best),
        },
        {
            "label": "Exit discount (0 ↔ stress)",
            "low": run(discount=property_input.exit_discount_stress_pct),
            "high": run(discount=0.0),
        },
    ]

    # Only keep drivers that actually move the needle; sort by spread.
    for it in items:
        it["spread"] = abs(it["high"] - it["low"])
    items = [it for it in items if it["spread"] > 1e-6]
    items.sort(key=lambda it: it["spread"], reverse=True)

    # Annotate the base for the chart reference line
    for it in items:
        it["base"] = base_irr
    return items


def compute_probability_weighted(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
    p_best: float = 0.25,
    p_base: float = 0.50,
    p_worst: float = 0.25,
) -> ExpectedValueResult:
    """Folds Best / Base / Worst into a single probability-weighted expected view."""
    total_p = p_best + p_base + p_worst
    p_best, p_base, p_worst = p_best / total_p, p_base / total_p, p_worst / total_p

    scenario = _base_scenario(property_input, financing_input, investor, macro)
    best = _run({**scenario, "scenario_appreciation_rate": property_input.appreciation_rate + 0.015,
                 "scenario_vacancy_months": property_input.vacancy_months_best,
                 "scenario_loan_rate": max(0.065, financing_input.home_loan_interest_rate_pct - 0.005),
                 "scenario_exit_discount_pct": 0.0})
    base = _run(scenario)
    worst = _run({**scenario, "scenario_appreciation_rate": max(0.02, property_input.appreciation_rate - 0.025),
                  "scenario_vacancy_months": property_input.vacancy_months_worst,
                  "scenario_loan_rate": financing_input.home_loan_interest_rate_pct + 0.02,
                  "scenario_exit_discount_pct": property_input.exit_discount_stress_pct})

    scenarios = [(p_best, best), (p_base, base), (p_worst, worst)]

    exp_terminal = sum(p * r.exit.net_exit_proceeds for p, r in scenarios)
    exp_irr = sum(p * (r.equity_irr_pct if r.equity_irr_pct is not None else 0.0) for p, r in scenarios)
    exp_net_exit = sum(p * r.exit.net_exit_proceeds for p, r in scenarios)

    upfront = base.loan.required_upfront_cash
    def invested(r: UnderwritingResult) -> float:
        return upfront + sum(max(0.0, -cf.net_cash_flow) for cf in r.annual_cash_flows)
    exp_invested = sum(p * invested(r) for p, r in scenarios)

    fd_pct = base.fd_benchmark.annual_return_pct
    nifty_pct = (
        base.nifty_benchmark.post_tax_irr_pct
        if base.nifty_benchmark.post_tax_irr_pct is not None
        else financing_input.nifty_benchmark_cagr_pct * 100.0
    )

    def p_beat(hurdle_pct: float) -> float:
        return sum(p for p, r in scenarios if (r.equity_irr_pct or -1e9) > hurdle_pct)

    return ExpectedValueResult(
        p_best=p_best,
        p_base=p_base,
        p_worst=p_worst,
        expected_terminal_wealth=exp_terminal,
        expected_equity_irr_pct=exp_irr,
        expected_net_exit_proceeds=exp_net_exit,
        expected_total_cash_invested=exp_invested,
        best_irr_pct=best.equity_irr_pct,
        base_irr_pct=base.equity_irr_pct,
        worst_irr_pct=worst.equity_irr_pct,
        best_terminal_wealth=best.exit.net_exit_proceeds,
        base_terminal_wealth=base.exit.net_exit_proceeds,
        worst_terminal_wealth=worst.exit.net_exit_proceeds,
        p_beat_fd=p_beat(fd_pct),
        p_beat_nifty=p_beat(nifty_pct),
        fd_post_tax_pct=fd_pct,
        nifty_pct=nifty_pct,
    )


def compute_return_decomposition(base_res: UnderwritingResult) -> dict[str, float]:
    """
    Computes the exact 5-row Net Wealth Bridge reconciling to ₹0.00 discrepancy:
    net_wealth_created = net_post_tax_capital_gain + cumulative_noi - cumulative_loan_interest - cumulative_operating_tax
    reconciles identically to:
    net_exit_proceeds - required_upfront_cash + sum(annual_net_cfs)
    """
    upfront = base_res.loan.required_upfront_cash
    total_capital_invested = upfront + sum(
        abs(cf.net_cash_flow) for cf in base_res.annual_cash_flows if cf.net_cash_flow < 0
    )
    gross_price_appreciation = base_res.exit.terminal_gross_value - base_res.acquisition.agreement_value_inr
    total_friction_and_cg_tax = (
        base_res.acquisition.stamp_duty_inr
        + base_res.acquisition.registration_fee_inr
        + base_res.acquisition.gst_inr
        + base_res.acquisition.buy_brokerage_inr
        + base_res.exit.sell_brokerage_inr
        + base_res.exit.cg_tax_inr
        + base_res.acquisition.interiors_and_misc_inr
        + getattr(base_res.loan, "loan_origination_and_motd_inr", 0.0)
    )
    net_post_tax_capital_gain = gross_price_appreciation - total_friction_and_cg_tax
    cumulative_noi = sum(cf.noi for cf in base_res.annual_cash_flows)
    cumulative_loan_interest = sum(cf.interest_paid for cf in base_res.annual_cash_flows)
    cumulative_operating_tax = sum(cf.tax_impact for cf in base_res.annual_cash_flows)
    cumulative_interest_and_tax = cumulative_loan_interest + cumulative_operating_tax

    net_wealth_created = (
        net_post_tax_capital_gain
        + cumulative_noi
        - cumulative_loan_interest
        - cumulative_operating_tax
    )
    actual_net_wealth = (
        base_res.exit.net_exit_proceeds
        - upfront
        + sum(cf.net_cash_flow for cf in base_res.annual_cash_flows)
    )
    assert abs(net_wealth_created - actual_net_wealth) < 1.0, (
        f"Return decomposition discrepancy: {net_wealth_created} vs {actual_net_wealth}"
    )

    h = max(1, len(base_res.annual_cash_flows))
    appreciation_cagr = (
        (base_res.exit.terminal_gross_value / base_res.acquisition.agreement_value_inr)
        ** (1.0 / h)
        - 1.0
    ) if base_res.acquisition.agreement_value_inr > 0 else 0.0

    return {
        "total_capital_invested": total_capital_invested,
        "total_equity_invested": total_capital_invested,
        "gross_price_appreciation": gross_price_appreciation,
        "total_friction_and_cg_tax": total_friction_and_cg_tax,
        "net_post_tax_capital_gain": net_post_tax_capital_gain,
        "capital_gain": net_post_tax_capital_gain,
        "cumulative_noi": cumulative_noi,
        "total_rent_collected": cumulative_noi,
        "cumulative_loan_interest": cumulative_loan_interest,
        "cumulative_operating_tax": cumulative_operating_tax,
        "cumulative_interest_and_tax": cumulative_interest_and_tax,
        "net_wealth_created": net_wealth_created,
        "gross_rental_yield_pct": base_res.year_1_gross_yield_pct,
        "net_rental_yield_pct": base_res.year_1_net_yield_pct,
        "appreciation_cagr_pct": appreciation_cagr,
    }
