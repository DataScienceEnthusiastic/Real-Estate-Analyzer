"""Smart Pune Micro-Market Recommender Engine.

Evaluates all 43 Pune localities against the user's exact financial profile,
acquisition preferences, and mortgage assumptions using the core underwriting
and 4-gate decision engines, returning the Top N ranked pincodes and areas.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional
import pandas as pd

from src.advanced_analysis import compute_return_decomposition
from src.charts import format_inr
from src.data_loader import get_property_market_defaults
from src.decision_gates import evaluate_master_verdict
from src.finance_engine import run_full_underwriting_matrix
from src.models import (
    FinancingInput,
    InvestorProfile,
    LocalityRecommendation,
)


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def rank_top_localities(
    bhk: str,
    use_case: str,
    property_status: str,
    buyer_ownership: str,
    brokerage_enabled: bool,
    sell_brokerage_enabled: bool,
    expected_possession_months: int,
    financing_template: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
    loc_df: pd.DataFrame,
    top_n: int = 3,
    include_all: bool = False,
) -> list[LocalityRecommendation]:
    """
    Runs full multi-scenario underwriting and 4-gate audits across all Pune
    localities for the given configuration and investor profile, ranking them
    by financial alignment, profitability, downside safety, and market liquidity.
    """
    candidates: list[LocalityRecommendation] = []
    unique_locs = loc_df[["pincode", "locality_name"]].drop_duplicates()

    for _, row in unique_locs.iterrows():
        pincode = str(row["pincode"]).strip().zfill(6)
        loc_name = str(row["locality_name"]).strip()

        try:
            prop = get_property_market_defaults(pincode, loc_name, bhk)
        except ValueError:
            continue

        # Skip configurations with zero rental comps when evaluating pure rental investments
        if use_case == "investment_rental" and prop.monthly_rent_inr <= 0.0:
            continue

        prop = replace(
            prop,
            use_case=use_case,
            property_status=property_status,
            buyer_ownership=buyer_ownership,
            brokerage_enabled=brokerage_enabled,
            sell_brokerage_enabled=sell_brokerage_enabled,
            expected_possession_months=expected_possession_months,
        )

        # Calibrate per-unit society maintenance and Ready Reckoner municipal tax
        loc_maint = float(round(prop.expected_sba_sqft * 4.0))
        rr_val = prop.carpet_area_sqft * prop.rr_rate_primary_sqft
        loc_ptax = float(round((rr_val if rr_val > 0 else prop.agreement_value_inr) * 0.0025))

        fin = replace(
            financing_template,
            monthly_society_maintenance_inr=loc_maint,
            annual_property_tax_inr=loc_ptax,
        )

        matrix = run_full_underwriting_matrix(prop, fin, investor, macro)
        verdict = evaluate_master_verdict(investor, prop, fin, matrix)
        base = matrix.base
        decomp = compute_return_decomposition(base)

        g1, g2, g3, g4 = verdict.gates
        upfront = base.loan.required_upfront_cash
        post_cash = investor.total_liquid_savings_inr - upfront
        foir_pct = float(g1.metrics.get("foir_pct", 100.0))
        buffer_months = float(g1.metrics.get("buffer_months_actual", 0.0))
        market_to_rr = float(g3.metrics.get("market_to_rr_ratio", 1.0))
        monthly_deficit = float(g3.metrics.get("year_1_monthly_deficit", 0.0))
        monthly_surplus = float(g3.metrics.get("monthly_disposable_surplus", 0.0))
        stressed_foir = float(g2.metrics.get("stressed_foir_200bps", 100.0))
        yr3_recovery = float(g2.metrics.get("yr3_capital_recovery_pct", 0.0))

        # ------------------------------------------------------------------
        # 1. Solvency & Gate Alignment Score (40 pts max)
        # ------------------------------------------------------------------
        verdict_pts = 32.0 if verdict.status == "GO" else (18.0 if verdict.status == "CAUTION" else 0.0)
        g1_pts = 8.0 if g1.status == "PASS" else (4.0 if g1.status == "WARN" else -25.0)
        solvency_score = verdict_pts + g1_pts

        # ------------------------------------------------------------------
        # 2. Financial Profit & Capital Efficiency Score (35 pts max)
        # ------------------------------------------------------------------
        fd_rate = base.fd_benchmark.annual_return_pct
        base_irr = base.equity_irr_pct
        irr_alpha = (base_irr - fd_rate) if base_irr is not None else -10.0
        irr_pts = _clamp((irr_alpha + 2.0) / 9.0, 0.0, 1.0) * 18.0

        npv_ratio = (base.npv_inr / upfront) if upfront > 0 else -0.5
        npv_pts = _clamp((npv_ratio + 0.20) / 0.80, 0.0, 1.0) * 10.0

        if use_case == "investment_rental":
            yield_pts = _clamp((base.year_1_net_yield_pct - 1.2) / 1.8, 0.0, 1.0) * 4.0
        else:
            yield_pts = _clamp((base.year_1_gross_yield_pct - 1.5) / 2.0, 0.0, 1.0) * 4.0

        deficit_ratio = (monthly_deficit / max(1.0, monthly_surplus)) if monthly_deficit > 0 else 0.0
        cf_fit_pts = _clamp(1.0 - deficit_ratio, 0.0, 1.0) * 3.0
        profit_score = irr_pts + npv_pts + yield_pts + cf_fit_pts

        # ------------------------------------------------------------------
        # 3. Market Research, Liquidity & Risk Fit Score (25 pts max)
        # ------------------------------------------------------------------
        target_buf = max(1.0, investor.emergency_buffer_months_target)
        buf_pts = _clamp(buffer_months / target_buf, 0.0, 1.0) * 7.0

        foir_stress_pts = 4.0 if stressed_foir <= 50.0 else (2.0 if stressed_foir <= 60.0 else 0.0)
        rec_pts = _clamp((yr3_recovery - 55.0) / 30.0, 0.0, 1.0) * 4.0

        tier_lower = prop.liquidity_tier.lower()
        liq_pts = 5.0 if tier_lower == "high" else (3.0 if tier_lower == "medium" else 1.0)
        rr_pts = 5.0 if market_to_rr <= 2.0 else (3.0 if market_to_rr <= 2.5 else 0.0)
        cantonment_adj = -2.0 if prop.jurisdiction.lower() == "cantonment" else 0.0

        market_score = buf_pts + foir_stress_pts + rec_pts + liq_pts + rr_pts + cantonment_adj

        total_score = _clamp(solvency_score + profit_score + market_score, 0.0, 100.0)
        # Enforce hard cap on unaffordable deals so they never outrank affordable ones
        if post_cash < 0 or foir_pct > 50.0:
            total_score = min(total_score, 28.0)

        # Personalized explanation bullets
        y1_monthly_cf = base.annual_cash_flows[0].net_cash_flow / 12.0
        irr_str = f"{base_irr:.2f}%" if base_irr is not None else "Negative"
        real_str = f"{base.real_equity_irr_pct:.2f}%" if base.real_equity_irr_pct is not None else "N/A"
        fd_spread = base.fd_benchmark.irr_spread_pct

        if post_cash >= 0:
            fit_reason = (
                f"Budget Fit: Upfront cash {format_inr(upfront)} leaves {format_inr(post_cash)} liquid buffer "
                f"({buffer_months:.1f} mo reserves) at a {foir_pct:.1f}% FOIR."
            )
        else:
            fit_reason = (
                f"Capital Stretch: Requires {format_inr(upfront)} upfront (shortfall of {format_inr(abs(post_cash))}) "
                f"with {foir_pct:.1f}% FOIR."
            )

        profit_reason = (
            f"Projected Return: {irr_str} Base IRR ({real_str} real after CPI), "
            f"{format_inr(decomp['net_wealth_created'])} net wealth created, and {fd_spread:+.2f} pp vs post-tax FD."
        )

        dscr_txt = f", DSCR {base.year_1_dscr:.2f}x" if base.year_1_dscr is not None else ""
        market_reason = (
            f"Market Fundamentals: {prop.appreciation_rate*100:.2f}% historical CAGR, "
            f"{base.year_1_gross_yield_pct:.2f}% gross yield{dscr_txt}, {market_to_rr:.2f}x Ready Reckoner ratio, "
            f"and {prop.liquidity_tier.upper()} zone liquidity ({prop.quarters_to_sell:.1f} QTS)."
        )

        candidates.append(
            LocalityRecommendation(
                rank=0,
                pincode=pincode,
                locality_name=loc_name,
                pune_zone=prop.pune_zone,
                jurisdiction=prop.jurisdiction,
                alignment_score=round(total_score, 1),
                verdict_status=verdict.status,
                gate_statuses=[g.status for g in verdict.gates],
                ticket_price_inr=prop.agreement_value_inr,
                all_in_cost_inr=base.acquisition.all_in_acquisition_cost,
                upfront_cash_inr=upfront,
                monthly_emi_inr=base.loan.monthly_emi,
                foir_pct=foir_pct,
                base_irr_pct=base_irr,
                real_irr_pct=base.real_equity_irr_pct,
                net_wealth_created_inr=decomp["net_wealth_created"],
                npv_inr=base.npv_inr,
                net_yield_pct=base.year_1_net_yield_pct,
                dscr=base.year_1_dscr,
                market_to_rr_ratio=market_to_rr,
                liquidity_tier=prop.liquidity_tier,
                quarters_to_sell=prop.quarters_to_sell,
                appreciation_rate_pct=prop.appreciation_rate * 100.0,
                year_1_monthly_cf_inr=y1_monthly_cf,
                alignment_reasons=[fit_reason, profit_reason, market_reason],
            )
        )

    candidates.sort(
        key=lambda c: (
            c.alignment_score,
            c.base_irr_pct if c.base_irr_pct is not None else -999.0,
            c.net_wealth_created_inr,
        ),
        reverse=True,
    )

    for idx, item in enumerate(candidates, start=1):
        item.rank = idx

    return candidates if include_all else candidates[:top_n]
