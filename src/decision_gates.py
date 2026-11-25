from __future__ import annotations

from typing import Any
from src.models import (
    GateResult,
    InvestorProfile,
    MasterVerdict,
    PropertyInput,
    ScenarioMatrixResult,
)


def evaluate_gate_1_affordability(
    investor: InvestorProfile,
    base_emi: float,
    required_upfront_cash: float,
) -> GateResult:
    """
    Gate 1 — Affordability & Liquidity Safety
    Ensures the investor has sufficient upfront capital and debt servicing capacity.
    """
    post_purchase_cash = investor.total_liquid_savings_inr - required_upfront_cash
    monthly_obligations = investor.monthly_living_expenses_inr + investor.existing_monthly_emis_inr + base_emi
    min_required_reserve = (investor.emergency_buffer_months_target * monthly_obligations) + investor.upcoming_cash_need_inr
    
    foir = (investor.existing_monthly_emis_inr + base_emi) / investor.net_monthly_income_inr if investor.net_monthly_income_inr > 0 else 1.0
    upfront_to_savings_pct = (required_upfront_cash / investor.total_liquid_savings_inr * 100.0) if investor.total_liquid_savings_inr > 0 else 100.0

    metrics = {
        "required_upfront_cash": required_upfront_cash,
        "post_purchase_liquid_cash": post_purchase_cash,
        "monthly_obligations": monthly_obligations,
        "min_required_reserve": min_required_reserve,
        "foir_pct": foir * 100.0,
        "buffer_months_actual": (post_purchase_cash / monthly_obligations) if monthly_obligations > 0 else 0.0,
        "upfront_to_savings_pct": upfront_to_savings_pct,
    }

    reasons: list[str] = []
    status = "PASS"

    # Strict FAIL conditions
    if post_purchase_cash < 0:
        status = "FAIL"
        reasons.append(f"Insufficient funds: Down payment + closing costs (INR {int(round(required_upfront_cash)):,}) exceed liquid savings (INR {int(round(investor.total_liquid_savings_inr)):,}).")
    elif post_purchase_cash < (3.0 * monthly_obligations):
        status = "FAIL"
        reasons.append(f"Critical liquidity depletion: Remaining liquid cash covers only {metrics['buffer_months_actual']:.1f} months (< 3 months minimum safety floor).")
    elif foir > 0.50:
        status = "FAIL"
        reasons.append(f"Dangerous leverage: Fixed Obligation to Income Ratio (FOIR) is {foir * 100:.1f}%, exceeding the 50.0% underwriting ceiling.")

    # Warning conditions
    if status != "FAIL":
        if post_purchase_cash < min_required_reserve:
            status = "WARN"
            shortfall = min_required_reserve - post_purchase_cash
            reasons.append(f"Reserve shortfall: Remaining cash (INR {int(round(post_purchase_cash)):,}) is below recommended {investor.emergency_buffer_months_target:.0f}-month reserve + upcoming cash need (INR {int(round(min_required_reserve)):,}; shortfall of INR {int(round(shortfall)):,}).")
        if 0.40 < foir <= 0.50:
            status = "WARN"
            reasons.append(f"Elevated debt ratio: FOIR is {foir * 100:.1f}% (above optimal 40% prudence mark).")

    if not reasons:
        reasons.append(f"Pass: FOIR is healthy at {foir * 100:.1f}% and liquid reserves cover {metrics['buffer_months_actual']:.1f} months of obligations.")

    if status != "FAIL" and upfront_to_savings_pct > 85.0:
        reasons.append(f"Liquidity Concentration Note: Upfront equity commits {upfront_to_savings_pct:.1f}% of your liquid savings into a single illiquid real estate asset.")

    return GateResult(
        gate_number=1,
        name="Affordability & Liquidity Safety",
        status=status,
        metrics=metrics,
        reasons=reasons,
    )


def evaluate_gate_2_downside_survival(
    investor: InvestorProfile,
    property_input: PropertyInput,
    holding_period_years: int,
    matrix: ScenarioMatrixResult,
) -> GateResult:
    """
    Gate 2 — Downside Survival & Liquidity Risk
    Tests resilience against a +200 bps mortgage rate shock, extended vacancy, micro-market overhang,
    and Year-3 forced distress liquidation.
    """
    stressed_rate_metrics = matrix.stress_interest_rate_shock.metrics
    stressed_foir_200 = stressed_rate_metrics.get("stressed_foir_200bps", 0.0) / 100.0

    vac_metrics = matrix.stress_vacancy_shock.metrics
    remaining_vac_buffer_months = vac_metrics.get("remaining_buffer_months", 0.0)

    # Forced exit at Year 3 recovery ratio
    yr3_exit_metrics = matrix.stress_forced_exit_year_3.metrics
    raw_yr3_recovery = yr3_exit_metrics.get("capital_recovery_pct", 0.0)
    yr3_capital_recovery_ratio = raw_yr3_recovery / 100.0 if raw_yr3_recovery > 1.0 else raw_yr3_recovery

    is_low_liquidity = (property_input.liquidity_tier.lower() == "low") or (property_input.quarters_to_sell > 6.0)
    has_short_horizon = holding_period_years < 5
    has_cash_need = investor.upcoming_cash_need_inr > 0

    metrics = {
        "stressed_foir_200bps": stressed_foir_200 * 100.0,
        "remaining_buffer_months_after_vacancy": remaining_vac_buffer_months,
        "liquidity_tier": property_input.liquidity_tier,
        "quarters_to_sell": property_input.quarters_to_sell,
        "holding_period_years": holding_period_years,
        "yr3_capital_recovery_pct": yr3_capital_recovery_ratio * 100.0,
    }

    reasons: list[str] = []
    status = "PASS"

    # Strict FAIL conditions:
    # 1. stressed_foir_200bps > 0.60
    # 2. vacancy_shock_buffer_months < 1.0
    # 3. yr3_capital_recovery_ratio < 0.65 AND (holding_period_years < 5 OR upcoming_cash_need_inr > 0)
    if stressed_foir_200 > 0.60:
        status = "FAIL"
        reasons.append(f"Insolvent debt shock: A +200 bps interest rate hike drives FOIR to {stressed_foir_200 * 100:.1f}%, exceeding the 60.0% insolvency ceiling.")
    elif remaining_vac_buffer_months < 1.0:
        status = "FAIL"
        reasons.append(f"Severe liquidity exhaustion: Extended 6-month vacancy shock depletes cash buffer below 1 month ({remaining_vac_buffer_months:.1f} months remaining).")
    elif yr3_capital_recovery_ratio < 0.65 and (has_short_horizon or has_cash_need):
        status = "FAIL"
        reasons.append(f"Severe capital impairment: Forced distress exit at Year 3 recovers only {yr3_capital_recovery_ratio * 100:.1f}% of invested equity (< 65% floor) combined with short {holding_period_years}y horizon or upcoming cash need.")

    # Warning conditions:
    # 1. 0.50 < stressed_foir_200bps <= 0.60
    # 2. vacancy_shock_buffer_months < 3.0
    # 3. liquidity_tier == 'low' (quarters_to_sell > 6.0)
    # 4. yr3_capital_recovery_ratio < 0.75
    if status != "FAIL":
        if 0.50 < stressed_foir_200 <= 0.60:
            status = "WARN"
            reasons.append(f"Tight debt cushion: +200 bps rate shock elevates FOIR to {stressed_foir_200 * 100:.1f}%.")
        if remaining_vac_buffer_months < 3.0:
            status = "WARN"
            reasons.append(f"Vulnerable vacancy buffer: 6-month vacancy reduces reserves to {remaining_vac_buffer_months:.1f} months.")
        if is_low_liquidity:
            status = "WARN"
            reasons.append(f"Overhang risk: Micro-market liquidity tier is '{property_input.liquidity_tier}' ({property_input.quarters_to_sell:.1f} quarters to liquidate, benchmark threshold 6.0 quarters).")
        if yr3_capital_recovery_ratio < 0.75:
            status = "WARN"
            reasons.append(f"Distress liquidation loss: Forced exit at Year 3 recovers {yr3_capital_recovery_ratio * 100:.1f}% of capital (below 75.0% prudence mark).")

    if not reasons:
        reasons.append(f"Pass: Stressed FOIR at +200 bps remains viable ({stressed_foir_200 * 100:.1f}%), vacancy shock leaves {remaining_vac_buffer_months:.1f} months of buffer, and Year-3 capital recovery is healthy at {yr3_capital_recovery_ratio * 100:.1f}%.")

    # Contextual Governance Advisories (Informational)
    if property_input.jurisdiction.lower() == "cantonment":
        reasons.append("Cantonment Title Risk: Verify freehold vs. Old Grant / leasehold tenure and Pune Cantonment Board FSI/redevelopment restrictions.")
    if property_input.quarters_to_sell < 1.0:
        reasons.append("Liquidity Proxy Note: Sub-1.0 QTS reflects low primary developer inventory in a mature micro-market; secondary resale liquidation typically takes 3–6 months.")

    return GateResult(
        gate_number=2,
        name="Downside Survival & Liquidity Risk",
        status=status,
        metrics=metrics,
        reasons=reasons,
    )


def evaluate_gate_3_yield_valuation(
    property_input: PropertyInput,
    investor: InvestorProfile,
    matrix: ScenarioMatrixResult,
) -> GateResult:
    """
    Gate 3 — Cash Flow, Yield & Valuation Sanity
    Validates gross/net yields, micro-market Ready Reckoner premiums, and monthly operating deficits.
    """
    base_res = matrix.base
    y1_cf = base_res.annual_cash_flows[0]
    all_in_cost = base_res.acquisition.all_in_acquisition_cost
    rr_market_val = base_res.acquisition.rr_market_value

    net_rental_yield = base_res.year_1_net_yield_pct
    gross_rental_yield = base_res.year_1_gross_yield_pct

    # Market to Ready Reckoner valuation ratio
    market_to_rr_ratio = (property_input.agreement_value_inr / rr_market_val) if rr_market_val > 0 else 1.0

    monthly_cash_deficit = max(0.0, -y1_cf.net_cash_flow / 12.0)
    monthly_disposable_surplus = max(
        0.0,
        investor.net_monthly_income_inr - (investor.monthly_living_expenses_inr + investor.existing_monthly_emis_inr + base_res.loan.monthly_emi)
    )

    metrics = {
        "gross_rental_yield_pct": gross_rental_yield,
        "net_rental_yield_pct": net_rental_yield,
        "market_to_rr_ratio": market_to_rr_ratio,
        "year_1_monthly_deficit": monthly_cash_deficit,
        "monthly_disposable_surplus": monthly_disposable_surplus,
        "year_1_dscr": base_res.year_1_dscr,
    }

    reasons: list[str] = []
    status = "PASS"

    # Severe cash-flow deficit exceeding disposable capacity
    if monthly_cash_deficit > (monthly_disposable_surplus + 0.10 * investor.net_monthly_income_inr):
        status = "FAIL"
        reasons.append(f"Unfunded deficit: Year-1 monthly operating cash drain (INR {int(round(monthly_cash_deficit)):,}) outstrips monthly disposable cash flow (INR {int(round(monthly_disposable_surplus)):,}).")

    if status != "FAIL":
        if market_to_rr_ratio > 2.50:
            status = "WARN"
            reasons.append(f"Valuation stretch: Market price is {market_to_rr_ratio:.2f}x the government Ready Reckoner benchmark (benchmark threshold 2.5x).")
        if property_input.use_case == "investment_rental" and net_rental_yield < 2.0:
            status = "WARN"
            reasons.append(f"Sub-optimal yield: Net rental yield is {net_rental_yield:.2f}%, well below standard commercial/residential inflation hurdle (2.0%).")
        if monthly_cash_deficit > monthly_disposable_surplus and monthly_cash_deficit > 0:
            status = "WARN"
            reasons.append(f"Cash flow drag: Monthly out-of-pocket deficit (INR {int(round(monthly_cash_deficit)):,}) requires partial liquidation of other reserves.")

    if not reasons:
        reasons.append(f"Pass: Net yield is {net_rental_yield:.2f}%, market/RR ratio is balanced ({market_to_rr_ratio:.2f}x), and operating cash flow is sustainable.")

    if property_input.use_case == "investment_rental" and base_res.year_1_dscr is not None and base_res.year_1_dscr < 0.50:
        reasons.append(f"Debt Coverage Note: Year-1 DSCR is {base_res.year_1_dscr:.2f}x (rent covers {base_res.year_1_dscr*100:.0f}% of EMI; remaining {100-base_res.year_1_dscr*100:.0f}% requires ongoing salary subsidy).")

    return GateResult(
        gate_number=3,
        name="Cash Flow, Yield & Valuation Sanity",
        status=status,
        metrics=metrics,
        reasons=reasons,
    )


def evaluate_gate_4_wealth_creation(
    property_input: PropertyInput,
    matrix: ScenarioMatrixResult,
) -> GateResult:
    """
    Gate 4 — Wealth Creation & Opportunity Cost
    Compares real estate equity returns against SBI post-tax Fixed Deposits and Nifty 50 TRI under parity.
    """
    base_res = matrix.base
    equity_irr = base_res.equity_irr_pct
    fd_rate = base_res.fd_benchmark.annual_return_pct
    nifty_cagr = base_res.nifty_benchmark.annual_return_pct
    nifty_post_tax_irr = base_res.nifty_benchmark.post_tax_irr_pct

    re_terminal_equity = base_res.exit.net_exit_proceeds
    fd_terminal = base_res.fd_benchmark.terminal_wealth_post_tax
    nifty_terminal = base_res.nifty_benchmark.terminal_wealth_post_tax

    advantage_vs_nifty = base_res.nifty_benchmark.net_wealth_advantage_inr
    advantage_vs_fd = base_res.fd_benchmark.net_wealth_advantage_inr

    # Risk-adjusted hurdle = post_tax_fd_rate + 400 bps (e.g. 4.16% + 4.00% = 8.16%)
    risk_adjusted_hurdle = fd_rate + 4.0

    metrics = {
        "base_equity_irr_pct": equity_irr,
        "real_equity_irr_pct": base_res.real_equity_irr_pct,
        "fd_benchmark_return_pct": fd_rate,
        "nifty_benchmark_return_pct": nifty_cagr,
        "nifty_post_tax_irr_pct": nifty_post_tax_irr,
        "risk_adjusted_hurdle_pct": risk_adjusted_hurdle,
        "re_terminal_net_equity": re_terminal_equity,
        "fd_terminal_wealth": fd_terminal,
        "nifty_terminal_wealth": nifty_terminal,
        "advantage_vs_fd": advantage_vs_fd,
        "advantage_vs_nifty": advantage_vs_nifty,
    }

    reasons: list[str] = []
    status = "PASS"

    if equity_irr is None:
        status = "FAIL"
        reasons.append("Zero/Negative equity return: The investment yields negative equity IRR across the holding period.")
    elif property_input.use_case == "investment_rental":
        if equity_irr <= fd_rate:
            status = "FAIL"
            reasons.append(f"Destroys capital efficiency: Equity IRR ({equity_irr:.2f}%) fails to beat the risk-free post-tax SBI FD rate ({fd_rate:.2f}%).")
        elif equity_irr >= risk_adjusted_hurdle:
            status = "PASS"
            nifty_irr_str = f"{nifty_post_tax_irr:.2f}%" if nifty_post_tax_irr is not None else f"{nifty_cagr:.2f}%"
            reasons.append(f"Pass: Base Equity IRR ({equity_irr:.2f}%) clears the risk-adjusted hurdle ({risk_adjusted_hurdle:.2f}%, risk-free post-tax FD + 400 bps premium). Post-tax Nifty 50 TRI comparison ({nifty_irr_str} post-tax IRR): RE terminal INR {int(round(re_terminal_equity)):,} vs Nifty INR {int(round(nifty_terminal)):,} (spread: {base_res.nifty_benchmark.irr_spread_pct:+.2f} pp).")
        else:
            status = "WARN"
            reasons.append(f"Thin risk premium: Base Equity IRR ({equity_irr:.2f}%) beats risk-free post-tax FD ({fd_rate:.2f}%), but fails to achieve the 400 bps risk premium ({risk_adjusted_hurdle:.2f}% hurdle).")
    else:
        # self_use
        if equity_irr < fd_rate - 1.0:
            status = "FAIL"
            reasons.append(f"Severe equity drag: Self-use Base Equity IRR ({equity_irr:.2f}%) lags risk-free post-tax FD by over 100 bps ({fd_rate:.2f}%).")
        elif equity_irr >= fd_rate + 2.0 or re_terminal_equity >= 0.90 * fd_terminal:
            status = "PASS"
            reasons.append(f"Pass: Self-use equity return ({equity_irr:.2f}%) satisfies housing utility with viable capital preservation vs risk-free FD ({fd_rate:.2f}%).")
        else:
            status = "WARN"
            reasons.append(f"Moderate equity drag: Base Equity IRR ({equity_irr:.2f}%) provides modest return relative to liquid FD ({fd_rate:.2f}%).")

    return GateResult(
        gate_number=4,
        name="Wealth Creation & Opportunity Cost",
        status=status,
        metrics=metrics,
        reasons=reasons,
    )


def evaluate_master_verdict(
    investor: InvestorProfile,
    property_input: PropertyInput,
    financing_input: FinancingInput,
    matrix: ScenarioMatrixResult,
) -> MasterVerdict:
    """
    Executes all 4 hierarchical gates and synthesizes the definitive Go / Caution / No-Go verdict.
    """
    base_res = matrix.base

    gate_1 = evaluate_gate_1_affordability(
        investor=investor,
        base_emi=base_res.loan.monthly_emi,
        required_upfront_cash=base_res.loan.required_upfront_cash,
    )
    gate_2 = evaluate_gate_2_downside_survival(
        investor=investor,
        property_input=property_input,
        holding_period_years=financing_input.holding_period_years,
        matrix=matrix,
    )
    gate_3 = evaluate_gate_3_yield_valuation(
        property_input=property_input,
        investor=investor,
        matrix=matrix,
    )
    gate_4 = evaluate_gate_4_wealth_creation(
        property_input=property_input,
        matrix=matrix,
    )

    all_gates = [gate_1, gate_2, gate_3, gate_4]

    # Master verdict logic:
    # 1. NO-GO (Red): Gate 1 or Gate 2 is FAIL or (investment_rental and Gate 4 is FAIL)
    if (
        gate_1.status == "FAIL"
        or gate_2.status == "FAIL"
        or (property_input.use_case == "investment_rental" and gate_4.status == "FAIL")
    ):
        status = "NO-GO"
        color = "red"
        summary = "CRITICAL RISK: Deal breaches mandatory solvency, downside risk survival, or minimum capital efficiency gates. Proceeding risks capital destruction or insolvency."
    # 2. GO (Green): All 4 gates PASS
    elif all(g.status == "PASS" for g in all_gates):
        status = "GO"
        color = "green"
        summary = "HIGH CONVICTION: Deal satisfies all 4 underwriting gates across affordability, downside survival, valuation sanity, and opportunity cost."
    # 3. CAUTION (Amber): All remaining cases
    else:
        status = "CAUTION"
        color = "amber"
        summary = "CONDITIONAL / CAUTION: Deal is financially survivable, but carries valuation stretches, low rental yield, or underperforms liquid benchmarks."

    return MasterVerdict(
        status=status,
        color=color,
        summary=summary,
        gates=all_gates,
    )


evaluate_decision_gates = evaluate_master_verdict

