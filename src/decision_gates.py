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
        reasons.append(f"Not enough cash upfront: Down payment + buying fees (INR {int(round(required_upfront_cash)):,}) exceed your available savings (INR {int(round(investor.total_liquid_savings_inr)):,}).")
    elif post_purchase_cash < (3.0 * monthly_obligations):
        status = "FAIL"
        reasons.append(f"Safety net too thin: Leftover cash after buying covers only {metrics['buffer_months_actual']:.1f} months of living costs & EMIs (minimum safe floor is 3 months).")
    elif foir > 0.50:
        status = "FAIL"
        reasons.append(f"Loan EMI too heavy: Total monthly EMIs take up {foir * 100:.1f}% of your take-home pay, above the 50.0% safe limit.")

    # Warning conditions
    if status != "FAIL":
        if post_purchase_cash < min_required_reserve:
            status = "WARN"
            shortfall = min_required_reserve - post_purchase_cash
            reasons.append(f"Low emergency buffer: Leftover cash (INR {int(round(post_purchase_cash)):,}) is below your {investor.emergency_buffer_months_target:.0f}-month emergency fund + planned cash need (INR {int(round(min_required_reserve)):,}; short by INR {int(round(shortfall)):,}).")
        if 0.40 < foir <= 0.50:
            status = "WARN"
            reasons.append(f"High EMI load: Monthly EMIs take up {foir * 100:.1f}% of your income (above the comfortable 40% mark).")

    if not reasons:
        reasons.append(f"Pass: Monthly EMIs are comfortable at {foir * 100:.1f}% of income and leftover savings cover {metrics['buffer_months_actual']:.1f} months of expenses.")

    if status != "FAIL" and upfront_to_savings_pct > 85.0:
        reasons.append(f"Concentration Note: This purchase puts {upfront_to_savings_pct:.1f}% of your cash savings into one property, which cannot be sold quickly.")

    return GateResult(
        gate_number=1,
        name="Affordability & Safety Net",
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
        reasons.append(f"Interest rate shock risk: A +2% loan rate hike pushes your EMIs to {stressed_foir_200 * 100:.1f}% of your income, above the 60.0% danger limit.")
    elif remaining_vac_buffer_months < 1.0:
        status = "FAIL"
        reasons.append(f"Vacancy cash crunch: A 6-month empty period drops your cash buffer below 1 month ({remaining_vac_buffer_months:.1f} months left).")
    elif yr3_capital_recovery_ratio < 0.65 and (has_short_horizon or has_cash_need):
        status = "FAIL"
        reasons.append(f"Heavy loss on early sale: A forced sale at Year 3 returns only {yr3_capital_recovery_ratio * 100:.1f}% of the cash you put in (< 65% floor), risky with a short {holding_period_years}y plan or upcoming cash need.")

    # Warning conditions:
    # 1. 0.50 < stressed_foir_200bps <= 0.60
    # 2. vacancy_shock_buffer_months < 3.0
    # 3. liquidity_tier == 'low' (quarters_to_sell > 6.0)
    # 4. yr3_capital_recovery_ratio < 0.75
    if status != "FAIL":
        if 0.50 < stressed_foir_200 <= 0.60:
            status = "WARN"
            reasons.append(f"Tight EMI cushion: A +2% rate hike pushes your EMIs to {stressed_foir_200 * 100:.1f}% of your income.")
        if remaining_vac_buffer_months < 3.0:
            status = "WARN"
            reasons.append(f"Thin vacancy buffer: A 6-month empty period shrinks your cash reserves to {remaining_vac_buffer_months:.1f} months.")
        if is_low_liquidity:
            status = "WARN"
            reasons.append(f"Slow resale market: Area resale speed is '{property_input.liquidity_tier}' (about {property_input.quarters_to_sell:.1f} quarters to sell, vs. 6.0-quarter comfort limit).")
        if yr3_capital_recovery_ratio < 0.75:
            status = "WARN"
            reasons.append(f"Early-sale loss risk: A forced sale at Year 3 returns {yr3_capital_recovery_ratio * 100:.1f}% of your cash put in (below the 75.0% safe mark).")

    if not reasons:
        reasons.append(f"Pass: Even with a +2% rate hike, EMIs stay manageable ({stressed_foir_200 * 100:.1f}% of income), a 6-month vacancy leaves {remaining_vac_buffer_months:.1f} months of cash buffer, and Year-3 exit recovers {yr3_capital_recovery_ratio * 100:.1f}% of your cash.")

    # Contextual Governance Advisories (Informational)
    if property_input.jurisdiction.lower() == "cantonment":
        reasons.append("Cantonment Area Title Check: Verify freehold vs. Old Grant / leasehold ownership and Pune Cantonment Board building/redevelopment rules.")
    if property_input.quarters_to_sell < 1.0:
        reasons.append("Resale Time Note: Low unsold builder stock (< 1 quarter) shows a mature area; reselling an individual flat usually still takes 3–6 months.")

    return GateResult(
        gate_number=2,
        name="Tough Times & Resale Risk",
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
        reasons.append(f"Monthly shortfall too high: Year-1 extra monthly cost (INR {int(round(monthly_cash_deficit)):,}) is higher than your spare monthly income (INR {int(round(monthly_disposable_surplus)):,}).")

    if status != "FAIL":
        if market_to_rr_ratio > 2.50:
            status = "WARN"
            reasons.append(f"Price looks high vs. govt rate: Asking price is {market_to_rr_ratio:.2f}x the government Ready Reckoner rate (comfort limit 2.5x).")
        if property_input.use_case == "investment_rental" and net_rental_yield < 2.0:
            status = "WARN"
            reasons.append(f"Low rental return: Net rental return is {net_rental_yield:.2f}% per year, below the 2.0% minimum benchmark.")
        if monthly_cash_deficit > monthly_disposable_surplus and monthly_cash_deficit > 0:
            status = "WARN"
            reasons.append(f"Monthly out-of-pocket drain: Extra monthly cost (INR {int(round(monthly_cash_deficit)):,}) requires dipping into other savings.")

    if not reasons:
        reasons.append(f"Pass: Net rental return is {net_rental_yield:.2f}%, price vs. govt rate is reasonable ({market_to_rr_ratio:.2f}x), and monthly cash flow is manageable.")

    if property_input.use_case == "investment_rental" and base_res.year_1_dscr is not None and base_res.year_1_dscr < 0.50:
        reasons.append(f"Rent vs. EMI Note: Year-1 rent covers {base_res.year_1_dscr*100:.0f}% of your EMI ({base_res.year_1_dscr:.2f}x); the remaining {100-base_res.year_1_dscr*100:.0f}% must come from your salary.")

    return GateResult(
        gate_number=3,
        name="Rent, Monthly Cost & Fair Price",
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
        reasons.append("Loss-making return: Your money earns a negative annual return over this holding period.")
    elif property_input.use_case == "investment_rental":
        if equity_irr <= fd_rate:
            status = "FAIL"
            reasons.append(f"Worse than a Bank FD: Your annual return ({equity_irr:.2f}%) is lower than a safe post-tax SBI FD ({fd_rate:.2f}%).")
        elif equity_irr >= risk_adjusted_hurdle:
            status = "PASS"
            nifty_irr_str = f"{nifty_post_tax_irr:.2f}%" if nifty_post_tax_irr is not None else f"{nifty_cagr:.2f}%"
            reasons.append(f"Pass: Your annual return ({equity_irr:.2f}%) clears the post-tax FD + 4% extra reward target ({risk_adjusted_hurdle:.2f}%). vs. Nifty 50 ({nifty_irr_str} after tax): Property gives INR {int(round(re_terminal_equity)):,} vs. Nifty INR {int(round(nifty_terminal)):,} (difference: {base_res.nifty_benchmark.irr_spread_pct:+.2f}%).")
        else:
            status = "WARN"
            reasons.append(f"Low extra reward: Your annual return ({equity_irr:.2f}%) beats a post-tax SBI FD ({fd_rate:.2f}%), but misses the +4% extra reward target ({risk_adjusted_hurdle:.2f}%) for property risk.")
    else:
        # self_use
        if equity_irr < fd_rate - 1.0:
            status = "FAIL"
            reasons.append(f"High financial cost for self-use: Your annual return ({equity_irr:.2f}%) trails a safe post-tax SBI FD ({fd_rate:.2f}%) by more than 1%.")
        elif equity_irr >= fd_rate + 2.0 or re_terminal_equity >= 0.90 * fd_terminal:
            status = "PASS"
            reasons.append(f"Pass: Self-use annual return ({equity_irr:.2f}%) preserves wealth well compared to a safe Bank FD ({fd_rate:.2f}%) while giving you a home.")
        else:
            status = "WARN"
            reasons.append(f"Modest return for self-use: Your annual return ({equity_irr:.2f}%) is somewhat lower than keeping cash in a liquid Bank FD ({fd_rate:.2f}%).")

    return GateResult(
        gate_number=4,
        name="Profit vs. Bank FD & Stock Market",
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
        summary = "HIGH RISK — DO NOT BUY AS-IS: This deal fails essential safety checks (affordability, surviving tough times, or beating a basic Bank FD)."
    # 2. GO (Green): All 4 gates PASS
    elif all(g.status == "PASS" for g in all_gates):
        status = "GO"
        color = "green"
        summary = "STRONG BUY: This deal passes all 4 safety checks — affordable EMIs, safe cash buffer, fair price, and solid returns vs. Bank FD."
    # 3. CAUTION (Amber): All remaining cases
    else:
        status = "CAUTION"
        color = "amber"
        summary = "PROCEED WITH CARE: You can afford the EMIs, but watch out for a high asking price, low rental return, or lower profit than safer options."

    return MasterVerdict(
        status=status,
        color=color,
        summary=summary,
        gates=all_gates,
    )


evaluate_decision_gates = evaluate_master_verdict

