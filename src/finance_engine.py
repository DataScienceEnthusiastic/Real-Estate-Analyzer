from __future__ import annotations

import math
from typing import Optional, Tuple
from scipy.optimize import brentq

from src.models import (
    AcquisitionCostBreakdown,
    AlternativeBenchmarkResult,
    CashFlowYear,
    ExitValuation,
    FinancingInput,
    InvestorProfile,
    LoanBreakdown,
    PropertyInput,
    ScenarioMatrixResult,
    StressTestResult,
    UnderwritingResult,
)


def compute_acquisition_costs(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    macro: dict[str, float],
) -> AcquisitionCostBreakdown:
    """
    Computes statutory acquisition floors, taxes, and all-in cash requirement.
    TDS is flagged for compliance but strictly excluded from all-in cost.
    """
    agreement_val = float(property_input.agreement_value_inr)
    carpet_sqft = float(property_input.carpet_area_sqft)
    rr_rate_sqft = float(property_input.rr_rate_primary_sqft)

    # Statutory floor: ready reckoner market value
    rr_market_value = carpet_sqft * rr_rate_sqft
    stamp_duty_base = max(agreement_val, rr_market_value)

    # Base stamp duty by jurisdiction
    urban_jurisdictions = {"PMC", "PCMC", "Cantonment"}
    if property_input.jurisdiction in urban_jurisdictions:
        base_sd_rate = macro["stamp_duty_urban_male_pct"]
    else:
        base_sd_rate = macro["stamp_duty_rural_male_pct"]

    # Ownership rebate (sole female, joint female, or 50:50 male-female split)
    rebate = 0.0
    female_rebate = macro["stamp_duty_female_rebate_pct"]
    if property_input.buyer_ownership in {"sole_female", "female_joint"}:
        rebate = female_rebate
    elif property_input.buyer_ownership == "male_female_50_50":
        rebate = female_rebate * 0.5

    effective_sd_rate = max(0.0, base_sd_rate - rebate)
    stamp_duty_inr = stamp_duty_base * effective_sd_rate

    # Registration fee: 1% capped at statutory limit (default INR 30,000)
    raw_reg_fee = stamp_duty_base * macro["registration_fee_pct"]
    registration_fee_inr = min(raw_reg_fee, macro["registration_fee_cap_inr"])

    # GST for under-construction units
    gst_inr = 0.0
    gst_rate = 0.0
    is_affordable_gst = False
    if property_input.property_status == "under_construction":
        # Affordable housing threshold: <= 45 Lakhs and <= 60 sq m (~645.83 sq ft)
        is_affordable_gst = (agreement_val <= 4_500_000.0) and (carpet_sqft <= 645.83)
        gst_rate = macro["gst_uc_affordable_pct"] if is_affordable_gst else macro["gst_uc_non_affordable_pct"]
        gst_inr = agreement_val * gst_rate

    # Brokerage (1.5% + 18% GST on brokerage)
    buy_brokerage_inr = 0.0
    if property_input.brokerage_enabled:
        buy_brokerage_inr = agreement_val * macro["standard_resale_brokerage_pct"] * 1.18

    # All-in acquisition cost
    interiors = float(financing_input.interiors_and_misc_inr)
    all_in_acquisition_cost = (
        agreement_val
        + stamp_duty_inr
        + registration_fee_inr
        + gst_inr
        + buy_brokerage_inr
        + interiors
    )

    # Section 194-IA / 393(1) TDS compliance flag
    tds_applicable = stamp_duty_base >= macro["tds_property_threshold_inr"]
    tds_amount_inr = stamp_duty_base * macro["tds_property_rate_pct"] if tds_applicable else 0.0

    return AcquisitionCostBreakdown(
        agreement_value_inr=agreement_val,
        rr_market_value=rr_market_value,
        stamp_duty_base=stamp_duty_base,
        stamp_duty_inr=stamp_duty_inr,
        effective_stamp_duty_pct=effective_sd_rate,
        registration_fee_inr=registration_fee_inr,
        gst_inr=gst_inr,
        gst_rate=gst_rate,
        is_affordable_gst=is_affordable_gst,
        buy_brokerage_inr=buy_brokerage_inr,
        interiors_and_misc_inr=interiors,
        all_in_acquisition_cost=all_in_acquisition_cost,
        tds_applicable=tds_applicable,
        tds_amount_inr=tds_amount_inr,
    )


def compute_rbi_max_ltv(agreement_value_inr: float, macro: dict[str, float]) -> float:
    """Computes RBI regulatory LTV cap strictly against agreement value."""
    if agreement_value_inr <= 3_000_000.0:
        return macro["max_ltv_upto_30l_pct"]
    elif agreement_value_inr <= 7_500_000.0:
        return macro["max_ltv_30l_to_75l_pct"]
    else:
        return macro["max_ltv_above_75l_pct"]


def calculate_monthly_emi(principal: float, annual_rate: float, tenure_years: int) -> float:
    """Standard fixed monthly EMI calculation."""
    if principal <= 0.0 or annual_rate <= 0.0 or tenure_years <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = tenure_years * 12
    factor = (1.0 + r) ** n
    return principal * (r * factor) / (factor - 1.0)


def build_amortization_schedule(
    loan_amount: float,
    annual_rate: float,
    tenure_years: int,
    holding_period_years: int,
    under_construction: bool = False,
    uc_months: int = 0,
) -> Tuple[list[dict[str, float]], list[dict[str, float]], float, float, float]:
    """
    Builds exact month-by-month loan amortization schedule and aggregates by year.
    Returns: (monthly_schedule, annual_schedule, monthly_emi, monthly_pre_emi, remaining_loan_balance_H).
    """
    monthly_schedule: list[dict[str, float]] = []
    annual_schedule: list[dict[str, float]] = []

    if loan_amount <= 0.0:
        for y in range(1, holding_period_years + 1):
            annual_schedule.append({
                "year": y,
                "annual_emi": 0.0,
                "interest_paid": 0.0,
                "principal_paid": 0.0,
                "closing_balance": 0.0,
            })
        return monthly_schedule, annual_schedule, 0.0, 0.0, 0.0

    monthly_rate = annual_rate / 12.0
    regular_emi = calculate_monthly_emi(loan_amount, annual_rate, tenure_years)

    # Pre-EMI during construction assuming linear average drawdown of 55%
    pre_emi_interest = (loan_amount * 0.55) * monthly_rate if under_construction else 0.0

    total_months = max(holding_period_years * 12, (uc_months if under_construction else 0) + tenure_years * 12)
    current_balance = loan_amount

    # Month-by-month simulation
    for m in range(1, total_months + 1):
        if under_construction and m <= uc_months:
            # Pre-EMI phase: interest only on drawn funds, no principal reduction
            interest_paid = pre_emi_interest
            principal_paid = 0.0
            emi_paid = pre_emi_interest
        else:
            if current_balance <= 0.0:
                interest_paid = 0.0
                principal_paid = 0.0
                emi_paid = 0.0
                current_balance = 0.0
            else:
                interest_paid = current_balance * monthly_rate
                # Check for loan closure in final month
                if current_balance + interest_paid <= regular_emi:
                    principal_paid = current_balance
                    emi_paid = principal_paid + interest_paid
                    current_balance = 0.0
                else:
                    principal_paid = regular_emi - interest_paid
                    current_balance = max(0.0, current_balance - principal_paid)
                    emi_paid = regular_emi

        monthly_schedule.append({
            "month": m,
            "opening_balance": current_balance + principal_paid,
            "emi": emi_paid,
            "interest_paid": interest_paid,
            "principal_paid": principal_paid,
            "closing_balance": current_balance,
        })

    # Year-by-year aggregation for 1..H
    remaining_balance_H = 0.0
    for y in range(1, holding_period_years + 1):
        m_start = (y - 1) * 12
        m_end = y * 12
        year_slice = monthly_schedule[m_start:m_end]

        sum_emi = sum(row["emi"] for row in year_slice)
        sum_interest = sum(row["interest_paid"] for row in year_slice)
        sum_principal = sum(row["principal_paid"] for row in year_slice)
        closing_bal = year_slice[-1]["closing_balance"] if year_slice else 0.0

        if y == holding_period_years:
            remaining_balance_H = closing_bal

        annual_schedule.append({
            "year": y,
            "annual_emi": sum_emi,
            "interest_paid": sum_interest,
            "principal_paid": sum_principal,
            "closing_balance": closing_bal,
        })

    return monthly_schedule, annual_schedule, regular_emi, pre_emi_interest, remaining_balance_H


def _investor_tax_multiplier(investor: InvestorProfile) -> float:
    """Combined surcharge and 4% Health & Education Cess multiplier."""
    return (1.0 + max(0.0, investor.surcharge_rate_pct)) * 1.04


def _cumulative_appreciation_factor(
    annual_rate: float,
    years: int,
    mean_revert: bool = False,
    anchor_rate: float = 0.065,
) -> float:
    """Computes compounded appreciation factor over `years`, with optional Year-6+ mean reversion."""
    if years <= 0:
        return 1.0
    if not mean_revert or years <= 5:
        return (1.0 + annual_rate) ** years

    factor = (1.0 + annual_rate) ** 5
    for yr in range(6, years + 1):
        # Smoothly revert 75% of the spread toward long-term Pune equilibrium by Year 10
        w = min(1.0, (yr - 5) / 5.0) * 0.75
        rate_yr = (1.0 - w) * annual_rate + w * anchor_rate
        factor *= (1.0 + rate_yr)
    return factor


def compute_loan_breakdown(
    acquisition: AcquisitionCostBreakdown,
    financing_input: FinancingInput,
    property_input: PropertyInput,
    macro: dict[str, float],
    scenario_loan_rate: Optional[float] = None,
    scenario_uc_delay_months: int = 0,
) -> LoanBreakdown:
    """Calculates borrowing eligibility under RBI caps and generates amortization tables."""
    agreement_val = acquisition.agreement_value_inr
    rbi_max = compute_rbi_max_ltv(agreement_val, macro)
    effective_ltv = min(financing_input.desired_ltv_pct, rbi_max)
    loan_amount = agreement_val * effective_ltv
    loan_orig_motd = loan_amount * max(0.0, financing_input.loan_origination_and_motd_pct)
    required_upfront_cash = acquisition.all_in_acquisition_cost - loan_amount + loan_orig_motd

    annual_rate = scenario_loan_rate if scenario_loan_rate is not None else financing_input.home_loan_interest_rate_pct
    is_uc = property_input.property_status == "under_construction"
    uc_months = (property_input.expected_possession_months + scenario_uc_delay_months) if is_uc else 0

    monthly_sched, annual_sched, reg_emi, pre_emi, remaining_bal_H = build_amortization_schedule(
        loan_amount=loan_amount,
        annual_rate=annual_rate,
        tenure_years=financing_input.loan_tenure_years,
        holding_period_years=financing_input.holding_period_years,
        under_construction=is_uc,
        uc_months=uc_months,
    )

    return LoanBreakdown(
        rbi_max_ltv=rbi_max,
        effective_ltv=effective_ltv,
        loan_amount=loan_amount,
        required_upfront_cash=required_upfront_cash,
        monthly_emi=reg_emi,
        monthly_pre_emi=pre_emi,
        monthly_schedule=monthly_sched,
        annual_schedule=annual_sched,
        remaining_loan_balance_H=remaining_bal_H,
        loan_origination_and_motd_inr=loan_orig_motd,
        home_loan_interest_rate_pct=annual_rate,
        loan_tenure_years=financing_input.loan_tenure_years,
        holding_period_years=financing_input.holding_period_years,
    )


def compute_operating_cash_flows(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    acquisition: AcquisitionCostBreakdown,
    loan: LoanBreakdown,
    macro: dict[str, float],
    scenario_vacancy_months: float,
    scenario_appreciation_rate: float,
    scenario_uc_delay_months: int = 0,
    scenario_exit_discount_pct: float = 0.0,
) -> list[CashFlowYear]:
    """
    Computes year-by-year operational cash flow, opex inflation, interest deduction, and tax impact.
    Differentiates between investment rental and self-use scenarios according to Indian tax code.
    Also computes mark-to-market net realizable equity at year-end.
    """
    cash_flows: list[CashFlowYear] = []
    h = financing_input.holding_period_years
    is_uc = property_input.property_status == "under_construction"
    total_uc_months = (property_input.expected_possession_months + scenario_uc_delay_months) if is_uc else 0

    base_rent = property_input.monthly_rent_inr
    base_maint = financing_input.monthly_society_maintenance_inr
    base_ptax = financing_input.annual_property_tax_inr

    tax_cost_basis = (
        acquisition.agreement_value_inr
        + acquisition.stamp_duty_inr
        + acquisition.registration_fee_inr
        + acquisition.gst_inr
    )
    tax_mult = _investor_tax_multiplier(investor)

    # Pre-construction interest (Sec 24(b)): deferred during construction and amortized 1/5th per year post-possession
    pre_construction_interest_pool = 0.0
    pre_emi_annual_installment = 0.0
    pre_emi_amort_years_left = 0
    # Unabsorbed HP losses under Old Regime carried forward up to 8 years (Sec 71B)
    loss_cf_queue: list[list[float]] = []

    for t in range(1, h + 1):
        m_start = (t - 1) * 12

        # Construction phase duration in year t
        if is_uc:
            uc_in_year = max(0, min(12, total_uc_months - m_start))
        else:
            uc_in_year = 0
        active_rent_months = max(0, 12 - uc_in_year)

        # Escalations: rent at rent_escalation, opex at configured CPI inflation
        rent_multiplier = (1.0 + financing_input.annual_rent_escalation_pct) ** (t - 1)
        opex_multiplier = (1.0 + financing_input.cpi_inflation_pct) ** (t - 1)

        gross_annual_rent_t = (base_rent * active_rent_months) * rent_multiplier

        # Society maintenance, property tax, and CapEx/re-letting reserve
        annual_ptax_t = base_ptax * opex_multiplier
        monthly_maint_t = base_maint * opex_multiplier

        # Loan outflows for year t
        annual_loan_row = loan.annual_schedule[t - 1]
        annual_emi_t = annual_loan_row["annual_emi"]
        annual_interest_t = annual_loan_row["interest_paid"]
        annual_principal_t = annual_loan_row["principal_paid"]
        closing_loan_bal_t = annual_loan_row["closing_balance"]

        # Section 24(b) pre-construction interest pooling vs post-possession 5-yr amortization
        pre_emi_amort_t = 0.0
        loss_cf_used_t = 0.0
        if is_uc and active_rent_months == 0:
            pre_construction_interest_pool += annual_interest_t
            deductible_interest_t = 0.0
        else:
            if pre_construction_interest_pool > 0.0 and pre_emi_amort_years_left == 0:
                pre_emi_annual_installment = pre_construction_interest_pool / 5.0
                pre_construction_interest_pool = 0.0
                pre_emi_amort_years_left = 5
            if pre_emi_amort_years_left > 0:
                pre_emi_amort_t = pre_emi_annual_installment
                pre_emi_amort_years_left -= 1
            deductible_interest_t = annual_interest_t + pre_emi_amort_t

        if property_input.use_case == "investment_rental":
            # Vacancy allowance applies only to active tenancy months
            vacancy_loss_t = gross_annual_rent_t * (scenario_vacancy_months / 12.0) if active_rent_months > 0 else 0.0
            egi_t = max(0.0, gross_annual_rent_t - vacancy_loss_t)
            capex_reserve_t = egi_t * max(0.0, financing_input.capex_and_turnover_reserve_pct) if active_rent_months > 0 else 0.0
            opex_t = (monthly_maint_t * active_rent_months) + annual_ptax_t + capex_reserve_t
            noi_t = egi_t - opex_t

            # Income tax from house property (Section 24(a) / Section 22(1)(a))
            nav_t = max(0.0, egi_t - annual_ptax_t) if active_rent_months > 0 else 0.0
            std_deduction_t = nav_t * macro["rental_std_deduction_pct"]
            net_hp_income_t = (nav_t - std_deduction_t - deductible_interest_t) if active_rent_months > 0 else 0.0

            # Tax regime branching
            if investor.tax_regime == "new":
                # New regime blocks house property loss set-off against salary
                tax_impact_t = max(0.0, net_hp_income_t) * investor.marginal_tax_slab_pct * tax_mult
            else:
                # Prune expired Sec 71B losses (> 8 years)
                loss_cf_queue = [item for item in loss_cf_queue if item[0] >= t and item[1] > 0.0]
                if net_hp_income_t > 0.0 and loss_cf_queue:
                    remaining_profit = net_hp_income_t
                    for item in loss_cf_queue:
                        if remaining_profit <= 0.0:
                            break
                        absorbed = min(remaining_profit, item[1])
                        item[1] -= absorbed
                        remaining_profit -= absorbed
                        loss_cf_used_t += absorbed
                    tax_impact_t = remaining_profit * investor.marginal_tax_slab_pct * tax_mult
                else:
                    # Old regime caps salary loss set-off at INR 2,00,000 per year; excess carried forward 8 years
                    taxable_adj_t = max(-200_000.0, net_hp_income_t)
                    tax_impact_t = taxable_adj_t * investor.marginal_tax_slab_pct * tax_mult
                    unabsorbed_loss = max(0.0, -(net_hp_income_t - taxable_adj_t))
                    if unabsorbed_loss > 0.0:
                        loss_cf_queue.append([t + 8, unabsorbed_loss])

            net_cash_flow_t = noi_t - annual_emi_t - tax_impact_t

        else:
            # Self-use: no vacancy, imputed rent saved, and Section 24(b) interest deduction
            vacancy_loss_t = 0.0
            egi_t = gross_annual_rent_t
            capex_reserve_t = egi_t * max(0.0, financing_input.capex_and_turnover_reserve_pct) if active_rent_months > 0 else 0.0
            opex_t = (monthly_maint_t * active_rent_months) + annual_ptax_t + capex_reserve_t
            noi_t = egi_t - opex_t

            # Old regime allows up to INR 2,00,000 interest deduction on self-occupied property (only post-possession)
            if investor.tax_regime == "old" and active_rent_months > 0:
                self_use_tax_saving_t = min(deductible_interest_t, 200_000.0) * investor.marginal_tax_slab_pct * tax_mult
            else:
                self_use_tax_saving_t = 0.0

            net_annual_outflow_buy_t = annual_emi_t + opex_t - self_use_tax_saving_t
            # Net economic cash flow vs renting
            net_cash_flow_t = gross_annual_rent_t - net_annual_outflow_buy_t
            tax_impact_t = -self_use_tax_saving_t

        # Property valuation and gross net equity at year end
        appr_factor_t = _cumulative_appreciation_factor(
            scenario_appreciation_rate,
            t,
            mean_revert=financing_input.mean_revert_appreciation_after_yr5,
            anchor_rate=financing_input.long_term_appreciation_anchor_pct,
        )
        terminal_prop_val_t = acquisition.agreement_value_inr * appr_factor_t
        net_equity_t = max(0.0, terminal_prop_val_t - closing_loan_bal_t)

        # Mark-to-market net realizable equity at year t end (post selling brokerage, LTCG tax & debt payoff)
        exit_price_t = terminal_prop_val_t * (1.0 - scenario_exit_discount_pct)
        sell_brok_t = exit_price_t * macro["standard_resale_brokerage_pct"] * 1.18 if property_input.sell_brokerage_enabled else 0.0
        cg_t = max(0.0, exit_price_t - sell_brok_t - tax_cost_basis)
        cg_rate_t = (
            macro["ltcg_tax_rate_post_july2024_pct"]
            if t > macro["ltcg_holding_threshold_years"]
            else investor.marginal_tax_slab_pct
        )
        cg_tax_t = cg_t * cg_rate_t * tax_mult
        net_realizable_equity_t = exit_price_t - sell_brok_t - closing_loan_bal_t - cg_tax_t

        cash_flows.append(CashFlowYear(
            year=t,
            gross_annual_rent=gross_annual_rent_t,
            vacancy_loss=vacancy_loss_t,
            egi=egi_t,
            opex=opex_t,
            noi=noi_t,
            annual_emi=annual_emi_t,
            interest_paid=annual_interest_t,
            principal_paid=annual_principal_t,
            tax_impact=tax_impact_t,
            net_cash_flow=net_cash_flow_t,
            remaining_loan_balance=closing_loan_bal_t,
            net_equity=net_equity_t,
            net_realizable_equity=net_realizable_equity_t,
            nav=nav_t if property_input.use_case == "investment_rental" else 0.0,
            std_deduction=std_deduction_t if property_input.use_case == "investment_rental" else 0.0,
            net_hp_income=net_hp_income_t if property_input.use_case == "investment_rental" else -deductible_interest_t,
            capex_reserve=capex_reserve_t,
            pre_emi_amort_deduction=pre_emi_amort_t,
            loss_cf_used=loss_cf_used_t,
        ))

    return cash_flows


def compute_exit_valuation(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    acquisition: AcquisitionCostBreakdown,
    loan: LoanBreakdown,
    macro: dict[str, float],
    scenario_appreciation_rate: float,
    scenario_exit_discount_pct: float,
) -> ExitValuation:
    """
    Computes holding period exit valuation, brokerage, Indian capital gains tax, and net equity proceeds.
    Applies Finance (No. 2) Act 2024 post-July 2024 LTCG rules (12.5% + 4% cess after 2-year threshold).
    """
    h = financing_input.holding_period_years
    agreement_val = acquisition.agreement_value_inr
    tax_mult = _investor_tax_multiplier(investor)

    appr_factor_H = _cumulative_appreciation_factor(
        scenario_appreciation_rate,
        h,
        mean_revert=financing_input.mean_revert_appreciation_after_yr5,
        anchor_rate=financing_input.long_term_appreciation_anchor_pct,
    )
    terminal_gross_val = agreement_val * appr_factor_H
    exit_sale_price = terminal_gross_val * (1.0 - scenario_exit_discount_pct)

    sell_brokerage_inr = 0.0
    if property_input.sell_brokerage_enabled:
        sell_brokerage_inr = exit_sale_price * macro["standard_resale_brokerage_pct"] * 1.18

    # Tax cost basis includes stamp duty, registration, and GST (statutory capital additions)
    tax_cost_basis = (
        acquisition.agreement_value_inr
        + acquisition.stamp_duty_inr
        + acquisition.registration_fee_inr
        + acquisition.gst_inr
    )

    # Net consideration after selling expenses
    capital_gain = max(0.0, exit_sale_price - sell_brokerage_inr - tax_cost_basis)

    # LTCG threshold: 24 months (2.0 years)
    is_ltcg = h > macro["ltcg_holding_threshold_years"]
    if is_ltcg:
        cg_tax_inr = capital_gain * macro["ltcg_tax_rate_post_july2024_pct"] * tax_mult
    else:
        cg_tax_inr = capital_gain * investor.marginal_tax_slab_pct * tax_mult

    remaining_loan = loan.remaining_loan_balance_H
    net_exit_proceeds = exit_sale_price - sell_brokerage_inr - remaining_loan - cg_tax_inr

    return ExitValuation(
        terminal_gross_value=terminal_gross_val,
        exit_sale_price=exit_sale_price,
        sell_brokerage_inr=sell_brokerage_inr,
        tax_cost_basis=tax_cost_basis,
        capital_gain=capital_gain,
        is_ltcg=is_ltcg,
        cg_tax_inr=cg_tax_inr,
        remaining_loan_balance_H=remaining_loan,
        net_exit_proceeds=net_exit_proceeds,
    )


def calculate_equity_irr(cash_flows: list[float]) -> Optional[float]:
    """
    Computes exact Equity IRR using Brent's method on NPV(r) = 0.
    Returns None if no root exists or all cash flows are negative.
    """
    if not cash_flows or len(cash_flows) < 2:
        return None

    # Check if there's at least one positive and one negative cash flow
    has_negative = any(cf < 0.0 for cf in cash_flows)
    has_positive = any(cf > 0.0 for cf in cash_flows)
    if not (has_negative and has_positive):
        return None

    def npv_func(r: float) -> float:
        total = 0.0
        for i, cf in enumerate(cash_flows):
            total += cf / ((1.0 + r) ** i)
        return total

    # Search for root in bracket [-0.95, 10.0]
    try:
        f_low = npv_func(-0.95)
        f_high = npv_func(10.0)
        if f_low * f_high > 0:
            # Try a narrower standard bracket [ -0.5, 2.0 ]
            f_low = npv_func(-0.50)
            f_high = npv_func(2.0)
            if f_low * f_high > 0:
                return None
            root = brentq(npv_func, -0.50, 2.0, maxiter=200)
        else:
            root = brentq(npv_func, -0.95, 10.0, maxiter=200)
        return float(root * 100.0)
    except (ValueError, RuntimeError):
        return None


def compute_liquid_benchmarks(
    required_upfront_cash: float,
    annual_cash_flows: list[CashFlowYear],
    re_terminal_net_equity: float,
    re_irr_pct: Optional[float],
    investor: InvestorProfile,
    financing_input: FinancingInput,
    macro: dict[str, float],
) -> Tuple[AlternativeBenchmarkResult, AlternativeBenchmarkResult]:
    """
    Simulates SBI Term Deposit and Nifty 50 TRI under strict annual cash-flow parity.
    Every annual deficit in real estate is funded by the benchmark investor as an extra deposit.
    """
    h = financing_input.holding_period_years

    tax_mult = _investor_tax_multiplier(investor)

    # SBI FD post-tax return
    fd_gross = macro["liquid_benchmark_fd_rate_pct"]
    fd_post_tax = fd_gross * max(0.0, 1.0 - investor.marginal_tax_slab_pct * tax_mult)

    # Nifty 50 TRI return
    nifty_cagr = financing_input.nifty_benchmark_cagr_pct

    # Year 0 initial investment
    fd_wealth = required_upfront_cash
    nifty_wealth = required_upfront_cash
    nifty_principal_invested = required_upfront_cash

    fd_trajectory: list[float] = [fd_wealth]
    nifty_trajectory: list[float] = [nifty_wealth]

    for idx, cf in enumerate(annual_cash_flows, start=1):
        # Cash-flow parity: real estate deficit (-CF) requires equivalent benchmark deposit (+parity_diff)
        parity_diff = -cf.net_cash_flow

        # FD annual compounding + parity adjustment
        fd_wealth = max(0.0, fd_wealth * (1.0 + fd_post_tax) + parity_diff)
        fd_trajectory.append(fd_wealth)

        # Nifty gross compounding + parity adjustment with proportional gain tax on withdrawals
        nifty_pre_wd = nifty_wealth * (1.0 + nifty_cagr)
        if parity_diff >= 0.0:
            nifty_wealth = max(0.0, nifty_pre_wd + parity_diff)
            nifty_principal_invested += parity_diff
        else:
            withdrawal = min(nifty_pre_wd, -parity_diff)
            if nifty_pre_wd > 0.0:
                cost_fraction = min(1.0, nifty_principal_invested / nifty_pre_wd)
                principal_withdrawn = withdrawal * cost_fraction
                realized_gain = max(0.0, withdrawal - principal_withdrawn)
                nifty_principal_invested = max(0.0, nifty_principal_invested - principal_withdrawn)
                if idx > 1:
                    wd_tax = max(0.0, realized_gain - 125_000.0) * 0.125 * tax_mult
                else:
                    wd_tax = realized_gain * 0.20 * tax_mult
                nifty_wealth = max(0.0, nifty_pre_wd - withdrawal - wd_tax)
            else:
                nifty_wealth = 0.0
        nifty_trajectory.append(nifty_wealth)

    # Nifty exit tax: 12.5% * tax_mult on equity capital gains above INR 1.25 Lakh exemption
    nifty_capital_gain = max(0.0, nifty_wealth - nifty_principal_invested)
    taxable_nifty_gain = max(0.0, nifty_capital_gain - 125_000.0)
    nifty_exit_tax = taxable_nifty_gain * 0.125 * tax_mult
    nifty_terminal_post_tax = max(0.0, nifty_wealth - nifty_exit_tax)

    # True post-tax Nifty 50 IRR on exact cash-flow parity stream
    nifty_cfs = [-required_upfront_cash] + [cf.net_cash_flow for cf in annual_cash_flows]
    nifty_cfs[-1] += nifty_terminal_post_tax
    nifty_post_tax_irr = calculate_equity_irr(nifty_cfs)

    # Spreads and advantages
    fd_advantage = re_terminal_net_equity - fd_wealth
    nifty_advantage = re_terminal_net_equity - nifty_terminal_post_tax

    fd_irr_spread = (re_irr_pct - (fd_post_tax * 100.0)) if re_irr_pct is not None else 0.0
    nifty_irr_spread = (re_irr_pct - nifty_post_tax_irr) if (re_irr_pct is not None and nifty_post_tax_irr is not None) else 0.0

    fd_result = AlternativeBenchmarkResult(
        benchmark_name="SBI Fixed Deposit (Post-Tax)",
        annual_return_pct=fd_post_tax * 100.0,
        terminal_wealth_post_tax=fd_wealth,
        net_wealth_advantage_inr=fd_advantage,
        irr_spread_pct=fd_irr_spread,
        trajectory=fd_trajectory,
        gross_wealth=fd_wealth,
        principal_invested=required_upfront_cash + sum(max(0.0, -cf.net_cash_flow) for cf in annual_cash_flows),
        exit_tax=0.0,
        pre_tax_cagr_pct=fd_gross * 100.0,
        post_tax_irr_pct=fd_post_tax * 100.0,
    )

    nifty_result = AlternativeBenchmarkResult(
        benchmark_name="Nifty 50 TRI (Post-Tax)",
        annual_return_pct=nifty_post_tax_irr if nifty_post_tax_irr is not None else (nifty_cagr * 100.0),
        terminal_wealth_post_tax=nifty_terminal_post_tax,
        net_wealth_advantage_inr=nifty_advantage,
        irr_spread_pct=nifty_irr_spread,
        trajectory=nifty_trajectory,
        gross_wealth=nifty_wealth,
        principal_invested=nifty_principal_invested,
        exit_tax=nifty_exit_tax,
        pre_tax_cagr_pct=nifty_cagr * 100.0,
        post_tax_irr_pct=nifty_post_tax_irr,
    )

    return fd_result, nifty_result


def run_scenario_underwriting(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
    scenario_appreciation_rate: float,
    scenario_vacancy_months: float,
    scenario_loan_rate: float,
    scenario_exit_discount_pct: float,
    scenario_uc_delay_months: int = 0,
) -> UnderwritingResult:
    """Executes a complete underwriting evaluation for a given scenario."""
    acq = compute_acquisition_costs(property_input, financing_input, macro)
    loan = compute_loan_breakdown(
        acq, financing_input, property_input, macro,
        scenario_loan_rate=scenario_loan_rate,
        scenario_uc_delay_months=scenario_uc_delay_months,
    )
    annual_cfs = compute_operating_cash_flows(
        property_input, financing_input, investor, acq, loan, macro,
        scenario_vacancy_months=scenario_vacancy_months,
        scenario_appreciation_rate=scenario_appreciation_rate,
        scenario_uc_delay_months=scenario_uc_delay_months,
        scenario_exit_discount_pct=scenario_exit_discount_pct,
    )
    exit_val = compute_exit_valuation(
        property_input, financing_input, investor, acq, loan, macro,
        scenario_appreciation_rate=scenario_appreciation_rate,
        scenario_exit_discount_pct=scenario_exit_discount_pct,
    )

    # Cash-flow vector: t=0 upfront cash, t=1..H operating CFs + exit net proceeds
    cf_vector = [-loan.required_upfront_cash] + [cf.net_cash_flow for cf in annual_cfs]
    cf_vector[-1] += exit_val.net_exit_proceeds

    equity_irr = calculate_equity_irr(cf_vector)

    # Real (CPI inflation-adjusted) Equity IRR via Fisher equation
    real_equity_irr = None
    if equity_irr is not None and (1.0 + financing_input.cpi_inflation_pct) > 0:
        real_equity_irr = (
            ((1.0 + equity_irr / 100.0) / (1.0 + financing_input.cpi_inflation_pct)) - 1.0
        ) * 100.0

    # NPV discounted at post-tax FD rate
    tax_mult = _investor_tax_multiplier(investor)
    fd_post_tax = macro["liquid_benchmark_fd_rate_pct"] * max(0.0, 1.0 - investor.marginal_tax_slab_pct * tax_mult)
    npv_val = sum(cf / ((1.0 + fd_post_tax) ** idx) for idx, cf in enumerate(cf_vector))

    # Benchmarks
    fd_bench, nifty_bench = compute_liquid_benchmarks(
        required_upfront_cash=loan.required_upfront_cash,
        annual_cash_flows=annual_cfs,
        re_terminal_net_equity=exit_val.net_exit_proceeds,
        re_irr_pct=equity_irr,
        investor=investor,
        financing_input=financing_input,
        macro=macro,
    )

    # Year 1 yield metrics & DSCR
    y1_gross = annual_cfs[0].gross_annual_rent
    y1_noi = annual_cfs[0].noi
    gross_yield = (y1_gross / acq.all_in_acquisition_cost * 100.0) if acq.all_in_acquisition_cost > 0 else 0.0
    net_yield = (y1_noi / acq.all_in_acquisition_cost * 100.0) if acq.all_in_acquisition_cost > 0 else 0.0

    y1_emi = annual_cfs[0].annual_emi
    year_1_dscr = (y1_noi / y1_emi) if y1_emi > 0 else None
    dscr_samples = [cf.noi / cf.annual_emi for cf in annual_cfs if cf.annual_emi > 0 and cf.gross_annual_rent > 0]
    avg_dscr = (sum(dscr_samples) / len(dscr_samples)) if dscr_samples else year_1_dscr

    net_realizable_equity_trajectory = [loan.required_upfront_cash] + [
        cf.net_realizable_equity for cf in annual_cfs
    ]

    return UnderwritingResult(
        acquisition=acq,
        loan=loan,
        annual_cash_flows=annual_cfs,
        exit=exit_val,
        equity_cash_flow_vector=cf_vector,
        equity_irr_pct=equity_irr,
        npv_inr=npv_val,
        fd_benchmark=fd_bench,
        nifty_benchmark=nifty_bench,
        year_1_gross_yield_pct=gross_yield,
        year_1_net_yield_pct=net_yield,
        net_realizable_equity_trajectory=net_realizable_equity_trajectory,
        real_equity_irr_pct=real_equity_irr,
        year_1_dscr=year_1_dscr,
        avg_dscr=avg_dscr,
        scenario_appreciation_rate=scenario_appreciation_rate,
        scenario_loan_rate=scenario_loan_rate,
        scenario_vacancy_months=scenario_vacancy_months,
        scenario_exit_discount_pct=scenario_exit_discount_pct,
        scenario_uc_delay_months=scenario_uc_delay_months,
    )


def run_full_underwriting_matrix(
    property_input: PropertyInput,
    financing_input: FinancingInput,
    investor: InvestorProfile,
    macro: dict[str, float],
) -> ScenarioMatrixResult:
    """Runs Best, Base, and Worst scenarios along with dedicated stress test modules."""
    med_appreciation = property_input.appreciation_rate
    base_loan_rate = financing_input.home_loan_interest_rate_pct

    # 1. Best Case
    best_res = run_scenario_underwriting(
        property_input=property_input,
        financing_input=financing_input,
        investor=investor,
        macro=macro,
        scenario_appreciation_rate=med_appreciation + 0.015,
        scenario_vacancy_months=property_input.vacancy_months_best,
        scenario_loan_rate=max(0.065, base_loan_rate - 0.005),
        scenario_exit_discount_pct=0.00,
        scenario_uc_delay_months=0,
    )

    # 2. Base Case
    base_res = run_scenario_underwriting(
        property_input=property_input,
        financing_input=financing_input,
        investor=investor,
        macro=macro,
        scenario_appreciation_rate=med_appreciation,
        scenario_vacancy_months=property_input.vacancy_months_avg,
        scenario_loan_rate=base_loan_rate,
        scenario_exit_discount_pct=0.00,
        scenario_uc_delay_months=int(round(property_input.avg_uc_delay_months)),
    )

    # 3. Worst Case (Stress)
    worst_res = run_scenario_underwriting(
        property_input=property_input,
        financing_input=financing_input,
        investor=investor,
        macro=macro,
        scenario_appreciation_rate=max(0.02, med_appreciation - 0.025),
        scenario_vacancy_months=property_input.vacancy_months_worst,
        scenario_loan_rate=base_loan_rate + 0.020,
        scenario_exit_discount_pct=property_input.exit_discount_stress_pct,
        scenario_uc_delay_months=int(round(property_input.avg_uc_delay_months * 1.5)),
    )

    # Stress Test 1: Interest Rate Shock (+100 bps & +200 bps)
    base_emi = base_res.loan.monthly_emi
    emi_100bps = calculate_monthly_emi(base_res.loan.loan_amount, base_loan_rate + 0.01, financing_input.loan_tenure_years)
    emi_200bps = calculate_monthly_emi(base_res.loan.loan_amount, base_loan_rate + 0.02, financing_input.loan_tenure_years)
    stressed_foir_200 = (investor.existing_monthly_emis_inr + emi_200bps) / investor.net_monthly_income_inr if investor.net_monthly_income_inr > 0 else 1.0

    rate_status = "FAIL" if stressed_foir_200 > 0.60 else ("WARN" if stressed_foir_200 > 0.50 else "PASS")
    stress_rate = StressTestResult(
        name="Interest Rate Shock (+100 bps & +200 bps)",
        description="Assesses EMI escalation and stressed debt servicing ratio under sharp rate tightening.",
        metrics={
            "base_monthly_emi": base_emi,
            "emi_plus_100bps": emi_100bps,
            "emi_plus_200bps": emi_200bps,
            "delta_emi_100bps": emi_100bps - base_emi,
            "delta_emi_200bps": emi_200bps - base_emi,
            "stressed_foir_200bps": stressed_foir_200 * 100.0,
        },
        status=rate_status,
        notes=[
            f"+100 bps increases monthly EMI by INR {int(round(emi_100bps - base_emi)):,}.",
            f"+200 bps increases monthly EMI by INR {int(round(emi_200bps - base_emi)):,}, pushing FOIR to {stressed_foir_200*100:.1f}%.",
        ],
    )

    # Stress Test 2: Extended Vacancy Shock (6 months vacancy in Year 1)
    base_rent = property_input.monthly_rent_inr
    additional_vacancy_drain = base_rent * max(0.0, 6.0 - property_input.vacancy_months_avg)
    post_purchase_cash = max(0.0, investor.total_liquid_savings_inr - base_res.loan.required_upfront_cash)
    monthly_obligations = investor.monthly_living_expenses_inr + investor.existing_monthly_emis_inr + base_emi
    remaining_buffer_after_shock = (post_purchase_cash - additional_vacancy_drain) / monthly_obligations if monthly_obligations > 0 else 0.0

    vac_status = "FAIL" if remaining_buffer_after_shock < 1.0 else ("WARN" if remaining_buffer_after_shock < 3.0 else "PASS")
    stress_vac = StressTestResult(
        name=f"Extended Vacancy Shock (6 Mo Total / +{6.0 - property_input.vacancy_months_avg:.2f} Incremental Mo vs. Base)",
        description="Simulates extended tenant search period and its impact on emergency liquidity reserves.",
        metrics={
            "vacancy_cash_drain": additional_vacancy_drain,
            "post_purchase_cash": post_purchase_cash,
            "remaining_buffer_months": remaining_buffer_after_shock,
        },
        status=vac_status,
        notes=[
            f"6 months vacancy creates an additional cash drain of INR {int(round(additional_vacancy_drain)):,}.",
            f"Remaining liquid buffer post-shock covers {remaining_buffer_after_shock:.1f} months of obligations.",
        ],
    )

    # Stress Test 3: Forced Early Liquidation at Year 3
    early_exit_financing = FinancingInput(
        desired_ltv_pct=financing_input.desired_ltv_pct,
        home_loan_interest_rate_pct=financing_input.home_loan_interest_rate_pct,
        loan_tenure_years=financing_input.loan_tenure_years,
        holding_period_years=3,
        annual_rent_escalation_pct=financing_input.annual_rent_escalation_pct,
        monthly_society_maintenance_inr=financing_input.monthly_society_maintenance_inr,
        annual_property_tax_inr=financing_input.annual_property_tax_inr,
        interiors_and_misc_inr=financing_input.interiors_and_misc_inr,
        nifty_benchmark_cagr_pct=financing_input.nifty_benchmark_cagr_pct,
        cpi_inflation_pct=financing_input.cpi_inflation_pct,
        capex_and_turnover_reserve_pct=financing_input.capex_and_turnover_reserve_pct,
        loan_origination_and_motd_pct=financing_input.loan_origination_and_motd_pct,
        mean_revert_appreciation_after_yr5=financing_input.mean_revert_appreciation_after_yr5,
        long_term_appreciation_anchor_pct=financing_input.long_term_appreciation_anchor_pct,
    )
    early_exit_res = run_scenario_underwriting(
        property_input=property_input,
        financing_input=early_exit_financing,
        investor=investor,
        macro=macro,
        scenario_appreciation_rate=med_appreciation,
        scenario_vacancy_months=property_input.vacancy_months_worst,
        scenario_loan_rate=base_loan_rate,
        scenario_exit_discount_pct=property_input.exit_discount_stress_pct,
        scenario_uc_delay_months=0,
    )
    total_cash_invested_3yr = base_res.loan.required_upfront_cash + sum(
        max(0.0, -cf.net_cash_flow) for cf in early_exit_res.annual_cash_flows
    )
    net_recovered_3yr = early_exit_res.exit.net_exit_proceeds
    capital_recovery_pct = (net_recovered_3yr / total_cash_invested_3yr * 100.0) if total_cash_invested_3yr > 0 else 0.0

    exit_status = "FAIL" if capital_recovery_pct < 70.0 else ("WARN" if capital_recovery_pct < 95.0 else "PASS")
    stress_exit = StressTestResult(
        name="Forced Distress Exit at Year 3",
        description=f"Forced Distress Exit at Year 3 (Applies {property_input.exit_discount_stress_pct*100:.0f}% Haircut + Worst-Case Vacancy {property_input.vacancy_months_worst:.1f} Mo/yr in Yrs 1–3)",
        metrics={
            "total_cash_invested_3yr": total_cash_invested_3yr,
            "net_recovered_3yr": net_recovered_3yr,
            "capital_recovery_pct": capital_recovery_pct,
            "exit_discount_applied_pct": property_input.exit_discount_stress_pct * 100.0,
        },
        status=exit_status,
        notes=[
            f"Net recovered cash: INR {int(round(net_recovered_3yr)):,} against INR {int(round(total_cash_invested_3yr)):,} total equity invested.",
            f"Capital recovery ratio: {capital_recovery_pct:.1f}%.",
        ],
    )

    return ScenarioMatrixResult(
        best=best_res,
        base=base_res,
        worst=worst_res,
        stress_interest_rate_shock=stress_rate,
        stress_vacancy_shock=stress_vac,
        stress_forced_exit_year_3=stress_exit,
    )
