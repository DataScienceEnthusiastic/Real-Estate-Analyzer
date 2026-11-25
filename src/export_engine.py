"""Consolidated Underwriting Dashboard Export Engine.

Exports the complete real estate underwriting dashboard into a single, comprehensive,
audit-ready CSV file. Includes investor inputs, property parameters, statutory breakdown,
loan amortization, annual cash flows, multi-scenario stress tests, chart trajectories,
and decision gate proofs.
"""
from __future__ import annotations

import csv
from datetime import datetime
import io
from typing import Any, Optional
import numpy as np

from src.advanced_analysis import (
    compute_break_even_appreciation,
    compute_probability_weighted,
    compute_return_decomposition,
)
from src.charts import format_inr
from src.models import (
    FinancingInput,
    InvestorProfile,
    MasterVerdict,
    PropertyInput,
    ScenarioMatrixResult,
)


def generate_full_dashboard_csv(
    property_obj: PropertyInput,
    financing_obj: FinancingInput,
    investor_obj: InvestorProfile,
    matrix: ScenarioMatrixResult,
    verdict: MasterVerdict,
    loc_row: Any,
    unit_row: Any,
    zone_row: Any,
    macro: dict[str, float],
) -> str:
    """
    Compiles all dashboard tabs, calculations, cash-flow schedules, scenarios,
    stress tests, chart data trajectories, and inputs into a single structured CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    base_res = matrix.base
    acq = base_res.acquisition
    loan = base_res.loan
    exit_val = base_res.exit
    h = financing_obj.holding_period_years

    # Helper row writer
    def write_section_header(title: str):
        writer.writerow([])
        writer.writerow([f"=== {title} ==="])

    # ---------------------------------------------------------
    # 0. REPORT HEADER & EXECUTIVE VERDICT
    # ---------------------------------------------------------
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow(["PUNE RESIDENTIAL REAL ESTATE FINANCIAL UNDERWRITING REPORT (COMPLETE MODEL AUDIT)"])
    writer.writerow(["Generated Timestamp", now_str])
    writer.writerow(["Locality", property_obj.locality_name, "Pincode", property_obj.pincode])
    writer.writerow(["Zone", property_obj.pune_zone, "Jurisdiction", property_obj.jurisdiction])
    writer.writerow(["Configuration", property_obj.bhk.replace("_", " "), "Property Status", property_obj.property_status])
    writer.writerow(["Use Case", property_obj.use_case, "Ownership Structure", property_obj.buyer_ownership])
    writer.writerow(["Master Underwriting Verdict", verdict.status, "Color Code", verdict.color.upper()])
    writer.writerow(["Verdict Summary", verdict.summary])

    # ---------------------------------------------------------
    # 1. INVESTOR PROFILE & FINANCIAL INPUTS
    # ---------------------------------------------------------
    write_section_header("SECTION 1: INVESTOR FINANCIAL PROFILE (INPUTS)")
    writer.writerow(["Parameter", "Value", "Unit / Basis", "Underwriting Context"])
    writer.writerow(["Monthly Net Income", f"{investor_obj.net_monthly_income_inr:,.0f}", "INR / month", "Take-home salary after PF and income tax deductions"])
    writer.writerow(["Monthly Living Expenses", f"{investor_obj.monthly_living_expenses_inr:,.0f}", "INR / month", "Routine household operational expenses"])
    writer.writerow(["Existing Monthly EMIs", f"{investor_obj.existing_monthly_emis_inr:,.0f}", "INR / month", "Pre-existing car, personal, or education loans"])
    writer.writerow(["Total Liquid Savings", f"{investor_obj.total_liquid_savings_inr:,.0f}", "INR", "Cash, bank FDs, and liquid mutual funds"])
    writer.writerow(["Upcoming Non-Property Cash Need", f"{investor_obj.upcoming_cash_need_inr:,.0f}", "INR", "Committed capital for education, wedding, medical"])
    writer.writerow(["Target Emergency Reserve", f"{investor_obj.emergency_buffer_months_target:.0f}", "Months", "Target uncommitted liquidity buffer"])
    writer.writerow(["Marginal Income Tax Slab", f"{investor_obj.marginal_tax_slab_pct*100:.1f}", "%", "Applicable individual tax bracket"])
    writer.writerow(["Active Tax Regime", investor_obj.tax_regime.upper(), "Regime", "New (Sec 115BAC loss set-off blocked) vs Old (₹2L Set-Off)"])

    # ---------------------------------------------------------
    # 2. PROPERTY SPECIFICATIONS & MICRO-MARKET DEFAULTS
    # ---------------------------------------------------------
    write_section_header("SECTION 2: PROPERTY ASSUMPTIONS & MICRO-MARKET DEFAULTS")
    writer.writerow(["Parameter", "Value", "Unit / Basis", "Source & Modeling Notes"])
    writer.writerow(["Agreement Value", f"{property_obj.agreement_value_inr:,.0f}", "INR", "Base transacted purchase consideration"])
    writer.writerow(["Carpet Area", f"{property_obj.carpet_area_sqft:,.1f}", "sq.ft.", "Net usable internal floor area under RERA"])
    writer.writerow(["Expected Super Built-Up Area (SBA)", f"{property_obj.expected_sba_sqft:,.1f}", "sq.ft.", "Gross built area including common loading"])
    writer.writerow(["Loading Factor Used", f"{property_obj.expected_sba_sqft / property_obj.carpet_area_sqft:.2f}" if property_obj.carpet_area_sqft > 0 else "N/A", "Multiplier", "SBA divided by Carpet area"])
    writer.writerow(["Monthly Rent (Year 1)", f"{property_obj.monthly_rent_inr:,.0f}", "INR / month", "Expected starting rental income"])
    writer.writerow(["Baseline Annual Appreciation Rate", f"{property_obj.appreciation_rate*100:.2f}", "% / year", "Locality median appreciation CAGR anchor"])
    writer.writerow(["Primary Ready Reckoner Rate", f"{property_obj.rr_rate_primary_sqft:,.0f}", "INR / sq.ft.", "Government statutory minimum valuation floor"])
    writer.writerow(["Quarters to Sell (Inventory Overhang)", f"{property_obj.quarters_to_sell:.1f}", "Quarters", "Knight Frank H1 2026 inventory overhang study"])
    writer.writerow(["Liquidity Tier", property_obj.liquidity_tier.upper(), "Tier", "High (<3 Qtrs), Medium (3-6 Qtrs), Low (>6 Qtrs)"])
    writer.writerow(["Stress Exit Discount Haircut", f"{property_obj.exit_discount_stress_pct*100:.1f}", "%", "Calibrated forced liquidation haircut"])
    writer.writerow(["Under-Construction Delay Risk", f"{property_obj.avg_uc_delay_months:.0f}", "Months", "Calibrated possession delay stress assumption"])
    writer.writerow(["Tenant Vacancy Downtime (Best)", f"{property_obj.vacancy_months_best:.1f}", "Months / year", "Low vacancy turnover scenario"])
    writer.writerow(["Tenant Vacancy Downtime (Base)", f"{property_obj.vacancy_months_avg:.1f}", "Months / year", "Baseline vacancy turnover scenario"])
    writer.writerow(["Tenant Vacancy Downtime (Worst)", f"{property_obj.vacancy_months_worst:.1f}", "Months / year", "Stress vacancy turnover scenario"])

    # ---------------------------------------------------------
    # 3. MORTGAGE FINANCING & DEBT SERVICE
    # ---------------------------------------------------------
    write_section_header("SECTION 3: MORTGAGE FINANCING & DEBT STRUCTURE")
    writer.writerow(["Financing Metric", "Value", "Unit / Basis", "Regulatory & Risk Rules"])
    writer.writerow(["Desired LTV", f"{financing_obj.desired_ltv_pct*100:.1f}", "%", "User target loan-to-value ratio"])
    writer.writerow(["RBI Maximum Permitted LTV Cap", f"{loan.rbi_max_ltv*100:.1f}", "%", "RBI Master Circular statutory ceiling"])
    writer.writerow(["Effective LTV Applied", f"{loan.effective_ltv*100:.1f}", "%", "min(Desired LTV, RBI Cap)"])
    writer.writerow(["Total Mortgage Loan Amount", f"{loan.loan_amount:,.0f}", "INR", "Sanctioned borrowing principal"])
    writer.writerow(["Required Upfront Cash Equity", f"{loan.required_upfront_cash:,.0f}", "INR", "All-In Acquisition Cost minus Loan Amount"])
    writer.writerow(["Home Loan Interest Rate", f"{financing_obj.home_loan_interest_rate_pct*100:.2f}", "% / year", "Annual floating home loan borrowing rate"])
    writer.writerow(["Loan Tenure", f"{financing_obj.loan_tenure_years}", "Years", "Mortgage amortization duration (months = tenure * 12)"])
    writer.writerow(["Underwriting Holding Period", f"{financing_obj.holding_period_years}", "Years", "Target holding duration before exit"])
    writer.writerow(["Monthly Home Loan EMI", f"{loan.monthly_emi:,.0f}", "INR / month", "Equated Monthly Installment for debt service"])
    writer.writerow(["Monthly Pre-EMI Interest (if UC)", f"{loan.monthly_pre_emi:,.0f}", "INR / month", "Construction-phase interest (55% linear drawdown)"])
    foir = (investor_obj.existing_monthly_emis_inr + loan.monthly_emi) / investor_obj.net_monthly_income_inr * 100.0 if investor_obj.net_monthly_income_inr > 0 else 0.0
    post_cash = max(0.0, investor_obj.total_liquid_savings_inr - loan.required_upfront_cash)
    m_obligations = investor_obj.monthly_living_expenses_inr + investor_obj.existing_monthly_emis_inr + loan.monthly_emi
    buf_months = (post_cash / m_obligations) if m_obligations > 0 else 0.0
    writer.writerow(["Fixed Obligation to Income Ratio (FOIR)", f"{foir:.2f}", "%", "Target: <= 40% (Safe), 40-50% (Elevated), > 50% (Dangerous)"])
    writer.writerow(["Post-Purchase Liquid Reserves", f"{post_cash:,.0f}", "INR", "Remaining cash after down payment and closing costs"])
    writer.writerow(["Post-Purchase Emergency Buffer", f"{buf_months:.1f}", "Months", "Months of living & debt obligations covered by liquid reserves"])

    # ---------------------------------------------------------
    # 4. STATUTORY ACQUISITION COST WATERFALL
    # ---------------------------------------------------------
    write_section_header("SECTION 4: STATUTORY ACQUISITION COST WATERFALL")
    writer.writerow(["Acquisition Item", "Amount (INR)", "% of Base", "Statutory Floor & Legal Basis"])
    writer.writerow(["Agreement Value", f"{acq.agreement_value_inr:,.0f}", "100.00%", "Base contracted property consideration"])
    writer.writerow(["Government Ready Reckoner Value", f"{acq.rr_market_value:,.0f}", f"{acq.rr_market_value/acq.agreement_value_inr*100:.2f}%", "Carpet Area × IGR Maharashtra RR Rate"])
    writer.writerow(["Statutory Stamp Duty Base", f"{acq.stamp_duty_base:,.0f}", f"{acq.stamp_duty_base/acq.agreement_value_inr*100:.2f}%", "max(Agreement Value, Ready Reckoner Value)"])
    writer.writerow(["Effective Stamp Duty Rate", f"{acq.effective_stamp_duty_pct*100:.2f}", "%", "Jurisdiction base rate minus ownership rebate"])
    writer.writerow(["Stamp Duty Paid", f"{acq.stamp_duty_inr:,.0f}", f"{acq.stamp_duty_inr/acq.agreement_value_inr*100:.2f}%", "Maharashtra state revenue stamp duty"])
    writer.writerow(["Registration Fee", f"{acq.registration_fee_inr:,.0f}", f"{acq.registration_fee_inr/acq.agreement_value_inr*100:.2f}%", "1% of base, capped at ₹30,000 maximum"])
    writer.writerow(["Goods & Services Tax (GST)", f"{acq.gst_inr:,.0f}", f"{acq.gst_inr/acq.agreement_value_inr*100:.2f}%", f"{acq.gst_rate*100:.1f}% GST on Under-Construction (0% on Ready)"])
    writer.writerow(["Buy Brokerage (+18% GST)", f"{acq.buy_brokerage_inr:,.0f}", f"{acq.buy_brokerage_inr/acq.agreement_value_inr*100:.2f}%", "1.5% transaction commission + 18% GST"])
    writer.writerow(["Interiors & Upfront Fit-Outs", f"{acq.interiors_and_misc_inr:,.0f}", f"{acq.interiors_and_misc_inr/acq.agreement_value_inr*100:.2f}%", "Furnishing and initial improvement budget"])
    writer.writerow(["ALL-IN ACQUISITION COST", f"{acq.all_in_acquisition_cost:,.0f}", f"{acq.all_in_acquisition_cost/acq.agreement_value_inr*100:.2f}%", "Total all-inclusive cash capital outlay required"])
    writer.writerow(["TDS Status (Sec 194-IA / Sec 393)", "APPLICABLE" if acq.tds_applicable else "EXEMPT", "Threshold: ₹50L", "Mandatory 1% withholding from seller payout"])
    writer.writerow(["TDS Withholding Amount", f"{acq.tds_amount_inr:,.0f}", "1.00%", "Withheld from seller payment; NOT added to All-In cost"])

    # ---------------------------------------------------------
    # 5. 4-GATE DECISION ENGINE AUDIT
    # ---------------------------------------------------------
    write_section_header("SECTION 5: 4-GATE DECISION ENGINE AUDIT & MASTER VERDICT")
    writer.writerow(["Gate Number", "Gate Name", "Status", "Audit Details & Decision Rationale"])
    for gate in verdict.gates:
        writer.writerow([f"Gate {gate.gate_number}", gate.name, gate.status, " | ".join(gate.reasons)])
    writer.writerow(["FINAL VERDICT", "Master Underwriting Decision", verdict.status, verdict.summary])

    # ---------------------------------------------------------
    # 6. YEAR-BY-YEAR OPERATING CASH FLOW & AMORTIZATION SCHEDULE
    # ---------------------------------------------------------
    write_section_header("SECTION 6: YEAR-BY-YEAR OPERATING CASH FLOW & AMORTIZATION SCHEDULE")
    writer.writerow([
        "Year", "Gross Annual Rent", "Vacancy Loss", "Effective Gross Income (EGI)",
        "Operating Expenses (Opex)", "Net Operating Income (NOI)", "Annual Debt Service (EMI)",
        "Interest Paid", "Principal Repaid", "Net Annual Value (NAV)", "Std Deduction (30%)",
        "Net HP Income", "Tax Impact", "Net Cash Flow", "Closing Loan Balance", "Net Equity"
    ])
    for row in base_res.annual_cash_flows:
        writer.writerow([
            f"Year {row.year}",
            f"{row.gross_annual_rent:,.2f}",
            f"{row.vacancy_loss:,.2f}",
            f"{row.egi:,.2f}",
            f"{row.opex:,.2f}",
            f"{row.noi:,.2f}",
            f"{row.annual_emi:,.2f}",
            f"{row.interest_paid:,.2f}",
            f"{row.principal_paid:,.2f}",
            f"{row.nav:,.2f}",
            f"{row.std_deduction:,.2f}",
            f"{row.net_hp_income:,.2f}",
            f"{row.tax_impact:,.2f}",
            f"{row.net_cash_flow:,.2f}",
            f"{row.remaining_loan_balance:,.2f}",
            f"{row.net_equity:,.2f}",
        ])

    # ---------------------------------------------------------
    # 7. MULTI-SCENARIO UNDERWRITING MATRIX
    # ---------------------------------------------------------
    write_section_header("SECTION 7: MULTI-SCENARIO UNDERWRITING MATRIX")
    writer.writerow(["Metric", "Best Case", "Base Case", "Worst Case (Stress)"])
    writer.writerow(["Appreciation Rate (CAGR)", f"{property_obj.appreciation_rate*100 + 1.5:.2f}%", f"{property_obj.appreciation_rate*100:.2f}%", f"{max(0.0, property_obj.appreciation_rate*100 - 2.5):.2f}%"])
    writer.writerow(["Tenant Vacancy Downtime", f"{property_obj.vacancy_months_best:.1f} Months", f"{property_obj.vacancy_months_avg:.1f} Months", f"{property_obj.vacancy_months_worst:.1f} Months"])
    writer.writerow(["Mortgage Loan Rate", f"{matrix.best.scenario_loan_rate*100:.2f}%", f"{matrix.base.scenario_loan_rate*100:.2f}%", f"{matrix.worst.scenario_loan_rate*100:.2f}%"])
    writer.writerow(["Exit Haircut Discount", "0.0%", "0.0%", f"{property_obj.exit_discount_stress_pct*100:.1f}%"])
    uc_delay_best = "N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else "0 Months"
    uc_delay_base = "N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else f"{property_obj.avg_uc_delay_months:.0f} Months"
    uc_delay_worst = "N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else f"{int(round(property_obj.avg_uc_delay_months * 1.5))} Months"
    writer.writerow(["Under-Construction Delay", uc_delay_best, uc_delay_base, uc_delay_worst])
    b_irr = f"{matrix.best.equity_irr_pct:.2f}%" if matrix.best.equity_irr_pct is not None else "N/A"
    base_irr = f"{matrix.base.equity_irr_pct:.2f}%" if matrix.base.equity_irr_pct is not None else "N/A"
    w_irr = f"{matrix.worst.equity_irr_pct:.2f}%" if matrix.worst.equity_irr_pct is not None else "N/A"
    writer.writerow(["Equity IRR (Annualized)", b_irr, base_irr, w_irr])
    writer.writerow(["Net Terminal Exit Proceeds", f"{matrix.best.exit.net_exit_proceeds:,.0f}", f"{matrix.base.exit.net_exit_proceeds:,.0f}", f"{matrix.worst.exit.net_exit_proceeds:,.0f}"])
    writer.writerow(["Net Present Value (NPV @ FD Rate)", f"{matrix.best.npv_inr:,.0f}", f"{matrix.base.npv_inr:,.0f}", f"{matrix.worst.npv_inr:,.0f}"])

    # ---------------------------------------------------------
    # 8. DEDICATED STRESS TESTS AUDIT
    # ---------------------------------------------------------
    write_section_header("SECTION 8: DEDICATED STRESS TESTS AUDIT")
    writer.writerow(["Stress Test Module", "Status", "Stressed Metrics", "Stress Impact & Notes"])
    s1 = matrix.stress_interest_rate_shock
    s2 = matrix.stress_vacancy_shock
    s3 = matrix.stress_forced_exit_year_3
    writer.writerow([s1.name, s1.status, f"Stressed FOIR: {s1.metrics['stressed_foir_200bps']:.1f}% | +200 bps EMI: ₹{s1.metrics['emi_plus_200bps']:,.0f}", " | ".join(s1.notes)])
    writer.writerow([s2.name, s2.status, f"Cash Drain: ₹{s2.metrics['vacancy_cash_drain']:,.0f} | Buffer: {s2.metrics['remaining_buffer_months']:.1f} Mo", " | ".join(s2.notes)])
    writer.writerow([s3.name, s3.status, f"Recovery Ratio: {s3.metrics['capital_recovery_pct']:.1f}% | Recovered: ₹{s3.metrics['net_recovered_3yr']:,.0f}", " | ".join(s3.notes)])

    # ---------------------------------------------------------
    # 9. OPPORTUNITY COST & BENCHMARK PARITY (CHART DATA)
    # ---------------------------------------------------------
    write_section_header("SECTION 9: OPPORTUNITY COST & BENCHMARK PARITY (CHART DATA)")
    writer.writerow([
        "Holding Year", "Real Estate Base Net Equity", "Real Estate Worst Net Equity",
        "SBI Fixed Deposit (Post-Tax Parity)", "Nifty 50 TRI (Post-Tax Parity)"
    ])
    re_base_traj = [base_res.loan.required_upfront_cash] + [cf.net_equity for cf in base_res.annual_cash_flows]
    re_worst_traj = [matrix.worst.loan.required_upfront_cash] + [cf.net_equity for cf in matrix.worst.annual_cash_flows]
    fd_traj = base_res.fd_benchmark.trajectory
    nifty_traj = base_res.nifty_benchmark.trajectory

    for yr in range(len(re_base_traj)):
        writer.writerow([
            f"Year {yr}",
            f"{re_base_traj[yr]:,.2f}" if yr < len(re_base_traj) else "N/A",
            f"{re_worst_traj[yr]:,.2f}" if yr < len(re_worst_traj) else "N/A",
            f"{fd_traj[yr]:,.2f}" if yr < len(fd_traj) else "N/A",
            f"{nifty_traj[yr]:,.2f}" if yr < len(nifty_traj) else "N/A",
        ])

    writer.writerow([])
    writer.writerow(["Benchmark Summary Comparison", "Real Estate Base", "SBI Fixed Deposit", "Nifty 50 TRI"])
    writer.writerow(["Annualized Compounding Rate", base_irr, f"{base_res.fd_benchmark.annual_return_pct:.2f}%", f"{base_res.nifty_benchmark.annual_return_pct:.2f}%"])
    writer.writerow(["Terminal Post-Tax Wealth", f"{base_res.exit.net_exit_proceeds:,.0f}", f"{base_res.fd_benchmark.terminal_wealth_post_tax:,.0f}", f"{base_res.nifty_benchmark.terminal_wealth_post_tax:,.0f}"])
    writer.writerow(["Net Wealth Advantage vs Real Estate", "Baseline", f"{base_res.fd_benchmark.net_wealth_advantage_inr:,.0f}", f"{base_res.nifty_benchmark.net_wealth_advantage_inr:,.0f}"])
    writer.writerow(["IRR Spread vs Real Estate", "Baseline", f"{base_res.fd_benchmark.irr_spread_pct:+.2f}%", f"{base_res.nifty_benchmark.irr_spread_pct:+.2f}%"])

    # ---------------------------------------------------------
    # 10. VALUE DRIVERS, SENSITIVITY & EXPECTED VALUE
    # ---------------------------------------------------------
    write_section_header("SECTION 10: SENSITIVITY, BREAK-EVEN & EXPECTED VALUE")
    be = compute_break_even_appreciation(property_obj, financing_obj, investor_obj, macro)
    ev = compute_probability_weighted(property_obj, financing_obj, investor_obj, macro, 0.25, 0.50, 0.25)
    decomp = compute_return_decomposition(base_res)

    writer.writerow(["Metric", "Value", "Unit / Basis", "Analytical Meaning"])
    be_fd_txt = f"{be['break_even_vs_fd_pct']:.2f}%" if be['break_even_vs_fd_pct'] is not None else "> +30%"
    be_nifty_txt = f"{be['break_even_vs_nifty_pct']:.2f}%" if be['break_even_vs_nifty_pct'] is not None else "> +30%"
    writer.writerow(["Break-Even Appreciation vs Post-Tax SBI FD", be_fd_txt, "% / year", f"Minimum appreciation required to match post-tax SBI FD ({be['fd_post_tax_pct']:.2f}%)"])
    writer.writerow(["Break-Even Appreciation vs Nifty 50 TRI", be_nifty_txt, "% / year", f"Minimum appreciation required to match Nifty 50 TRI ({be['nifty_pct']:.2f}%)"])
    writer.writerow(["Locality Baseline Expected Appreciation", f"{property_obj.appreciation_rate*100:.2f}%", "% / year", "Used as base case projection anchor"])
    ev_irr_txt = f"{ev.expected_equity_irr_pct:.2f}%" if ev.expected_equity_irr_pct is not None else "N/A"
    writer.writerow(["Probability-Weighted Expected Terminal Wealth", f"{ev.expected_terminal_wealth:,.0f}", "INR", "Consolidated 25% Best / 50% Base / 25% Worst expectation"])
    writer.writerow(["Probability-Weighted Expected Equity IRR", ev_irr_txt, "% / year", "Expected blended equity return"])
    writer.writerow(["Probability of Beating Post-Tax SBI FD", f"{ev.p_beat_fd*100:.0f}%", "%", "Likelihood that real estate beats post-tax fixed deposit"])
    writer.writerow(["Probability of Beating Nifty 50 TRI", f"{ev.p_beat_nifty*100:.0f}%", "%", "Likelihood that real estate beats equity index fund"])
    total_cap = decomp.get("total_capital_invested", decomp.get("total_equity_invested", 0.0))
    net_noi = decomp.get("cumulative_noi", decomp.get("total_rent_collected", 0.0))
    net_cg = decomp.get("net_post_tax_capital_gain", decomp.get("capital_gain", 0.0))
    net_w = decomp.get("net_wealth_created", base_res.exit.net_exit_proceeds - total_cap)
    writer.writerow(["Return Decomposition: Total Capital Invested", f"{total_cap:,.0f}", "INR", "Initial down payment + cumulative operating cash top-ups"])
    writer.writerow(["Return Decomposition: Cumulative NOI", f"{net_noi:,.0f}", "INR", "Cumulative net operational rental income received"])
    writer.writerow(["Return Decomposition: Post-Tax Capital Gain", f"{net_cg:,.0f}", "INR", "Net capital gain after 12.5% LTCG tax upon exit"])
    writer.writerow(["Return Decomposition: Net Wealth Created", f"{net_w:,.0f}", "INR", "Exact additive reconciliation (Post-Tax Gain + NOI - Loan Interest - Operating Tax)"])

    return output.getvalue()
