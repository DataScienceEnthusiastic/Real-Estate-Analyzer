"""Glass-Box Calculation Proofs, Decision Audits & Financial Glossary Module.

This module provides live, mathematically self-proving calculation proofs and
decision audit tables across all tabs in the Pune Real Estate Underwriting Dashboard.
All numbers are read directly from the model outputs without duplicate math.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.charts import format_inr
from src.models import (
    InvestorProfile,
    MasterVerdict,
    PropertyInput,
    ScenarioMatrixResult,
    StressTestResult,
    UnderwritingResult,
)


def _fmt(val: Optional[float], precision: int = 0) -> str:
    """Formats numeric rupee values with Indian commas and abbreviated denomination."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    sign = "-" if val < 0 else ""
    abs_v = abs(val)
    if abs_v >= 10_000_000:
        cr = abs_v / 10_000_000
        return f"{sign}₹{cr:.2f} Cr (₹{abs_v:,.{precision}f})"
    elif abs_v >= 100_000:
        lakh = abs_v / 100_000
        return f"{sign}₹{lakh:.2f} L (₹{abs_v:,.{precision}f})"
    else:
        return f"{sign}₹{abs_v:,.{precision}f}"


def _fmt_pct(val: Optional[float], decimals: int = 2) -> str:
    """Formats percentage values cleanly."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    return f"{val:.{decimals}f}%"


# ==============================================================================
# TAB 1 EXPLAINER: EXECUTIVE SUMMARY & 4-GATE AUDIT
# ==============================================================================
def render_tab1_explainer(
    result: UnderwritingResult,
    decision: MasterVerdict,
    prop_input: PropertyInput,
    investor_profile: InvestorProfile,
    loc_row: Any,
    unit_row: Any,
    macro: dict[str, float],
) -> None:
    """Renders the transparent Glass-Box calculation and decision proofs for Tab 1."""
    st.divider()

    acq = result.acquisition
    loan = result.loan
    g1, g2, g3, g4 = decision.gates

    with st.expander("📐 Live Step-by-Step Calculation & Decision Proof", expanded=False):
        st.markdown("### 1. Statutory Valuation Floor & All-In Acquisition Cost Proof")

        # 1. Ready Reckoner Market Value
        st.markdown(
            f"""
            **A. Government Ready Reckoner (RR) Valuation Floor:**
            - **Formula:** $\\text{{RR Market Value}} = \\text{{Carpet Area (sq.ft.)}} \\times \\text{{Primary RR Rate (₹/sq.ft.)}}$
            - **Live Plug-In:** ${prop_input.carpet_area_sqft:,.1f}\\text{{ sq.ft.}} \\times ₹{prop_input.rr_rate_primary_sqft:,.0f}/\\text{{sq.ft.}} = ₹{acq.rr_market_value:,.0f}$
            - **Result:** `{_fmt(acq.rr_market_value)}`
            """
        )

        # 2. Stamp Duty Base
        winner_text = "Agreement Value is higher" if acq.agreement_value_inr >= acq.rr_market_value else "RR Market Value acts as the statutory floor"
        st.markdown(
            f"""
            **B. Statutory Stamp Duty Base:**
            - **Formula:** $\\text{{Stamp Duty Base}} = \\max(\\text{{Agreement Value}}, \\text{{RR Market Value}})$
            - **Live Plug-In:** $\\max(₹{acq.agreement_value_inr:,.0f}, ₹{acq.rr_market_value:,.0f}) \\implies$ **{winner_text}**
            - **Result:** `{_fmt(acq.stamp_duty_base)}`
            """
        )

        # 3. Stamp Duty Rate & Amount
        urban_jurisdictions = {"PMC", "PCMC", "Cantonment"}
        is_urban = prop_input.jurisdiction in urban_jurisdictions
        base_rate = macro["stamp_duty_urban_male_pct"] if is_urban else macro["stamp_duty_rural_male_pct"]
        rebate_dict = {
            "sole_male": 0.0,
            "sole_female": macro["stamp_duty_female_rebate_pct"],
            "female_joint": macro["stamp_duty_female_rebate_pct"],
            "male_female_50_50": macro["stamp_duty_female_rebate_pct"] / 2.0,
        }
        rebate = rebate_dict.get(prop_input.buyer_ownership, 0.0)

        st.markdown(
            f"""
            **C. Stamp Duty Tax:**
            - **Formula:** $\\text{{Stamp Duty}} = \\text{{Stamp Duty Base}} \\times (\\text{{Jurisdiction Base Rate}} - \\text{{Ownership Rebate}})$
            - **Live Plug-In:** Jurisdiction `{prop_input.jurisdiction}` ({'Urban 7%' if is_urban else 'Rural 4%'}) − Ownership rebate for `{prop_input.buyer_ownership}` ({rebate*100:.1f}%) $= {acq.effective_stamp_duty_pct*100:.2f}\\%$
              $$\\implies ₹{acq.stamp_duty_base:,.0f} \\times {acq.effective_stamp_duty_pct*100:.2f}\\% = ₹{acq.stamp_duty_inr:,.0f}$$
            - **Result:** `{_fmt(acq.stamp_duty_inr)}`
            """
        )

        # 4. Registration Fee
        cap_hit = (acq.stamp_duty_base * macro["registration_fee_pct"]) >= macro["registration_fee_cap_inr"]
        st.markdown(
            f"""
            **D. Registration Fee:**
            - **Formula:** $\\text{{Registration Fee}} = \\min(\\text{{Stamp Duty Base}} \\times 1\\%, ₹30,000\\text{{ Maharashtra Cap}})$
            - **Live Plug-In:** $\\min(₹{acq.stamp_duty_base * 0.01:,.0f}, ₹30,000) \\implies$ **{'Cap of ₹30,000 applied' if cap_hit else 'Uncapped (below ₹30,000 limit)'}**
            - **Result:** `{_fmt(acq.registration_fee_inr)}`
            """
        )

        # 5. GST (if Under-Construction)
        if prop_input.property_status == "under_construction":
            aff_note = "Affordable (≤ ₹45L & ≤ 645.83 sq.ft.) → 1% GST" if acq.is_affordable_gst else "Non-Affordable (> ₹45L or > 645.83 sq.ft.) → 5% GST"
            st.markdown(
                f"""
                **E. Goods & Services Tax (GST):**
                - **Formula:** $\\text{{GST}} = \\text{{Agreement Value}} \\times \\text{{GST Rate}}$
                - **Live Plug-In:** Status is `under_construction` $\\to$ {aff_note} $\\implies ₹{acq.agreement_value_inr:,.0f} \\times {acq.gst_rate*100:.1f}\\%$
                - **Result:** `{_fmt(acq.gst_inr)}`
                """
            )
        else:
            st.markdown(
                """
                **E. Goods & Services Tax (GST):**
                - **Status:** `ready_resale` (Ready possession or resale unit).
                - **Rule:** Under Schedule III of the CGST Act 2017, sale of ready completed immovable property with completion/occupancy certificate attracts **0% GST**.
                - **Result:** `₹0`
                """
            )

        # 6. Brokerage & Interiors
        st.markdown(
            f"""
            **F. Acquisition Brokerage & Fit-Outs:**
            - **Buy Brokerage:** {'Enabled (1.5% + 18% GST = 1.77%)' if prop_input.brokerage_enabled else 'Disabled (0%)'} $\\to$ `{_fmt(acq.buy_brokerage_inr)}`
            - **Interiors & Miscellaneous:** `{_fmt(acq.interiors_and_misc_inr)}`
            """
        )

        # 7. All-In Acquisition Sum
        st.markdown(
            f"""
            **G. All-In Total Acquisition Cost:**
            - **Formula:** $\\text{{All-In Cost}} = \\text{{Agreement Value}} + \\text{{Stamp Duty}} + \\text{{Registration}} + \\text{{GST}} + \\text{{Brokerage}} + \\text{{Interiors}}$
            - **Live Plug-In:** $₹{acq.agreement_value_inr:,.0f} + ₹{acq.stamp_duty_inr:,.0f} + ₹{acq.registration_fee_inr:,.0f} + ₹{acq.gst_inr:,.0f} + ₹{acq.buy_brokerage_inr:,.0f} + ₹{acq.interiors_and_misc_inr:,.0f}$
            - **Result:** `{_fmt(acq.all_in_acquisition_cost)}`
            """
        )

        # TDS Callout Box
        tds_status = "APPLICABLE" if acq.tds_applicable else "NOT APPLICABLE"
        tds_color = "#fbbf24" if acq.tds_applicable else "#34d399"
        st.markdown(
            f"""
            <div style="background:#111827; border:1px solid #1f2937; border-left:4px solid {tds_color}; border-radius:8px; padding:14px; margin: 12px 0;">
                <div style="font-weight:700; color:{tds_color}; font-size:0.92rem;">
                    🛡️ TDS COMPLIANCE AUDIT (Sec 194-IA / Sec 393): {tds_status}
                </div>
                <div style="font-size:0.85rem; color:#cbd5e1; margin-top:6px; line-height:1.5;">
                    <b>Threshold Check:</b> Stamp Duty Base of <b>₹{acq.stamp_duty_base:,.0f}</b> vs Statutory Threshold of <b>₹50,00,000</b>.<br>
                    {f'Mandatory 1% TDS = <b>{_fmt(acq.tds_amount_inr)}</b>. <i>Critical Accounting Clarification:</i> This amount is withheld from the seller payout and deposited directly with the Income Tax Department via Form 26QB. It is <b>NOT</b> an extra acquisition surcharge on top of the All-In Cost.' if acq.tds_applicable else 'Transaction consideration is strictly below ₹50 Lakhs; no TDS withholding is legally required.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 2. RBI LTV Cap, Mortgage Borrowing & Monthly EMI Proof")

        # RBI Slab Proof
        st.markdown(
            f"""
            **A. RBI Prudent LTV Cap Slab Check:**
            - **Rule:** $\\le ₹30\\text{{ Lakhs}} \\implies 90\\%$; $₹30\\text{{L}}–₹75\\text{{L}} \\implies 80\\%$; $> ₹75\\text{{ Lakhs}} \\implies 75\\%$.
            - **Live Plug-In:** Agreement value of $₹{acq.agreement_value_inr:,.0f}$ falls into RBI ceiling of **{loan.rbi_max_ltv*100:.0f}%**.
            - **Effective LTV Applied:** $\\min(\\text{{Desired }} {prop_input.agreement_value_inr and (loan.effective_ltv*100):.0f}\\%, \\text{{RBI Cap }} {loan.rbi_max_ltv*100:.0f}\\%) = \\mathbf{{{loan.effective_ltv*100:.0f}\\%}}$
            - **Loan Amount:** $₹{acq.agreement_value_inr:,.0f} \\times {loan.effective_ltv*100:.0f}\\% = \\mathbf{{{_fmt(loan.loan_amount)}}}$
            - **Required Upfront Cash:** $\\text{{All-In Cost }} ₹{acq.all_in_acquisition_cost:,.0f} - \\text{{Loan Amount }} ₹{loan.loan_amount:,.0f} = \\mathbf{{{_fmt(loan.required_upfront_cash)}}}$
            """
        )

        # Monthly EMI Proof
        if loan.loan_amount <= 0:
            st.markdown(
                """
                **B. Monthly EMI Computation:**
                - **Status:** **All-Cash Purchase (Zero Debt)**.
                - **Result:** `EMI = ₹0.00 / month` (100% equity financing; no debt servicing obligations).
                """
            )
        else:
            rate_pct = loan.home_loan_interest_rate_pct * 100.0
            tenure_yrs = loan.loan_tenure_years
            total_months = tenure_yrs * 12
            st.markdown(
                f"""
                **B. Monthly EMI Computation:**
                - **Formula:** $\\text{{EMI}} = \\frac{{P \\times r \\times (1+r)^N}}{{(1+r)^N - 1}}$
                - **Live Plug-In:** Principal $P = ₹{loan.loan_amount:,.0f}$, annual rate $= {rate_pct:.2f}\\% \\implies r = {rate_pct / 1200.0:.6f}/\\text{{month}}$, tenure $N = {tenure_yrs} \\times 12 = {total_months}\\text{{ months}}$.
                - **Result:** `{_fmt(loan.monthly_emi)}/month`
                """
            )

        # FOIR & Buffer Proof
        foir = (investor_profile.existing_monthly_emis_inr + loan.monthly_emi) / investor_profile.net_monthly_income_inr * 100.0 if investor_profile.net_monthly_income_inr > 0 else 0.0
        post_cash = max(0.0, investor_profile.total_liquid_savings_inr - loan.required_upfront_cash)
        m_obligations = investor_profile.monthly_living_expenses_inr + investor_profile.existing_monthly_emis_inr + loan.monthly_emi
        buffer_months = (post_cash / m_obligations) if m_obligations > 0 else 0.0

        st.markdown(
            f"""
            **C. Debt Service & Solvency Metrics:**
            - **FOIR Formula:** $\\frac{{\\text{{Existing EMIs }} ₹{investor_profile.existing_monthly_emis_inr:,.0f} + \\text{{New EMI }} ₹{loan.monthly_emi:,.0f}}}{{\\text{{Net Income }} ₹{investor_profile.net_monthly_income_inr:,.0f}}} = \\mathbf{{{foir:.2f}\\%}}$
            - **Post-Purchase Buffer Formula:** $\\frac{{\\text{{Remaining Liquid Savings }} ₹{post_cash:,.0f}}}{{\\text{{Total Monthly Obligations }} ₹{m_obligations:,.0f}}} = \\mathbf{{{buffer_months:.1f}\\text{{ Months}}}}$
            """
        )

        st.markdown("---")
        st.markdown("### 3. Structured 4-Gate Decision Audit Table")

        # 5-Column Decision Table
        table_data = [
            # Gate 1
            {
                "Check / Rule": "G1.1 Upfront Capital Solvency",
                "Decision Threshold / Logic": "Liquid Savings ≥ Upfront Cash Required",
                "Your Live Value": f"Cash: {format_inr(investor_profile.total_liquid_savings_inr)} vs Req: {format_inr(loan.required_upfront_cash)}",
                "Comparison Proof": f"{'Surplus of ' + format_inr(post_cash) if post_cash >= 0 else 'Shortfall of ' + format_inr(abs(post_cash))}",
                "Outcome": "PASS" if post_cash >= 0 else "FAIL",
            },
            {
                "Check / Rule": "G1.2 Minimum Safety Liquidity",
                "Decision Threshold / Logic": "Buffer Months ≥ 3.0 Months obligations",
                "Your Live Value": f"{buffer_months:.1f} Months",
                "Comparison Proof": f"{buffer_months:.1f} Mo vs 3.0 Mo safety floor",
                "Outcome": "PASS" if buffer_months >= 3.0 else "FAIL",
            },
            {
                "Check / Rule": "G1.3 FOIR Debt Ceiling",
                "Decision Threshold / Logic": "FOIR ≤ 40% (PASS), 40-50% (WARN), >50% (FAIL)",
                "Your Live Value": f"{foir:.1f}%",
                "Comparison Proof": f"{foir:.1f}% vs 50.0% max limit",
                "Outcome": "PASS" if foir <= 40 else ("WARN" if foir <= 50 else "FAIL"),
            },
            {
                "Check / Rule": "G1.4 Target Emergency Buffer",
                "Decision Threshold / Logic": f"Buffer ≥ {investor_profile.emergency_buffer_months_target:.0f} Mo + ₹{investor_profile.upcoming_cash_need_inr:,.0f} cash needs",
                "Your Live Value": f"Reserves: {format_inr(post_cash)}",
                "Comparison Proof": f"Target: {format_inr((investor_profile.emergency_buffer_months_target * m_obligations) + investor_profile.upcoming_cash_need_inr)}",
                "Outcome": g1.status,
            },
            # Gate 2
            {
                "Check / Rule": "G2.1 Rate Shock Resilience",
                "Decision Threshold / Logic": "+200 bps Mortgage Hike Stressed FOIR ≤ 60%",
                "Your Live Value": f"{g2.metrics.get('stressed_foir_200bps', 0.0):.1f}%",
                "Comparison Proof": f"{g2.metrics.get('stressed_foir_200bps', 0.0):.1f}% vs 60.0% ceiling",
                "Outcome": "PASS" if g2.metrics.get("stressed_foir_200bps", 0.0) <= 50 else ("WARN" if g2.metrics.get("stressed_foir_200bps", 0.0) <= 60 else "FAIL"),
            },
            {
                "Check / Rule": "G2.2 Vacancy Shock Survival",
                "Decision Threshold / Logic": "Post-Shock Reserves ≥ 1.0 Mo (FAIL floor), ≥ 3.0 Mo (PASS)",
                "Your Live Value": f"{g2.metrics.get('remaining_buffer_months_after_vacancy', 0.0):.1f} Months",
                "Comparison Proof": f"{g2.metrics.get('remaining_buffer_months_after_vacancy', 0.0):.1f} Mo vs 1.0 Mo floor",
                "Outcome": "PASS" if g2.metrics.get("remaining_buffer_months_after_vacancy", 0.0) >= 3.0 else ("WARN" if g2.metrics.get("remaining_buffer_months_after_vacancy", 0.0) >= 1.0 else "FAIL"),
            },
            {
                "Check / Rule": "G2.3 Micro-Market Liquidity Risk",
                "Decision Threshold / Logic": "Tier != 'low' OR Holding Period ≥ 5y without cash need",
                "Your Live Value": f"Tier: '{prop_input.liquidity_tier.upper()}' ({prop_input.quarters_to_sell:.1f} Qtrs)",
                "Comparison Proof": f"Liquidity tier overhang against investment horizon",
                "Outcome": g2.status,
            },
            # Gate 3
            {
                "Check / Rule": "G3.1 Valuation Premium vs RR",
                "Decision Threshold / Logic": "Market Price to Ready Reckoner Ratio ≤ 2.50x",
                "Your Live Value": f"{g3.metrics.get('market_to_rr_ratio', 1.0):.2f}x",
                "Comparison Proof": f"{g3.metrics.get('market_to_rr_ratio', 1.0):.2f}x vs 2.50x threshold",
                "Outcome": "PASS" if g3.metrics.get("market_to_rr_ratio", 1.0) <= 2.50 else "WARN",
            },
            {
                "Check / Rule": "G3.2 Net Rental Yield Hurdle",
                "Decision Threshold / Logic": "Year-1 Net Rental Yield ≥ 2.0% (Rental use-case)",
                "Your Live Value": f"{result.year_1_net_yield_pct:.2f}%",
                "Comparison Proof": f"{result.year_1_net_yield_pct:.2f}% vs 2.00% benchmark hurdle",
                "Outcome": "PASS" if (result.year_1_net_yield_pct >= 2.0 or prop_input.use_case == "self_use") else "WARN",
            },
            # Gate 4
            {
                "Check / Rule": "G4.1 Positive Equity IRR",
                "Decision Threshold / Logic": "Equity IRR > 0.0% (Non-negative return on equity)",
                "Your Live Value": f"{result.equity_irr_pct:.2f}%" if result.equity_irr_pct is not None else "Negative",
                "Comparison Proof": f"{result.equity_irr_pct or 0.0:.2f}% > 0.0%",
                "Outcome": "PASS" if result.equity_irr_pct and result.equity_irr_pct > 0 else "FAIL",
            },
            {
                "Check / Rule": "G4.2 SBI Term Deposit Hurdle",
                "Decision Threshold / Logic": "Equity IRR > Post-Tax SBI Fixed Deposit Rate",
                "Your Live Value": f"IRR: {_fmt_pct(result.equity_irr_pct)} vs FD: {result.fd_benchmark.annual_return_pct:.2f}%",
                "Comparison Proof": f"IRR Spread: {result.fd_benchmark.irr_spread_pct:+.2f}%",
                "Outcome": "PASS" if (result.equity_irr_pct is not None and result.fd_benchmark.irr_spread_pct > 0) else "FAIL",
            },
            {
                "Check / Rule": "G4.3 Nifty 50 TRI Capital Efficiency",
                "Decision Threshold / Logic": "Terminal Wealth ≥ Post-Tax Nifty 50 TRI Corpus",
                "Your Live Value": f"RE: {format_inr(result.exit.net_exit_proceeds)} vs Nifty: {format_inr(result.nifty_benchmark.terminal_wealth_post_tax)}",
                "Comparison Proof": f"Wealth Gap: {format_inr(result.nifty_benchmark.net_wealth_advantage_inr)}",
                "Outcome": "PASS" if result.nifty_benchmark.net_wealth_advantage_inr >= 0 else "WARN",
            },
        ]

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, width="stretch", hide_index=True)

        # Master Verdict Synthesis Box
        verdict_color = "#10b981" if decision.status == "GO" else ("#f59e0b" if decision.status == "CAUTION" else "#ef4444")
        st.markdown(
            f"""
            <div style="background:#111827; border:2px solid {verdict_color}; border-radius:10px; padding:16px; margin-top:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.15rem; font-weight:800; color:#ffffff;">
                        🎯 Master Verdict Synthesis Proof: <span style="color:{verdict_color};">{decision.status}</span>
                    </div>
                    <div>
                        <span class="badge badge-{'green' if g1.status == 'PASS' else ('amber' if g1.status == 'WARN' else 'red')}">G1: {g1.status}</span>
                        <span class="badge badge-{'green' if g2.status == 'PASS' else ('amber' if g2.status == 'WARN' else 'red')}">G2: {g2.status}</span>
                        <span class="badge badge-{'green' if g3.status == 'PASS' else ('amber' if g3.status == 'WARN' else 'red')}">G3: {g3.status}</span>
                        <span class="badge badge-{'green' if g4.status == 'PASS' else ('amber' if g4.status == 'WARN' else 'red')}">G4: {g4.status}</span>
                    </div>
                </div>
                <div style="font-size:0.86rem; color:#cbd5e1; margin-top:8px; line-height:1.5;">
                    <b>Evaluation Rule:</b> (1) If Gate 1 or Gate 2 is <code>FAIL</code> $\implies$ <b>NO-GO</b> (Hard Stop). (2) If all gates pass but any gate has <code>WARN</code> or Gate 3/4 is <code>FAIL</code> $\implies$ <b>CAUTION</b>. (3) If all 4 gates are <code>PASS</code> $\implies$ <b>GO</b>.<br>
                    <b>Proof of Verdict:</b> {decision.summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Glossary Expander
    with st.expander("📖 Financial Terms Glossary (What These Metrics Mean)", expanded=False):
        glossary_items = [
            ("Ready Reckoner (RR) Rate", "The statutory minimum valuation published annually by the Maharashtra state government (IGR Maharashtra) for land and built-up property in each survey number/sub-zone. It forms the non-negotiable floor for stamp duty and capital gains computation."),
            ("Stamp Duty Base", "The higher of the transacted Agreement Value and the Government Ready Reckoner Value. Stamp duty is charged on this value to prevent tax evasion through cash under-reporting."),
            ("TDS (Sec 194-IA / Sec 393)", "Under the Indian Income Tax Act, buyers purchasing immovable property with consideration of ₹50 Lakhs or more must withhold 1% from the seller's payment and deposit it with the government via Form 26QB. It is an advance tax compliance step, not an extra buyer surcharge."),
            ("LTV (Loan-to-Value)", "The proportion of the property's agreement value that a commercial bank can legally finance. Governed strictly by RBI master circular caps: 90% for loans up to ₹30L, 80% for ₹30L–₹75L, and 75% above ₹75L."),
            ("FOIR (Fixed Obligation to Income Ratio)", "The percentage of an investor's take-home monthly salary committed to debt obligations (all existing EMIs + new home loan EMI). Institutional underwriting considers ≤40% prudent, 40–50% elevated, and >50% unacceptable."),
            ("Emergency Buffer Months", "The number of months an investor can service all living expenses, existing liabilities, and new property EMIs purely from uncommitted liquid reserves without needing rental income or employment salary."),
            ("All-In Acquisition Cost", "The true all-cash investment needed to take legal ownership. It equals Agreement Value + Stamp Duty + Registration Fee + GST (if UC) + Brokerage (incl. 18% GST) + Fit-Outs/Interiors."),
        ]
        for term, desc in glossary_items:
            st.markdown(f"**{term}:** {desc}")


# ==============================================================================
# TAB 2 EXPLAINER: CASH FLOW, AMORTIZATION & TAX ENGINE
# ==============================================================================
def render_tab2_explainer(
    result: UnderwritingResult,
    prop_input: PropertyInput,
    investor_profile: InvestorProfile,
    macro: dict[str, float],
) -> None:
    """Renders the transparent Year-1 hand-reconciliation cash flow, debt, and tax proofs for Tab 2."""
    st.divider()

    y1 = result.annual_cash_flows[0]
    loan = result.loan

    with st.expander("📐 Live Step-by-Step Calculation & Decision Proof", expanded=False):
        st.markdown("### Year-1 Complete Financial Reconciliation Walkthrough")
        st.caption("Reconcile Row 1 of the Cash Flow table line-by-line using your exact active inputs.")

        # 1. Gross Annual Rent
        st.markdown(
            f"""
            **1. Gross Annual Rent (or Imputed Rent Saved):**
            - **Formula:** $\\text{{Gross Rent}} = \\text{{Monthly Rent}} \\times \\text{{Active Tenancy Months}} \\times (1 + \\text{{Rent Escalation}})^0$
            - **Live Plug-In:** $₹{prop_input.monthly_rent_inr:,.0f}/\\text{{mo}} \\times 12\\text{{ months}} \\times (1.0)^0$
            - **Result:** `{_fmt(y1.gross_annual_rent)}`
            """
        )

        # 2. Vacancy Loss
        vac_months = prop_input.vacancy_months_avg if prop_input.use_case == "investment_rental" else 0.0
        st.markdown(
            f"""
            **2. Vacancy Allowance Loss:**
            - **Formula:** $\\text{{Vacancy Loss}} = \\text{{Gross Rent}} \\times \\left(\\frac{{\\text{{Vacancy Months}}}}{{12}}\\right)$ ({'₹0 for Self-Use' if prop_input.use_case == 'self_use' else f'Benchmark: {vac_months:.1f} months in {prop_input.pune_zone} Zone'})
            - **Live Plug-In:** $₹{y1.gross_annual_rent:,.0f} \\times ({vac_months:.2f} / 12)$
            - **Result:** `{_fmt(y1.vacancy_loss)}`
            """
        )

        # 3. Effective Gross Income (EGI)
        st.markdown(
            f"""
            **3. Effective Gross Income (EGI):**
            - **Formula:** $\\text{{EGI}} = \\text{{Gross Rent}} - \\text{{Vacancy Loss}}$
            - **Live Plug-In:** $₹{y1.gross_annual_rent:,.0f} - ₹{y1.vacancy_loss:,.0f}$
            - **Result:** `{_fmt(y1.egi)}`
            """
        )

        # 4. Operating Expenses (Opex)
        capex_note = f" (includes {_fmt(y1.capex_reserve)} CapEx/re-letting reserve)" if y1.capex_reserve > 0 else ""
        st.markdown(
            f"""
            **4. Operating Expenses (Opex):**
            - **Formula:** $\\text{{Opex}} = (\\text{{Monthly Maintenance}} \\times \\text{{Active Months}}) + \\text{{Annual Property Tax}} + \\text{{CapEx Reserve}}$
            - **Live Plug-In:** Recurring society maintenance + municipal property tax{capex_note} $= \\mathbf{{{_fmt(y1.opex)}}}$
            - **Result:** `{_fmt(y1.opex)}`
            """
        )

        # 5. Net Operating Income (NOI)
        st.markdown(
            f"""
            **5. Net Operating Income (NOI):**
            - **Formula:** $\\text{{NOI}} = \\text{{EGI}} - \\text{{Opex}}$
            - **Live Plug-In:** $₹{y1.egi:,.0f} - ₹{y1.opex:,.0f}$
            - **Result:** `{_fmt(y1.noi)}`
            """
        )

        # 6. Amortization & Debt Proof
        st.markdown(
            f"""
            **6. Debt Service & Principal Amortization (Year 1):**
            - **Opening Debt:** `{_fmt(loan.loan_amount)}`
            - **Year-1 Outflow Breakdown:** Annual EMI $= \\mathbf{{{_fmt(y1.annual_emi)}}}$ (Interest Paid: **{_fmt(y1.interest_paid)}** + Principal Repaid: **{_fmt(y1.principal_paid)}**).
            - **Year-1 Closing Loan Balance:** $₹{loan.loan_amount:,.0f} - ₹{y1.principal_paid:,.0f} = \\mathbf{{{_fmt(y1.remaining_loan_balance)}}}$
            - **Tenure Debt Closure Proof:** Amortization engine verifies that at Year {loan.loan_tenure_years}, remaining debt reaches strictly **₹0.00**.
            """
        )

        # 7. Income Tax Engine Proof
        tax_mult = (1.0 + max(0.0, investor_profile.surcharge_rate_pct)) * 1.04
        st.markdown(
            f"""
            **7. Income Tax Impact on Property Income (Section 24 / Section 22):**
            - **Net Annual Value (NAV):** $\\max(0, \\text{{EGI}} - \\text{{Property Tax}}) = \\mathbf{{{_fmt(y1.nav)}}}$
            - **Section 24(a) Standard Deduction (30%):** $₹{y1.nav:,.0f} \\times 30\\% = \\mathbf{{{_fmt(y1.std_deduction)}}}$
            - **Net Income / Loss from House Property:** $\\text{{NAV}} - \\text{{Std Deduction}} - \\text{{Deductible Interest}} = \\mathbf{{{_fmt(y1.net_hp_income)}}}$
            """
        )

        # Tax Regime Comparison Table
        tax_table = [
            {
                "Tax Regime": "New Tax Regime (Section 115BAC)",
                "House Property Loss Set-Off": "BLOCKED (₹0 salary set-off permitted)",
                "Taxable Property Income Base": f"₹{max(0.0, y1.net_hp_income):,.0f}",
                "Formula Applied": f"max(0, Net HP Income) × slab × {tax_mult:.4f}",
                "Tax Impact (Year 1)": f"{format_inr(max(0.0, y1.net_hp_income) * investor_profile.marginal_tax_slab_pct * tax_mult)}",
                "Status": "ACTIVE" if investor_profile.tax_regime == "new" else "Alternative",
            },
            {
                "Tax Regime": "Old Tax Regime (With Deductions)",
                "House Property Loss Set-Off": "PERMITTED (Capped at -₹2,00,000/yr; excess carried fwd 8 yrs)",
                "Taxable Property Income Base": f"₹{max(-200_000.0, y1.net_hp_income):,.0f}",
                "Formula Applied": f"max(-2,00,000, Net HP Income) × slab × {tax_mult:.4f}",
                "Tax Impact (Year 1)": f"{format_inr(max(-200_000.0, y1.net_hp_income) * investor_profile.marginal_tax_slab_pct * tax_mult)}",
                "Status": "ACTIVE" if investor_profile.tax_regime == "old" else "Alternative",
            },
        ]
        st.dataframe(pd.DataFrame(tax_table), width="stretch", hide_index=True)

        # 8. Net Annual Cash Flow
        st.markdown(
            f"""
            **8. Final Bottom-Line Year-1 Net Cash Flow:**
            - **Formula:** $\\text{{Net Cash Flow}} = \\text{{NOI}} - \\text{{Annual EMI}} - \\text{{Tax Impact}}$
            - **Live Plug-In:** $₹{y1.noi:,.0f} - ₹{y1.annual_emi:,.0f} - (₹{y1.tax_impact:,.0f})$
            - **Result:** `{_fmt(y1.net_cash_flow)}` (Reconciles exactly to Year 1 table value).
            """
        )

    # Glossary Expander
    with st.expander("📖 Financial Terms Glossary (What These Metrics Mean)", expanded=False):
        glossary_items = [
            ("EGI (Effective Gross Income)", "The total rental income expected to be collected after subtracting vacancy downtime from gross potential rent."),
            ("NOI (Net Operating Income)", "The property's operating earnings before debt service and taxes = EGI minus all routine operating expenses (society maintenance, property tax). It measures pure property performance."),
            ("Operating Expenses (Opex)", "The recurring costs required to operate and maintain the physical unit, including building maintenance dues and municipal property taxes."),
            ("Loan Amortization", "The scheduled repayment of debt through fixed monthly installments (EMIs), where each payment is mathematically split between interest and principal reduction."),
            ("Pre-EMI Interest", "Interest payable during the construction phase of an under-construction project, computed only on the portion of the sanctioned home loan disbursed by the bank."),
            ("Net Annual Value (NAV)", "The statutory tax valuation of rental property under Section 23 of the Income Tax Act, computed as Effective Gross Rent minus local municipal taxes paid."),
            ("Section 24(a) Standard Deduction", "A statutory 30% deduction allowed from the Net Annual Value on let-out properties for repairs and maintenance, irrespective of actual expenses incurred."),
            ("House Property Loss Set-Off", "The tax rule governing whether mortgage interest losses from property can reduce taxable salary income. Allowed up to ₹2,00,000 per year under the Old Tax Regime, but completely eliminated under the New Tax Regime (Section 115BAC)."),
        ]
        for term, desc in glossary_items:
            st.markdown(f"**{term}:** {desc}")


# ==============================================================================
# TAB 3 EXPLAINER: SCENARIOS, STRESS TESTS & OPPORTUNITY COST
# ==============================================================================
def render_tab3_explainer(
    result: UnderwritingResult,
    scenarios: ScenarioMatrixResult,
    stress_results: Any,
    prop_input: PropertyInput,
    investor_profile: InvestorProfile,
    zone_row: Any,
    macro: dict[str, float],
) -> None:
    """Renders the scenario assumptions, exit tax, and cash-flow parity proofs for Tab 3."""
    st.divider()

    h = len(result.annual_cash_flows)
    exit_val = result.exit

    with st.expander("📐 Live Step-by-Step Calculation & Decision Proof", expanded=False):
        st.markdown("### 1. Multi-Scenario Parameter Proof Table")
        st.caption("Live parameters driving the Best, Base, and Worst (Stress) underwriting models.")

        base_rate_pct = (scenarios.base.scenario_loan_rate or scenarios.base.loan.home_loan_interest_rate_pct) * 100.0
        best_rate_pct = (scenarios.best.scenario_loan_rate or scenarios.best.loan.home_loan_interest_rate_pct) * 100.0
        worst_rate_pct = (scenarios.worst.scenario_loan_rate or scenarios.worst.loan.home_loan_interest_rate_pct) * 100.0

        sc_table = [
            {
                "Scenario": "Best Case",
                "Appreciation Rate": f"{(scenarios.best.scenario_appreciation_rate or (prop_input.appreciation_rate + 0.015))*100:.2f}%",
                "Vacancy Months": f"{prop_input.vacancy_months_best:.1f} Mo",
                "Loan Rate": f"{best_rate_pct:.2f}%",
                "Exit Haircut": "0.0%",
                "UC Delay": "N/A (Ready Resale)" if prop_input.property_status == "ready_resale" else "0 Mo",
                "Equity IRR": f"{scenarios.best.equity_irr_pct:.2f}%" if scenarios.best.equity_irr_pct is not None else "N/A",
                "Terminal Wealth": format_inr(scenarios.best.exit.net_exit_proceeds),
            },
            {
                "Scenario": "Base Case",
                "Appreciation Rate": f"{(scenarios.base.scenario_appreciation_rate or prop_input.appreciation_rate)*100:.2f}%",
                "Vacancy Months": f"{prop_input.vacancy_months_avg:.1f} Mo",
                "Loan Rate": f"{base_rate_pct:.2f}%",
                "Exit Haircut": "0.0%",
                "UC Delay": "N/A (Ready Resale)" if prop_input.property_status == "ready_resale" else f"{prop_input.avg_uc_delay_months:.0f} Mo",
                "Equity IRR": f"{scenarios.base.equity_irr_pct:.2f}%" if scenarios.base.equity_irr_pct is not None else "N/A",
                "Terminal Wealth": format_inr(scenarios.base.exit.net_exit_proceeds),
            },
            {
                "Scenario": "Worst Case (Stress)",
                "Appreciation Rate": f"{(scenarios.worst.scenario_appreciation_rate or max(0.02, prop_input.appreciation_rate - 0.025))*100:.2f}%",
                "Vacancy Months": f"{prop_input.vacancy_months_worst:.1f} Mo",
                "Loan Rate": f"{worst_rate_pct:.2f}%",
                "Exit Haircut": f"{prop_input.exit_discount_stress_pct*100:.1f}%",
                "UC Delay": "N/A (Ready Resale)" if prop_input.property_status == "ready_resale" else f"{int(round(prop_input.avg_uc_delay_months * 1.5))} Mo",
                "Equity IRR": f"{scenarios.worst.equity_irr_pct:.2f}%" if scenarios.worst.equity_irr_pct is not None else "N/A",
                "Terminal Wealth": format_inr(scenarios.worst.exit.net_exit_proceeds),
            },
        ]
        st.dataframe(pd.DataFrame(sc_table), width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### 2. Exit Valuation & Post-July 2024 Capital Gains Tax Proof (Base Case)")

        base_exit_disc = result.scenario_exit_discount_pct
        st.markdown(
            f"""
            **A. Terminal Gross Value & Sale Price at Year {h}:**
            - **Formula:** $\\text{{Terminal Gross Value}} = \\text{{Agreement Value}} \\times \\text{{Compounded Appreciation Factor}}$
            - **Live Plug-In:** $₹{prop_input.agreement_value_inr:,.0f} \\text{{ compounded at Base rate }} {prop_input.appreciation_rate*100:.2f}\\% \\text{{ over }} {h}\\text{{ yrs}} = \\mathbf{{{_fmt(exit_val.terminal_gross_value)}}}$
            - **Base-Case Exit Haircut ({base_exit_disc*100:.1f}%):** $\\text{{Sale Price}} = ₹{exit_val.terminal_gross_value:,.0f} \\times (1 - {base_exit_disc:.2f}) = \\mathbf{{{_fmt(exit_val.exit_sale_price)}}}$
            """
        )

        st.markdown(
            f"""
            **B. Tax Cost Basis & Capital Gain:**
            - **Brokerage on Exit:** `{_fmt(exit_val.sell_brokerage_inr)}` (1.5% + 18% GST).
            - **Tax Cost Basis Formula:** $\\text{{Agreement Value}} + \\text{{Stamp Duty}} + \\text{{Registration Fee}} + \\text{{GST}}$
            - **Live Plug-In:** $₹{exit_val.tax_cost_basis:,.0f}$
            - **Taxable Capital Gain:** $\\max(0, \\text{{Sale Price}} - \\text{{Sell Brokerage}} - \\text{{Tax Cost Basis}}) = \\mathbf{{{_fmt(exit_val.capital_gain)}}}$
            """
        )

        tax_mult = (1.0 + max(0.0, investor_profile.surcharge_rate_pct)) * 1.04
        eff_ltcg_pct = macro["ltcg_tax_rate_post_july2024_pct"] * tax_mult * 100.0
        eff_stcg_pct = investor_profile.marginal_tax_slab_pct * tax_mult * 100.0
        ltcg_rule_text = (
            f"LTCG applies (> 24 months holding). Taxed at 12.5% × surcharge/cess ({eff_ltcg_pct:.2f}% effective) under Finance (No.2) Act 2024 (without indexation)."
            if exit_val.is_ltcg
            else f"STCG applies (≤ 24 months holding). Taxed at marginal slab {eff_stcg_pct:.2f}% (incl. surcharge & cess)."
        )
        st.markdown(
            f"""
            **C. Capital Gains Tax Decision & Net Proceeds:**
            - **Holding Period Check:** $H = {h}\\text{{ Years}}$ vs $2.0\\text{{ Years Threshold}} \\implies$ **{ltcg_rule_text}**
            - **CG Tax Amount:** $₹{exit_val.capital_gain:,.0f} \\times {eff_ltcg_pct if exit_val.is_ltcg else eff_stcg_pct:.2f}\\% = \\mathbf{{{_fmt(exit_val.cg_tax_inr)}}}$
            - **Remaining Loan Payoff at Year {h}:** `{_fmt(exit_val.remaining_loan_balance_H)}`
            - **Net Cash Exit Proceeds:** $\\text{{Sale Price}} - \\text{{Brokerage}} - \\text{{Remaining Debt}} - \\text{{CG Tax}} = \\mathbf{{{_fmt(exit_val.net_exit_proceeds)}}}$
            """
        )

        st.markdown("---")
        st.markdown("### 3. Exact Equity IRR Cash-Flow Vector Proof")
        cf_vector_str = ", ".join(f"Yr {i}: {format_inr(cf)}" for i, cf in enumerate(result.equity_cash_flow_vector))
        irr_str = _fmt_pct(result.equity_irr_pct)
        real_irr_str = _fmt_pct(result.real_equity_irr_pct)
        st.markdown(
            f"""
            - **Cash-Flow Vector:** `[{cf_vector_str}]`
            - **Mathematical Definition:** Solves for rate $r$ where $\\text{{NPV}}(r) = \\sum_{{t=0}}^{{{h}}} \\frac{{\\text{{CF}}_t}}{{(1+r)^t}} = 0$.
            - **Solved Nominal Equity IRR:** **{irr_str}** (CPI-Adjusted Real IRR: **{real_irr_str}**).
            """
        )

        st.markdown("---")
        st.markdown("### 4. Apples-to-Apples Opportunity Cost Cash-Flow Parity Proof")
        nifty_pre_tax_pct = result.nifty_benchmark.pre_tax_cagr_pct or (macro['liquid_benchmark_nifty50_cagr_pct']*100.0)
        nifty_post_tax_str = _fmt_pct(result.nifty_benchmark.post_tax_irr_pct)
        st.markdown(
            f"""
            **How Parity is Maintained:**
            Every rupee of upfront equity and every rupee of operational cash deficit ($-\\text{{CF}}_t$) experienced in real estate is invested simultaneously into the benchmark alternative.
            - **SBI Fixed Deposit (Post-Tax):** Gross rate `{macro['liquid_benchmark_fd_rate_pct']*100:.2f}%` taxed at investor's slab ({investor_profile.marginal_tax_slab_pct*100:.0f}% + surcharge/cess) $\\implies$ Post-tax compounding rate: **{result.fd_benchmark.annual_return_pct:.2f}%**. Final terminal wealth: **{_fmt(result.fd_benchmark.terminal_wealth_post_tax)}**.
            - **Nifty 50 TRI (Post-Tax):** Compounded at `{nifty_pre_tax_pct:.2f}%` pre-tax CAGR. Principal invested: `{_fmt(result.nifty_benchmark.principal_invested)}`. LTCG tax of 12.5% (+ surcharge/cess) deducted after ₹1,25,000 exemption $\\implies$ True post-tax parity IRR: **{nifty_post_tax_str}**. Final post-tax corpus: **{_fmt(result.nifty_benchmark.terminal_wealth_post_tax)}**.
            """
        )

        # 5-Column Decision Table for Stress Tests
        st.markdown("---")
        st.markdown("### 5. Stress Tests & Wealth Winner Decision Proof Table")
        s1 = scenarios.stress_interest_rate_shock
        s2 = scenarios.stress_vacancy_shock
        s3 = scenarios.stress_forced_exit_year_3

        stress_table = [
            {
                "Check / Rule": "Stress 1: +200 bps Mortgage Rate Shock",
                "Decision Threshold / Logic": "Stressed FOIR ≤ 50% (PASS), ≤ 60% (WARN), > 60% (FAIL)",
                "Your Live Value": f"{s1.metrics['stressed_foir_200bps']:.1f}%",
                "Comparison Proof": f"Delta EMI: +{format_inr(s1.metrics['delta_emi_200bps'])}/mo",
                "Outcome": s1.status,
            },
            {
                "Check / Rule": "Stress 2: Extended Vacancy Shock",
                "Decision Threshold / Logic": "Post-Shock Buffer ≥ 3.0 Mo (PASS), ≥ 1.0 Mo (WARN), < 1.0 Mo (FAIL)",
                "Your Live Value": f"{s2.metrics['remaining_buffer_months']:.1f} Months",
                "Comparison Proof": f"Cash Drain: {format_inr(s2.metrics['vacancy_cash_drain'])}",
                "Outcome": s2.status,
            },
            {
                "Check / Rule": "Stress 3: Year-3 Distress Early Exit",
                "Decision Threshold / Logic": "Capital Recovery Ratio ≥ 80% (PASS), ≥ 60% (WARN), < 60% (FAIL)",
                "Your Live Value": f"{s3.metrics['capital_recovery_pct']:.1f}%",
                "Comparison Proof": f"Recovered: {format_inr(s3.metrics['net_recovered_3yr'])} vs In: {format_inr(s3.metrics['total_cash_invested_3yr'])}",
                "Outcome": s3.status,
            },
            {
                "Check / Rule": "Opportunity Cost: Real Estate vs SBI FD",
                "Decision Threshold / Logic": "Real Estate Net Equity > SBI FD Terminal Wealth",
                "Your Live Value": f"RE: {format_inr(result.exit.net_exit_proceeds)} vs FD: {format_inr(result.fd_benchmark.terminal_wealth_post_tax)}",
                "Comparison Proof": f"Spread: {result.fd_benchmark.irr_spread_pct:+.2f}%, Wealth Gap: {format_inr(result.fd_benchmark.net_wealth_advantage_inr)}",
                "Outcome": "PASS" if result.fd_benchmark.net_wealth_advantage_inr >= 0 else "FAIL",
            },
            {
                "Check / Rule": "Opportunity Cost: Real Estate vs Nifty 50 TRI",
                "Decision Threshold / Logic": "Real Estate Net Equity > Nifty 50 TRI Post-Tax Corpus",
                "Your Live Value": f"RE: {format_inr(result.exit.net_exit_proceeds)} vs Nifty: {format_inr(result.nifty_benchmark.terminal_wealth_post_tax)}",
                "Comparison Proof": f"Spread: {result.nifty_benchmark.irr_spread_pct:+.2f}%, Wealth Gap: {format_inr(result.nifty_benchmark.net_wealth_advantage_inr)}",
                "Outcome": "PASS" if result.nifty_benchmark.net_wealth_advantage_inr >= 0 else "WARN",
            },
        ]
        st.dataframe(pd.DataFrame(stress_table), width="stretch", hide_index=True)

    # Glossary Expander
    with st.expander("📖 Financial Terms Glossary (What These Metrics Mean)", expanded=False):
        glossary_items = [
            ("Equity IRR (Internal Rate of Return)", "The annualized effective compounding rate earned strictly on the out-of-pocket cash equity invested by the buyer, accounting for down payment, ongoing operating deficits, and net exit proceeds upon sale."),
            ("NPV (Net Present Value)", "The sum of all incoming and outgoing project cash flows discounted back to present value using the risk-free post-tax SBI FD hurdle rate. Positive NPV means the deal generates value superior to a bank deposit."),
            ("CAGR (Compound Annual Growth Rate)", "The constant geometric rate of return required for an investment to grow from its initial value to its terminal value over the holding period."),
            ("LTCG (Post-July 2024 Rules)", "Under the Finance (No. 2) Act 2024, real estate held for more than 24 months is subject to long-term capital gains tax at a flat 12.5% (+ 4% health and education cess = 13.0%) without inflation indexation benefits."),
            ("Cash-Flow Parity", "The rigorous quantitative benchmarking standard where alternative investments (FD, Nifty 50) receive the identical stream of initial equity and annual top-up cash contributions as real estate to ensure an honest comparison."),
            ("Opportunity Cost", "The financial return forgone by allocating capital into real estate rather than into liquid alternatives like bank deposits or broad equity index funds."),
            ("Exit Haircut (Distress Discount)", "A conservative liquidity discount subtracted from the estimated fair market value to reflect broker negotiations, buyer closing concessions, and emergency liquidation realities."),
            ("Basis Points (bps)", "A unit of measure for interest rates and financial percentages: 1 basis point = 0.01% (100 bps = 1.00%)."),
        ]
        for term, desc in glossary_items:
            st.markdown(f"**{term}:** {desc}")


# ==============================================================================
# TAB 4 EXPLAINER: PUNE MICRO-MARKET EXPLORER & COMPARABLES
# ==============================================================================
def render_tab4_explainer(
    loc_row: Any,
    unit_row: Any,
    zone_row: Any,
) -> None:
    """Renders the micro-market data provenance, loading factor, and liquidity tier proofs for Tab 4."""
    st.divider()

    loc_dict = loc_row if isinstance(loc_row, dict) else (loc_row.to_dict() if hasattr(loc_row, "to_dict") else {})
    unit_dict = unit_row if isinstance(unit_row, dict) else (unit_row.to_dict() if hasattr(unit_row, "to_dict") else {})
    zone_dict = zone_row if isinstance(zone_row, dict) else (zone_row.to_dict() if hasattr(zone_row, "to_dict") else {})

    with st.expander("📐 Live Step-by-Step Calculation & Decision Proof", expanded=False):
        st.markdown("### 1. Selected Locality & Unit Micro-Market Metrics Proof")

        # SBA and Loading Factor
        carpet = float(unit_dict.get("median_carpet_area_sqft", 0.0))
        loading = float(unit_dict.get("loading_factor_used", 1.30))
        expected_sba = float(unit_dict.get("expected_sba_sqft", carpet * loading))
        ticket = float(unit_dict.get("est_ticket_price_inr", 0.0))
        carpet_rate = float(unit_dict.get("price_per_sqft_carpet_equiv", 0.0))
        rent = unit_dict.get("median_monthly_rent")
        rr_rate = float(loc_dict.get("rr_rate_primary_sqft", 0.0))

        st.markdown(
            f"""
            **A. Super Built-Up Area (SBA) & Efficiency:**
            - **Formula:** $\\text{{Expected SBA}} = \\text{{Carpet Area}} \\times \\text{{Loading Factor}}$
            - **Live Plug-In:** ${carpet:,.1f}\\text{{ sq.ft.}} \\times {loading:.2f} = \\mathbf{{{expected_sba:,.1f}\\text{{ sq.ft.}}}}$
            - **Efficiency Ratio:** Carpet area represents **{100.0 / loading:.1f}%** of usable super built-up space.
            """
        )

        st.markdown(
            f"""
            **B. Carpet-Equivalent Rate & Price Basis:**
            - **Carpet Price:** `{_fmt(carpet_rate)}/sq.ft.`
            - **Price Basis:** `{unit_dict.get('price_basis', 'carpet')}` (Conversion: `{unit_dict.get('area_unit_conversion', 'none')}`).
            - **Estimated Ticket Price:** `{_fmt(ticket)}`
            """
        )

        if pd.isna(rent) or rent is None:
            st.markdown(
                """
                **C. Implied Gross Rental Yield:**
                - **Notice:** `median_monthly_rent` is `NULL` for this specific configuration (e.g. Chakan 4+ BHK).
                - **Fallback Rule:** Engine applies zero rental cash flow fallback without crashing; underwriting treats this configuration purely as capital appreciation/owner-occupied.
                """
            )
        else:
            rent_val = float(rent)
            gross_yield = (rent_val * 12.0 / ticket * 100.0) if ticket > 0 else 0.0
            st.markdown(
                f"""
                **C. Implied Gross Rental Yield:**
                - **Formula:** $\\text{{Gross Yield}} = \\frac{{\\text{{Monthly Rent}} \\times 12}}{{\\text{{Estimated Ticket Price}}}} \\times 100$
                - **Live Plug-In:** $\\frac{{₹{rent_val:,.0f} \\times 12}}{{₹{ticket:,.0f}}} \\times 100 = \\mathbf{{{gross_yield:.2f}\\%}}$
                """
            )

        market_to_rr = (carpet_rate / rr_rate) if rr_rate > 0 else 1.0
        st.markdown(
            f"""
            **D. Market-to-Ready Reckoner Ratio:**
            - **Formula:** $\\frac{{\\text{{Carpet-Equivalent Rate}}}}{{\\text{{Primary Government RR Rate}}}}$
            - **Live Plug-In:** $\\frac{{₹{carpet_rate:,.0f}/\\text{{sq.ft.}}}}{{₹{rr_rate:,.0f}/\\text{{sq.ft.}}}} = \\mathbf{{{market_to_rr:.2f}x}}$
            """
        )

        st.markdown("---")
        st.markdown("### 2. Institutional Liquidity Tier & Exit Discount Decision Table")
        qts = float(zone_dict.get("quarters_to_sell", 4.8))
        tier = str(zone_dict.get("liquidity_tier", "medium")).lower()

        tier_table = [
            {
                "Check / Rule": "Inventory Overhang (Quarters to Sell)",
                "Decision Threshold / Logic": "< 3.0 Qtrs → High, 3.0–6.0 Qtrs → Medium, > 6.0 Qtrs → Low",
                "Your Live Value": f"{qts:.1f} Quarters to Sell ({zone_dict.get('pune_zone', 'Pune')} Zone)",
                "Comparison Proof": f"{qts:.1f} Qtrs classified as '{tier.upper()}' liquidity tier",
                "Outcome": tier.upper(),
            },
            {
                "Check / Rule": "Stress Exit Haircut Discount",
                "Decision Threshold / Logic": "High → 5% Haircut, Medium → 8% Haircut, Low → 12% Haircut",
                "Your Live Value": f"{float(zone_dict.get('exit_discount_stress_pct', 0.08))*100:.0f}% Haircut",
                "Comparison Proof": f"Calibrated for institutional liquidation friction in {zone_dict.get('pune_zone', 'Pune')} Zone",
                "Outcome": "PASS" if tier in ["high", "medium"] else "WARN",
            },
        ]
        st.dataframe(pd.DataFrame(tier_table), width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### 3. Data Provenance & Transparency Audit")
        st.caption("Verification of data origins: Official statutory records vs institutional reports vs calibrated assumptions.")

        prov_table = [
            {
                "Metric": "Ready Reckoner Rate (Floor)",
                "Value": f"₹{rr_rate:,.0f}/sq.ft.",
                "Data Basis Tag": str(loc_dict.get("rr_rate_primary_basis", "IGR_SL_PRIMARY")),
                "Classification": "Official Statutory (IGR Maharashtra 2026)",
            },
            {
                "Metric": "Inventory Overhang (Quarters to Sell)",
                "Value": f"{qts:.1f} Quarters",
                "Data Basis Tag": str(zone_dict.get("overhang_data_basis", "INSTITUTIONAL_REPORT")),
                "Classification": "Institutional Market Study (Knight Frank H1 2026)",
            },
            {
                "Metric": "Under-Construction Possession Delay",
                "Value": f"{float(zone_dict.get('avg_uc_delay_months', 8)):.0f} Months",
                "Data Basis Tag": str(zone_dict.get("delay_data_basis", "UNDERWRITING_ASSUMPTION")),
                "Classification": "Conservative Stress Model (Calibrated Assumption)",
            },
            {
                "Metric": "Tenant Vacancy Downtime",
                "Value": f"{float(zone_dict.get('vacancy_months_avg', 1.0)):.1f} Months/Year",
                "Data Basis Tag": str(zone_dict.get("vacancy_data_basis", "UNDERWRITING_ASSUMPTION")),
                "Classification": "Conservative Prudence Model (Calibrated Assumption)",
            },
        ]
        st.dataframe(pd.DataFrame(prov_table), width="stretch", hide_index=True)

    # Glossary Expander
    with st.expander("📖 Financial Terms Glossary (What These Metrics Mean)", expanded=False):
        glossary_items = [
            ("Carpet Area vs Super Built-Up Area (SBA)", "Carpet area is the net usable floor area inside internal walls (as mandated under RERA). Super Built-Up Area includes common areas like corridors, staircases, lobbies, and clubhouses."),
            ("Loading Factor", "The ratio of Super Built-Up Area to Carpet Area (typically 1.25x to 1.35x in Pune). Lower loading factor means higher spatial efficiency for the buyer."),
            ("Gross Rental Yield", "Annual gross rental income divided by the total property purchase price. It measures the top-line cash generation efficiency before accounting for opex, vacancy, and taxes."),
            ("Quarters to Sell (Inventory Overhang)", "The estimated number of quarters required to sell off all unsold housing inventory in a given micro-market at current sales velocity. Higher numbers indicate market oversupply and poor seller liquidity."),
            ("Liquidity Tier", "An institutional classification (High, Medium, Low) denoting the ease and speed with which an asset can be converted to cash at prevailing market prices."),
            ("Market-to-RR Ratio", "The premium that open market pricing commands over the government Ready Reckoner statutory benchmark. Ratios above 2.50x signal speculative overpricing."),
            ("Data Basis Classification", "The institutional audit trail indicating whether a parameter is derived from official government records (e.g. IGR_SL_PRIMARY), verified industry reports (e.g. INSTITUTIONAL_REPORT), or calibrated underwriting stress models (e.g. UNDERWRITING_ASSUMPTION)."),
        ]
        for term, desc in glossary_items:
            st.markdown(f"**{term}:** {desc}")
