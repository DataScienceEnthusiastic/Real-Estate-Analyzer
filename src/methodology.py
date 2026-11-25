"""Plain-English explanation engine.

Every headline number in the app is accompanied by a short, human-readable
narrative that explains *what* it means and *how* it was calculated. This is
the transparency layer: a user should never have to guess where a figure comes
from. All functions return Markdown strings suitable for ``st.markdown``.
"""
from __future__ import annotations

from src.charts import format_inr
from src.models import (
    AcquisitionCostBreakdown,
    CashFlowYear,
    ExitValuation,
    GateResult,
    InvestorProfile,
    LoanBreakdown,
    MasterVerdict,
    UnderwritingResult,
)

# ---------------------------------------------------------------------------
# Glossary of financial terms (used by the "Methodology & Glossary" UI)
# ---------------------------------------------------------------------------
FINANCIAL_GLOSSARY: dict[str, str] = {
    "All-In Acquisition Cost": (
        "The total cash you must pay to take ownership, excluding the loan. "
        "= Agreement Value + Stamp Duty + Registration Fee + (GST if under-construction) "
        "+ Brokerage (incl. GST) + Interiors & misc."
    ),
    "Stamp Duty": (
        "A state tax paid at registration. It is levied on the *higher* of the "
        "agreement value and the government Ready Reckoner (RR) value of the unit. "
        "Pune urban: 7% (male) / 6% (sole female) / 6.5% (joint 50:50); rural 4%."
    ),
    "Ready Reckoner (RR) Value": (
        "The minimum valuation published by the state government (IGR Maharashtra) "
        "for a property. It acts as a statutory floor for stamp duty and TDS."
    ),
    "Registration Fee": (
        "1% of the stamp-duty base, capped at ₹30,000 in Maharashtra."
    ),
    "GST": (
        "Applicable only on under-construction units. 1% on affordable units "
        "(≤ ₹45L and ≤ 60 sq.m), else 5%. Not charged on ready/resale properties."
    ),
    "TDS (Sec 194-IA / 393(1))": (
        "If the stamp-duty base is ≥ ₹50 lakh, the buyer must deduct 1% of the "
        "consideration and deposit it with the income-tax department. This is a "
        "compliance obligation, NOT an extra cost to the buyer, so it is excluded "
        "from the all-in cost."
    ),
    "LTV (Loan-to-Value)": (
        "Loan amount as a share of the agreement value. RBI caps: 90% ≤ ₹30L, "
        "80% ₹30–75L, 75% > ₹75L."
    ),
    "EMI": (
        "Equated Monthly Instalment = P × r × (1+r)^n / ((1+r)^n − 1), where "
        "P = loan, r = monthly interest rate, n = loan tenure in months."
    ),
    "Pre-EMI Interest": (
        "During construction you only pay interest on the portion of the loan "
        "already drawn down (here modelled as a linear drawdown of 55% of the "
        "loan), not the full EMI. Principal repayment starts after possession."
    ),
    "FOIR (Fixed Obligation to Income Ratio)": (
        "The share of your take-home income consumed by EMIs. "
        "= (Existing EMIs + New EMI) / Monthly Take-Home Income. "
        "Healthy ≤ 40%; 40–50% elevated; > 50% rejected by underwriting."
    ),
    "EGI (Effective Gross Income)": (
        "Gross annual rent minus vacancy loss — the rent you actually expect to collect."
    ),
    "NOI (Net Operating Income)": (
        "EGI minus operating expenses (maintenance + property tax, inflation-adjusted)."
    ),
    "Gross Rental Yield": (
        "Year-1 gross annual rent ÷ All-In Acquisition Cost. A headline 'how much "
        "does it rent for' figure."
    ),
    "Net Rental Yield": (
        "Year-1 NOI ÷ All-In Acquisition Cost. The yield after running costs."
    ),
    "Equity IRR": (
        "The annualised return on the cash you actually put in (down payment + "
        "closing costs + operating deficits), after receiving rent and the final "
        "net exit proceeds. It is the single best 'did this beat my alternatives?' "
        "number."
    ),
    "NPV": (
        "Net Present Value of all deal cash flows discounted at the post-tax SBI "
        "FD rate. Positive = the deal adds value over a risk-free deposit."
    ),
    "LTCG (Long-Term Capital Gain)": (
        "Profit on sale after holding > 24 months. Taxed at a flat 12.5% (+4% "
        "cess) without indexation under Finance (No.2) Act 2024 for acquisitions "
        "post 23-Jul-2024; below 24 months it is taxed at your slab rate."
    ),
    "Cash-flow parity (opportunity cost)": (
        "To compare fairly, the 'alternatives' (SBI FD and Nifty 50 TRI) are "
        "simulated with exactly the same cash outlays as real estate: every rupee "
        "deficit the property would cost you is added as an extra deposit in the "
        "benchmark. Only then are terminal wealth figures comparable."
    ),
    "Holding Period": (
        "How many years you plan to own before selling. Longest relevant to "
        "amortization, LTCG eligibility and the wealth trajectory."
    ),
}

# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------
def build_verdict_narrative(verdict: MasterVerdict) -> str:
    """A one-paragraph, human reading of the master verdict."""
    gate_summary = []
    for g in verdict.gates:
        glyph = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[g.status]
        gate_summary.append(f"{glyph} **Gate {g.gate_number} ({g.name}): {g.status}**")
    return (
        f"### {verdict.status}\n\n"
        f"{verdict.summary}\n\n"
        + "\n".join(gate_summary)
    )


def explain_acquisition(acq: AcquisitionCostBreakdown) -> str:
    """Explains how the all-in acquisition figure is constructed."""
    parts = [
        f"**Agreement Value** — the price you and the seller agreed on: "
        f"**{format_inr(acq.agreement_value_inr)}**.",
        f"**Stamp Duty base** — the *higher* of the agreement value and the "
        f"government Ready Reckoner value (**{format_inr(acq.rr_market_value)}**): "
        f"**{format_inr(acq.stamp_duty_base)}**. Stamp duty at **{acq.effective_stamp_duty_pct*100:.2f}%** "
        f"= **{format_inr(acq.stamp_duty_inr)}**.",
        f"**Registration Fee** — 1% of the base, capped at ₹30,000: "
        f"**{format_inr(acq.registration_fee_inr)}**.",
    ]
    if acq.gst_inr > 0:
        parts.append(
            f"**GST ({acq.gst_rate*100:.0f}%)** — applies because this is an "
            f"under-construction unit: **{format_inr(acq.gst_inr)}**. "
            f"({'Affordable-housing rate' if acq.is_affordable_gst else 'Non-affordable rate'})."
        )
    if acq.buy_brokerage_inr > 0:
        parts.append(
            f"**Buy-side brokerage + 18% GST** — **{format_inr(acq.buy_brokerage_inr)}**."
        )
    if acq.interiors_and_misc_inr > 0:
        parts.append(
            f"**Interiors & misc** — **{format_inr(acq.interiors_and_misc_inr)}**."
        )
    tds_line = (
        f"**TDS (not a cost):** ₹{acq.tds_amount_inr:,.0f} must be deducted and "
        f"remitted by you under Sec 194-IA — this flows to the seller/tax dept, "
        f"not out of your pocket, so it is excluded above."
        if acq.tds_applicable else "**TDS:** not applicable (below ₹50L threshold)."
    )
    return (
        "**All-In Acquisition Cost** = the total cash needed to take ownership:  "
        f"**{format_inr(acq.all_in_acquisition_cost)}**.\n\n"
        + "\n\n".join(parts)
        + f"\n\n{tds_line}"
    )


def explain_loan(loan: LoanBreakdown, investor: InvestorProfile, net_income: float) -> str:
    foir = (investor.existing_monthly_emis_inr + loan.monthly_emi) / net_income * 100.0 if net_income > 0 else 0.0
    return (
        f"**Loan amount** = Agreement Value × effective LTV ({loan.effective_ltv*100:.0f}%, "
        f"RBI cap {loan.rbi_max_ltv*100:.0f}%): **{format_inr(loan.loan_amount)}**.\n\n"
        f"**Upfront equity you must bring** = All-In Cost − Loan = "
        f"**{format_inr(loan.required_upfront_cash)}**.\n\n"
        f"**Monthly EMI** = **{format_inr(loan.monthly_emi)}** → FOIR "
        f"**{foir:.1f}%** ({'safe' if foir <= 40 else 'elevated'}).\n\n"
        f"**Pre-EMI** during construction: interest only on drawn funds "
        f"(≈ ₹{loan.monthly_pre_emi:,.0f}/mo)."
    )


def explain_tax_year(cf: CashFlowYear, investor: InvestorProfile) -> str:
    regime = "old" if investor.tax_regime == "old" else "new"
    if cf.tax_impact >= 0:
        return (
            f"Year {cf.year}: net **tax payable ₹{cf.tax_impact:,.0f}** (regime: {regime}). "
            f"This is the tax on net house-property income after the 30% standard "
            f"deduction and interest set-off (new regime blocks loss set-off)."
        )
    return (
        f"Year {cf.year}: **tax saving ₹{-cf.tax_impact:,.0f}** (regime: {regime}). "
        f"The home-loan interest deduction lowers your taxable income, saving you tax."
    )


def explain_exit(exit_val: ExitValuation) -> str:
    ltcg_tag = "LTCG @ 12.5% + 4% cess" if exit_val.is_ltcg else "STCG @ your slab rate"
    return (
        f"**Terminal gross value** after appreciation = **{format_inr(exit_val.terminal_gross_value)}**.\n\n"
        f"**Resale price** (after {exit_val.exit_sale_price and 'any'} exit discount) = "
        f"**{format_inr(exit_val.exit_sale_price)}**.\n\n"
        f"**Tax cost basis** (agreement + stamp duty + registration + GST) = "
        f"**{format_inr(exit_val.tax_cost_basis)}** → capital gain "
        f"**{format_inr(exit_val.capital_gain)}**, taxed as {ltcg_tag} → "
        f"**{format_inr(exit_val.cg_tax_inr)}**.\n\n"
        f"**Net exit proceeds** = sale price − sale brokerage − remaining loan − capital-gains tax = "
        f"**{format_inr(exit_val.net_exit_proceeds)}**."
    )


def explain_benchmarks(res: UnderwritingResult) -> str:
    fd = res.fd_benchmark
    ny = res.nifty_benchmark
    return (
        f"**SBI Fixed Deposit (post-tax)** — {fd.annual_return_pct:.2f}%/yr → "
        f"terminal **{format_inr(fd.terminal_wealth_post_tax)}**; advantage vs real estate "
        f"**{format_inr(fd.net_wealth_advantage_inr)}** (IRR spread {fd.irr_spread_pct:+.2f}%).\n\n"
        f"**Nifty 50 TRI (post-tax)** — {ny.annual_return_pct:.2f}%/yr → "
        f"terminal **{format_inr(ny.terminal_wealth_post_tax)}**; advantage vs real estate "
        f"**{format_inr(ny.net_wealth_advantage_inr)}** (IRR spread {ny.irr_spread_pct:+.2f}%).\n\n"
        f"*These use cash-flow parity: the alternatives deposit exactly the same "
        f"rupees the property would require — a fair apples-to-apples comparison.*"
    )


def explain_gate(gate: GateResult) -> str:
    level = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}[gate.status]
    body = "\n".join(f"- {r}" for r in gate.reasons)
    return f"**{level}** — {gate.name}\n\n{body}"
