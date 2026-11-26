"""Institutional PDF Investment Memorandum Generator.

Generates an executive-grade, multi-page PDF financial underwriting memorandum
for Pune residential real estate, detailing deal parameters, statutory acquisition costs,
loan amortization schedules, cash flows, multi-scenario stress tests, benchmark comparisons,
embedded visual charts, and decision gate audits.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.advanced_analysis import (
    compute_break_even_appreciation,
    compute_probability_weighted,
    compute_return_decomposition,
    compute_sensitivity_grid,
    compute_tornado_data,
)
from src.charts import (
    create_acquisition_waterfall_chart,
    create_amortization_chart,
    create_cash_flow_trajectory_chart,
    create_expected_value_chart,
    create_micromarket_scatter_chart,
    create_sensitivity_heatmap,
    create_tornado_chart,
    create_wealth_trajectory_chart,
    format_inr,
    prepare_fig_for_pdf,
)
from src.data_loader import get_merged_dataset
from src.pdf_charts_fallback import (
    render_amortization_png,
    render_cash_flow_png,
    render_expected_value_png,
    render_heatmap_png,
    render_scatter_png,
    render_tornado_png,
    render_waterfall_png,
    render_wealth_trajectory_png,
)
from src.models import (
    FinancingInput,
    InvestorProfile,
    LocalityRecommendation,
    MasterVerdict,
    PropertyInput,
    ScenarioMatrixResult,
)


def fmt_inr_pdf(val: Optional[float]) -> str:
    """Formats numeric values into standard INR strings for PDF without missing glyphs."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    abs_v = abs(val)
    sign = "-" if val < 0 else ""
    if abs_v >= 10_000_000:
        cr = abs_v / 10_000_000
        return f"{sign}INR {cr:.2f} Cr"
    elif abs_v >= 100_000:
        lakh = abs_v / 100_000
        return f"{sign}INR {lakh:.2f} L"
    elif abs_v >= 1_000:
        return f"{sign}INR {abs_v:,.0f}"
    else:
        return f"{sign}INR {abs_v:.2f}"


_inr = fmt_inr_pdf


_logger = logging.getLogger(__name__)
_KALEIDO_WORKING: Optional[bool] = None


def _render_plotly_png(fig_pdf: go.Figure, scale: float) -> bytes:
    """Renders a prepared Plotly figure to PNG bytes."""
    return fig_pdf.to_image(format="png", width=960, height=480, scale=scale)


def _fig_to_reportlab_image(
    fig: go.Figure,
    width: float,
    height: float,
    scale: float = 2.0,
    fallback_fn: Optional[Callable[[], bytes]] = None,
) -> Optional[Image]:
    """Converts a Plotly Figure into a ReportLab Image Flowable, falling back to Matplotlib if Kaleido/Chrome is unavailable."""
    global _KALEIDO_WORKING

    if _KALEIDO_WORKING is not False:
        try:
            fig_pdf = prepare_fig_for_pdf(fig)
            try:
                img_bytes = _render_plotly_png(fig_pdf, scale)
            except Exception:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    img_bytes = pool.submit(_render_plotly_png, fig_pdf, scale).result(timeout=30)
            _KALEIDO_WORKING = True
            return Image(io.BytesIO(img_bytes), width=width, height=height)
        except Exception as exc:
            _KALEIDO_WORKING = False
            _logger.warning("Kaleido/Chromium unavailable (%s); using pure-Python Matplotlib fallback.", exc)

    if fallback_fn is not None:
        try:
            fb_bytes = fallback_fn()
            return Image(io.BytesIO(fb_bytes), width=width, height=height)
        except Exception as fb_exc:
            _logger.warning("Matplotlib fallback failed for PDF chart: %s", fb_exc)

    return None


def generate_underwriting_pdf(
    property_obj: PropertyInput,
    financing_obj: FinancingInput,
    investor_obj: InvestorProfile,
    matrix: ScenarioMatrixResult,
    verdict: MasterVerdict,
    loc_row: Any,
    unit_row: Any,
    zone_row: Any,
    macro: dict[str, float],
    all_zone_recommendations: Optional[list[LocalityRecommendation]] = None,
) -> bytes:
    """Generates a complete, multi-page institutional PDF underwriting memorandum with embedded charts."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Executive color palette
    c_primary = colors.HexColor("#0f172a")  # Dark slate
    c_accent = colors.HexColor("#2563eb")   # Executive blue
    c_slate_dark = colors.HexColor("#1e293b")
    c_slate_light = colors.HexColor("#f8fafc")
    c_border = colors.HexColor("#cbd5e1")
    c_muted = colors.HexColor("#64748b")
    c_green = colors.HexColor("#059669")
    c_amber = colors.HexColor("#d97706")
    c_red = colors.HexColor("#dc2626")

    # Typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=c_primary,
    )
    style_subtitle = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_muted,
    )
    style_sec_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=5,
    )
    style_sec_heading_sm = ParagraphStyle(
        "SectionHeadingSm",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=c_slate_dark,
        spaceBefore=4,
        spaceAfter=3,
    )
    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=c_primary,
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.5,
        textColor=c_primary,
    )
    style_table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.5,
        textColor=c_primary,
    )
    style_table_cell_right = ParagraphStyle(
        "TableCellRight",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.5,
        textColor=c_primary,
        alignment=2,
    )
    style_table_cell_right_bold = ParagraphStyle(
        "TableCellRightBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.5,
        textColor=c_primary,
        alignment=2,
    )

    story: list[Any] = []
    base_res = matrix.base
    acq = base_res.acquisition
    loan = base_res.loan
    h = financing_obj.holding_period_years

    # =========================================================================
    # PAGE 1: EXECUTIVE UNDERWRITING MEMORANDUM & 4-GATE AUDIT
    # =========================================================================
    now_str = datetime.now().strftime("%d %B %Y · %H:%M IST")
    ownership_str = property_obj.buyer_ownership.replace("_", " ").title()
    status_str = property_obj.property_status.replace("_", " ").title()

    story.append(
        Paragraph(
            "<b>PUNE RESIDENTIAL REAL ESTATE INVESTMENT MEMORANDUM</b>",
            style_title,
        )
    )
    story.append(
        Paragraph(
            f"<b>Subject Asset:</b> {property_obj.locality_name} (PIN: {property_obj.pincode}) · "
            f"<b>Zone:</b> {property_obj.pune_zone} · <b>Jurisdiction:</b> {property_obj.jurisdiction} · "
            f"<b>Unit:</b> {property_obj.bhk.replace('_', ' ')} · <b>Ownership:</b> {ownership_str} · "
            f"<b>Status:</b> {status_str} · <b>Generated:</b> {now_str}",
            style_subtitle,
        )
    )
    story.append(Spacer(1, 8))

    # Verdict Box Banner
    v_color = c_green if verdict.status == "GO" else (c_amber if verdict.status == "CAUTION" else c_red)
    v_bg = (
        colors.HexColor("#ecfdf5")
        if verdict.status == "GO"
        else (colors.HexColor("#fffbeb") if verdict.status == "CAUTION" else colors.HexColor("#fef2f2"))
    )

    verdict_text = (
        f"<font size=13 color='{v_color.hexval()}'><b>UNDERWRITING VERDICT: {verdict.status}</b></font><br/>"
        f"<font size=8 color='#334155'>{verdict.summary}</font>"
    )
    verdict_table = Table([[Paragraph(verdict_text, style_body)]], colWidths=[523])
    verdict_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), v_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, v_color),
            ("PADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # Key Executive Snapshot Table (2-Row KPI Grid)
    irr_txt = f"{base_res.equity_irr_pct:.2f}%" if base_res.equity_irr_pct is not None else "N/A"
    foir = (
        (investor_obj.existing_monthly_emis_inr + loan.monthly_emi)
        / investor_obj.net_monthly_income_inr
        * 100.0
        if investor_obj.net_monthly_income_inr > 0
        else 0.0
    )
    post_cash = max(0.0, investor_obj.total_liquid_savings_inr - loan.required_upfront_cash)
    m_obligations = (
        investor_obj.monthly_living_expenses_inr
        + investor_obj.existing_monthly_emis_inr
        + loan.monthly_emi
    )
    buf_months = (post_cash / m_obligations) if m_obligations > 0 else 0.0

    kpi_data = [
        [
            Paragraph("<b>All-In Acquisition Cost</b>", style_table_header),
            Paragraph("<b>Required Upfront Equity</b>", style_table_header),
            Paragraph("<b>Monthly Debt Service (EMI)</b>", style_table_header),
            Paragraph("<b>Base Equity IRR</b>", style_table_header),
        ],
        [
            Paragraph(
                f"<b>{_inr(acq.all_in_acquisition_cost)}</b><br/><font color='#64748b' size=6.5>Base: {_inr(acq.agreement_value_inr)}</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b>{_inr(loan.required_upfront_cash)}</b><br/><font color='#64748b' size=6.5>Debt: {_inr(loan.loan_amount)} ({loan.effective_ltv*100:.0f}% LTV)</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b>{_inr(loan.monthly_emi)}</b>/mo<br/><font color='#64748b' size=6.5>FOIR: {foir:.1f}%</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b><font color='{c_green.hexval()}'>{irr_txt}</font></b><br/><font color='#64748b' size=6.5>NPV: {_inr(base_res.npv_inr)}</font>",
                style_table_cell_bold,
            ),
        ],
        [
            Paragraph("<b>Post-Purchase Buffer</b>", style_table_header),
            Paragraph("<b>Year-1 Net Rental Yield</b>", style_table_header),
            Paragraph("<b>Terminal Net Equity</b>", style_table_header),
            Paragraph("<b>Advantage vs SBI FD</b>", style_table_header),
        ],
        [
            Paragraph(
                f"<b>{buf_months:.1f} Months</b><br/><font color='#64748b' size=6.5>Liquid: {_inr(post_cash)}</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b>{base_res.year_1_net_yield_pct:.2f}%</b><br/><font color='#64748b' size=6.5>Gross: {base_res.year_1_gross_yield_pct:.2f}%</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b>{_inr(base_res.exit.net_exit_proceeds)}</b><br/><font color='#64748b' size=6.5>Year {h} Net Proceeds</font>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"<b>{'+' if base_res.fd_benchmark.net_wealth_advantage_inr >= 0 else ''}{_inr(base_res.fd_benchmark.net_wealth_advantage_inr)}</b><br/><font color='#64748b' size=6.5>Spread: {base_res.fd_benchmark.irr_spread_pct:+.2f}%</font>",
                style_table_cell_bold,
            ),
        ],
    ]
    t_kpi = Table(kpi_data, colWidths=[130.75, 130.75, 130.75, 130.75])
    t_kpi.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("BACKGROUND", (0, 2), (-1, 2), c_slate_dark),
            ("BACKGROUND", (0, 1), (-1, 1), c_slate_light),
            ("BACKGROUND", (0, 3), (-1, 3), c_slate_light),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # 4-Gate Decision Audit Table
    story.append(Paragraph("1. Institutional 4-Gate Decision Engine Audit", style_sec_heading))
    gate_table_data = [
        [
            Paragraph("<b>Gate</b>", style_table_header),
            Paragraph("<b>Gate Name & Focus</b>", style_table_header),
            Paragraph("<b>Status</b>", style_table_header),
            Paragraph("<b>Audit Evaluation & Core Reasons</b>", style_table_header),
        ]
    ]
    for g in verdict.gates:
        g_badge_color = c_green if g.status == "PASS" else (c_amber if g.status == "WARN" else c_red)
        gate_table_data.append([
            Paragraph(f"<b>Gate {g.gate_number}</b>", style_table_cell_bold),
            Paragraph(g.name, style_table_cell),
            Paragraph(f"<b><font color='{g_badge_color.hexval()}'>{g.status}</font></b>", style_table_cell_bold),
            Paragraph("<br/>".join(f"• {r}" for r in g.reasons), style_table_cell),
        ])

    t_gates = Table(gate_table_data, colWidths=[45, 130, 45, 303])
    t_gates.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 4.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(t_gates)
    story.append(Spacer(1, 9))

    # Investor Profile & Property Assumptions Summary
    story.append(Paragraph("2. Underwriting Inputs & Core Assumptions", style_sec_heading))
    inputs_data = [
        [
            Paragraph("<b>Investor Parameter</b>", style_table_header),
            Paragraph("<b>Value</b>", style_table_header),
            Paragraph("<b>Property & Mortgage Parameter</b>", style_table_header),
            Paragraph("<b>Value</b>", style_table_header),
        ],
        [
            Paragraph("Monthly Net Income", style_table_cell),
            Paragraph(f"INR {investor_obj.net_monthly_income_inr:,.0f}", style_table_cell_bold),
            Paragraph("Agreement Value", style_table_cell),
            Paragraph(f"INR {property_obj.agreement_value_inr:,.0f}", style_table_cell_bold),
        ],
        [
            Paragraph("Monthly Living Expenses", style_table_cell),
            Paragraph(f"INR {investor_obj.monthly_living_expenses_inr:,.0f}", style_table_cell_bold),
            Paragraph("Carpet Area / Expected SBA", style_table_cell),
            Paragraph(
                f"{property_obj.carpet_area_sqft:,.0f} / {property_obj.expected_sba_sqft:,.0f} sq.ft.",
                style_table_cell_bold,
            ),
        ],
        [
            Paragraph("Total Liquid Savings", style_table_cell),
            Paragraph(f"INR {investor_obj.total_liquid_savings_inr:,.0f}", style_table_cell_bold),
            Paragraph("Monthly Rent / Escalation", style_table_cell),
            Paragraph(
                f"INR {property_obj.monthly_rent_inr:,.0f} / {financing_obj.annual_rent_escalation_pct*100:.1f}%",
                style_table_cell_bold,
            ),
        ],
        [
            Paragraph("Tax Slab & Regime", style_table_cell),
            Paragraph(
                f"{investor_obj.marginal_tax_slab_pct*100:.0f}% ({investor_obj.tax_regime.upper()})",
                style_table_cell_bold,
            ),
            Paragraph("Loan LTV & Interest Rate", style_table_cell),
            Paragraph(
                f"{loan.effective_ltv*100:.0f}% @ {financing_obj.home_loan_interest_rate_pct*100:.2f}% ({financing_obj.loan_tenure_years}y)",
                style_table_cell_bold,
            ),
        ],
        [
            Paragraph("Emergency Buffer Target", style_table_cell),
            Paragraph(f"{investor_obj.emergency_buffer_months_target:.0f} Months", style_table_cell_bold),
            Paragraph("Appreciation CAGR & Horizon", style_table_cell),
            Paragraph(f"{property_obj.appreciation_rate*100:.2f}% / {h} Years", style_table_cell_bold),
        ],
    ]
    t_inputs = Table(inputs_data, colWidths=[140, 121.5, 140, 121.5])
    t_inputs.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_inputs)

    # =========================================================================
    # PAGE 2: STATUTORY VALUATION FLOOR & ACQUISITION WATERFALL VISUAL
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph("3. Statutory Valuation Floor & Acquisition Cost Waterfall", style_sec_heading)
    )

    acq_table_data = [
        [
            Paragraph("<b>Acquisition Cost Component</b>", style_table_header),
            Paragraph("<b>Statutory Formula & Calculation Basis</b>", style_table_header),
            Paragraph("<b>Rate / Slab</b>", style_table_header),
            Paragraph("<b>Amount (INR)</b>", style_table_header),
        ],
        [
            Paragraph("Agreement Value", style_table_cell_bold),
            Paragraph("Base transacted purchase consideration", style_table_cell),
            Paragraph("100.0%", style_table_cell),
            Paragraph(f"INR {acq.agreement_value_inr:,.0f}", style_table_cell_right_bold),
        ],
        [
            Paragraph("Ready Reckoner Floor", style_table_cell),
            Paragraph(
                f"Carpet ({property_obj.carpet_area_sqft:.0f} sq.ft.) × RR Rate (INR {property_obj.rr_rate_primary_sqft:,.0f}/sq.ft.)",
                style_table_cell,
            ),
            Paragraph("IGR Floor", style_table_cell),
            Paragraph(f"INR {acq.rr_market_value:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("Stamp Duty Base", style_table_cell),
            Paragraph("max(Agreement Value, Ready Reckoner Value)", style_table_cell),
            Paragraph("Floor Check", style_table_cell),
            Paragraph(f"INR {acq.stamp_duty_base:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("Stamp Duty", style_table_cell_bold),
            Paragraph(
                f"Base × ({property_obj.jurisdiction} base minus ownership rebate)",
                style_table_cell,
            ),
            Paragraph(f"{acq.effective_stamp_duty_pct*100:.2f}%", style_table_cell),
            Paragraph(f"INR {acq.stamp_duty_inr:,.0f}", style_table_cell_right_bold),
        ],
        [
            Paragraph("Registration Fee", style_table_cell),
            Paragraph("1% of Stamp Duty Base, capped at INR 30,000 maximum", style_table_cell),
            Paragraph("Capped at INR 30k", style_table_cell),
            Paragraph(f"INR {acq.registration_fee_inr:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("GST (Goods & Services Tax)", style_table_cell),
            Paragraph(
                "0% on Ready/Resale; 1% (Affordable) or 5% on Under-Construction",
                style_table_cell,
            ),
            Paragraph(f"{acq.gst_rate*100:.1f}%", style_table_cell),
            Paragraph(f"INR {acq.gst_inr:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("Acquisition Brokerage", style_table_cell),
            Paragraph("1.5% transaction commission + 18% GST on brokerage services", style_table_cell),
            Paragraph("1.77% eff.", style_table_cell),
            Paragraph(f"INR {acq.buy_brokerage_inr:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("Interiors & Fit-Outs", style_table_cell),
            Paragraph("Upfront initial improvement and furnishing allocation", style_table_cell),
            Paragraph("Fixed", style_table_cell),
            Paragraph(f"INR {acq.interiors_and_misc_inr:,.0f}", style_table_cell_right),
        ],
        [
            Paragraph("<b>ALL-IN ACQUISITION COST</b>", style_table_cell_bold),
            Paragraph("<b>Total capital required to complete purchase</b>", style_table_cell_bold),
            Paragraph("<b>100.0% All-In</b>", style_table_cell_bold),
            Paragraph(f"<b>INR {acq.all_in_acquisition_cost:,.0f}</b>", style_table_cell_right_bold),
        ],
    ]
    t_acq = Table(acq_table_data, colWidths=[130, 233, 70, 90])
    t_acq.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_acq)
    story.append(Spacer(1, 8))

    # TDS Callout Note
    tds_callout = (
        f"<b>TDS Compliance (Sec 194-IA / Sec 393):</b> Consideration of INR {acq.stamp_duty_base:,.0f} "
        f"{'exceeds INR 50 Lakhs threshold. Mandatory 1% TDS = <b>INR ' + f'{acq.tds_amount_inr:,.0f}' + '</b> must be deducted from seller payout and deposited with IT Department. <i>Not an extra cost to buyer.</i>' if acq.tds_applicable else 'is below INR 50 Lakhs. No TDS deduction required.'}"
    )
    t_tds = Table([[Paragraph(tds_callout, style_body)]], colWidths=[523])
    t_tds.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#fef3c7") if acq.tds_applicable else colors.HexColor("#ecfdf5"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                colors.HexColor("#f59e0b") if acq.tds_applicable else colors.HexColor("#10b981"),
            ),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(t_tds)
    story.append(Spacer(1, 10))

    # VISUAL 1: Acquisition Waterfall Chart
    story.append(Paragraph("<b>Visual Proof: All-In Acquisition Cost Waterfall</b>", style_sec_heading_sm))
    im_waterfall = _fig_to_reportlab_image(
        create_acquisition_waterfall_chart(acq),
        width=523,
        height=210,
        fallback_fn=lambda: render_waterfall_png(acq),
    )
    if im_waterfall:
        story.append(im_waterfall)
    story.append(Spacer(1, 6))

    # =========================================================================
    # PAGE 3: CASH FLOW SCHEDULE & DUAL DEBT / OPERATING CHARTS
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph("4. Year-by-Year Operating Cash Flow & Amortization Schedule", style_sec_heading)
    )

    cf_headers = [
        Paragraph("<b>Yr</b>", style_table_header),
        Paragraph("<b>Gross Rent</b>", style_table_header),
        Paragraph("<b>NOI</b>", style_table_header),
        Paragraph("<b>Annual EMI</b>", style_table_header),
        Paragraph("<b>Interest</b>", style_table_header),
        Paragraph("<b>Principal</b>", style_table_header),
        Paragraph("<b>Tax Imp.</b>", style_table_header),
        Paragraph("<b>Net CF</b>", style_table_header),
        Paragraph("<b>Debt Bal.</b>", style_table_header),
        Paragraph("<b>Net Equity</b>", style_table_header),
    ]
    cf_rows = [cf_headers]
    for row in base_res.annual_cash_flows:
        cf_rows.append([
            Paragraph(f"Y{row.year}", style_table_cell_bold),
            Paragraph(f"INR {row.gross_annual_rent:,.0f}", style_table_cell_right),
            Paragraph(f"INR {row.noi:,.0f}", style_table_cell_right),
            Paragraph(f"INR {row.annual_emi:,.0f}", style_table_cell_right),
            Paragraph(f"INR {row.interest_paid:,.0f}", style_table_cell_right),
            Paragraph(f"INR {row.principal_paid:,.0f}", style_table_cell_right),
            Paragraph(f"INR {row.tax_impact:,.0f}", style_table_cell_right),
            Paragraph(f"<b>INR {row.net_cash_flow:,.0f}</b>", style_table_cell_right_bold),
            Paragraph(f"INR {row.remaining_loan_balance:,.0f}", style_table_cell_right),
            Paragraph(f"<b>INR {row.net_equity:,.0f}</b>", style_table_cell_right_bold),
        ])

    t_cf = Table(cf_rows, colWidths=[30, 54, 52, 56, 52, 52, 49, 56, 61, 61])
    t_cf.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_cf)
    story.append(Spacer(1, 10))

    # VISUALS 2 & 3: Side-by-Side Cash Flow Trajectory & Amortization Charts
    story.append(
        Paragraph("<b>Visual Proof: Cash Flow Trajectory & Debt Amortization Schedule</b>", style_sec_heading_sm)
    )
    im_cf = _fig_to_reportlab_image(
        create_cash_flow_trajectory_chart(base_res.annual_cash_flows),
        width=255,
        height=180,
        fallback_fn=lambda: render_cash_flow_png(base_res.annual_cash_flows),
    )
    im_amort = _fig_to_reportlab_image(
        create_amortization_chart(base_res.loan.annual_schedule),
        width=255,
        height=180,
        fallback_fn=lambda: render_amortization_png(base_res.loan.annual_schedule),
    )
    if im_cf and im_amort:
        t_cf_amort = Table([[im_cf, im_amort]], colWidths=[261.5, 261.5])
        t_cf_amort.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 0),
            ])
        )
        story.append(t_cf_amort)
    elif im_cf:
        story.append(im_cf)
    elif im_amort:
        story.append(im_amort)

    # =========================================================================
    # PAGE 4: MULTI-SCENARIO MATRIX, WEALTH TRAJECTORY & STRESS TESTS
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph("5. Multi-Scenario Matrix & Risk Stress Testing", style_sec_heading)
    )

    sc_data = [
        [
            Paragraph("<b>Scenario Metric</b>", style_table_header),
            Paragraph("<b>Best Case</b>", style_table_header),
            Paragraph("<b>Base Case</b>", style_table_header),
            Paragraph("<b>Worst Case (Stress)</b>", style_table_header),
        ],
        [
            Paragraph("Annual Appreciation Rate", style_table_cell),
            Paragraph(f"{(matrix.best.scenario_appreciation_rate or (property_obj.appreciation_rate + 0.015))*100:.2f}%", style_table_cell_right),
            Paragraph(f"{(matrix.base.scenario_appreciation_rate or property_obj.appreciation_rate)*100:.2f}%", style_table_cell_right_bold),
            Paragraph(f"{(matrix.worst.scenario_appreciation_rate or max(0.02, property_obj.appreciation_rate - 0.025))*100:.2f}%", style_table_cell_right),
        ],
        [
            Paragraph("Annual Tenant Vacancy", style_table_cell),
            Paragraph(f"{property_obj.vacancy_months_best:.1f} Months", style_table_cell_right),
            Paragraph(f"{property_obj.vacancy_months_avg:.1f} Months", style_table_cell_right_bold),
            Paragraph(f"{property_obj.vacancy_months_worst:.1f} Months", style_table_cell_right),
        ],
        [
            Paragraph("Home Loan Interest Rate", style_table_cell),
            Paragraph(f"{(matrix.best.scenario_loan_rate or max(0.065, financing_obj.home_loan_interest_rate_pct - 0.005))*100:.2f}%", style_table_cell_right),
            Paragraph(f"{(matrix.base.scenario_loan_rate or financing_obj.home_loan_interest_rate_pct)*100:.2f}%", style_table_cell_right_bold),
            Paragraph(f"{(matrix.worst.scenario_loan_rate or (financing_obj.home_loan_interest_rate_pct + 0.02))*100:.2f}%", style_table_cell_right),
        ],
        [
            Paragraph("Exit Haircut Stress Discount", style_table_cell),
            Paragraph("0.0%", style_table_cell_right),
            Paragraph("0.0%", style_table_cell_right_bold),
            Paragraph(f"{property_obj.exit_discount_stress_pct*100:.1f}%", style_table_cell_right),
        ],
        [
            Paragraph("Under-Construction Delay", style_table_cell),
            Paragraph("N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else "0 Months", style_table_cell_right),
            Paragraph("N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else f"{property_obj.avg_uc_delay_months:.0f} Months", style_table_cell_right_bold),
            Paragraph("N/A (Ready Resale)" if property_obj.property_status == "ready_resale" else f"{int(round(property_obj.avg_uc_delay_months * 1.5))} Months", style_table_cell_right),
        ],
        [
            Paragraph("<b>Equity IRR (Annualized)</b>", style_table_cell_bold),
            Paragraph(
                f"<b>{matrix.best.equity_irr_pct:.2f}%</b>" if matrix.best.equity_irr_pct else "N/A",
                style_table_cell_right_bold,
            ),
            Paragraph(
                f"<b>{matrix.base.equity_irr_pct:.2f}%</b>" if matrix.base.equity_irr_pct else "N/A",
                style_table_cell_right_bold,
            ),
            Paragraph(
                f"<b>{matrix.worst.equity_irr_pct:.2f}%</b>" if matrix.worst.equity_irr_pct else "N/A",
                style_table_cell_right_bold,
            ),
        ],
        [
            Paragraph("<b>Net Terminal Exit Proceeds</b>", style_table_cell_bold),
            Paragraph(f"<b>{_inr(matrix.best.exit.net_exit_proceeds)}</b>", style_table_cell_right_bold),
            Paragraph(f"<b>{_inr(matrix.base.exit.net_exit_proceeds)}</b>", style_table_cell_right_bold),
            Paragraph(f"<b>{_inr(matrix.worst.exit.net_exit_proceeds)}</b>", style_table_cell_right_bold),
        ],
        [
            Paragraph("Net Present Value (@ FD Rate)", style_table_cell),
            Paragraph(f"{_inr(matrix.best.npv_inr)}", style_table_cell_right),
            Paragraph(f"{_inr(matrix.base.npv_inr)}", style_table_cell_right_bold),
            Paragraph(f"{_inr(matrix.worst.npv_inr)}", style_table_cell_right),
        ],
    ]
    t_sc = Table(sc_data, colWidths=[173, 116.6, 116.6, 116.6])
    t_sc.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_sc)
    story.append(Spacer(1, 8))

    # VISUAL 4: Wealth Trajectory Chart
    im_wealth = _fig_to_reportlab_image(
        create_wealth_trajectory_chart(matrix, h),
        width=523,
        height=185,
        fallback_fn=lambda: render_wealth_trajectory_png(matrix, h),
    )
    if im_wealth:
        story.append(im_wealth)
        story.append(Spacer(1, 8))

    # Dedicated Stress Test Summary
    story.append(Paragraph("6. Dedicated Downside Shock Simulations", style_sec_heading))
    s1 = matrix.stress_interest_rate_shock
    s2 = matrix.stress_vacancy_shock
    s3 = matrix.stress_forced_exit_year_3
    stress_data = [
        [
            Paragraph("<b>Stress Module</b>", style_table_header),
            Paragraph("<b>Status</b>", style_table_header),
            Paragraph("<b>Tested Metric & Shock Outcome</b>", style_table_header),
            Paragraph("<b>Underwriting Risk Assessment</b>", style_table_header),
        ],
        [
            Paragraph(s1.name, style_table_cell_bold),
            Paragraph(
                f"<b><font color='{(c_green if s1.status == 'PASS' else c_amber).hexval()}'>{s1.status}</font></b>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"Stressed FOIR: <b>{s1.metrics['stressed_foir_200bps']:.1f}%</b><br/>Delta EMI: +INR {s1.metrics['delta_emi_200bps']:,.0f}/mo",
                style_table_cell,
            ),
            Paragraph(s1.notes[0] if s1.notes else "Viable debt cushion", style_table_cell),
        ],
        [
            Paragraph(s2.name, style_table_cell_bold),
            Paragraph(
                f"<b><font color='{(c_green if s2.status == 'PASS' else c_amber).hexval()}'>{s2.status}</font></b>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"Post-Shock Buffer: <b>{s2.metrics['remaining_buffer_months']:.1f} Mo</b><br/>Drain: INR {s2.metrics['vacancy_cash_drain']:,.0f}",
                style_table_cell,
            ),
            Paragraph(s2.notes[0] if s2.notes else "Reserves withstand downtime", style_table_cell),
        ],
        [
            Paragraph(s3.name, style_table_cell_bold),
            Paragraph(
                f"<b><font color='{(c_green if s3.status == 'PASS' else c_amber).hexval()}'>{s3.status}</font></b>",
                style_table_cell_bold,
            ),
            Paragraph(
                f"Capital Recovered: <b>{s3.metrics['capital_recovery_pct']:.1f}%</b><br/>Net: INR {s3.metrics['net_recovered_3yr']:,.0f}",
                style_table_cell,
            ),
            Paragraph(s3.notes[0] if s3.notes else "Distress liquidity check", style_table_cell),
        ],
    ]
    t_stress = Table(stress_data, colWidths=[120, 45, 160, 198])
    t_stress.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(t_stress)
    story.append(Spacer(1, 8))

    # Opportunity Cost & Benchmark Parity
    story.append(
        Paragraph("7. Liquid Opportunity Cost & Cash-Flow Parity Benchmarks", style_sec_heading)
    )
    fd_res = base_res.fd_benchmark
    nifty_res = base_res.nifty_benchmark
    opp_data = [
        [
            Paragraph("<b>Benchmark Alternative</b>", style_table_header),
            Paragraph("<b>Annual Compounding</b>", style_table_header),
            Paragraph("<b>Terminal Wealth</b>", style_table_header),
            Paragraph("<b>Real Estate Wealth Advantage</b>", style_table_header),
            Paragraph("<b>IRR Spread vs RE</b>", style_table_header),
        ],
        [
            Paragraph("<b>Real Estate (Base Case)</b>", style_table_cell_bold),
            Paragraph(f"<b>{base_res.equity_irr_pct:.2f}%</b>", style_table_cell_right_bold),
            Paragraph(f"<b>{_inr(base_res.exit.net_exit_proceeds)}</b>", style_table_cell_right_bold),
            Paragraph("<b>Baseline Asset</b>", style_table_cell_bold),
            Paragraph("<b>0.00%</b>", style_table_cell_bold),
        ],
        [
            Paragraph("SBI Term Deposit (Post-Tax)", style_table_cell),
            Paragraph(f"{fd_res.annual_return_pct:.2f}%", style_table_cell_right),
            Paragraph(f"{_inr(fd_res.terminal_wealth_post_tax)}", style_table_cell_right),
            Paragraph(
                f"<b>{'+' if fd_res.net_wealth_advantage_inr >= 0 else ''}{_inr(fd_res.net_wealth_advantage_inr)}</b>",
                style_table_cell_right_bold,
            ),
            Paragraph(f"{fd_res.irr_spread_pct:+.2f}%", style_table_cell_right),
        ],
        [
            Paragraph("Nifty 50 TRI (Post-Tax Equity)", style_table_cell),
            Paragraph(f"{nifty_res.annual_return_pct:.2f}% (13.48% pre-tax CAGR)", style_table_cell_right),
            Paragraph(f"{_inr(nifty_res.terminal_wealth_post_tax)}", style_table_cell_right),
            Paragraph(
                f"<b>{'+' if nifty_res.net_wealth_advantage_inr >= 0 else ''}{_inr(nifty_res.net_wealth_advantage_inr)}</b>",
                style_table_cell_right_bold,
            ),
            Paragraph(f"{nifty_res.irr_spread_pct:+.2f}%", style_table_cell_right),
        ],
    ]
    t_opp = Table(opp_data, colWidths=[140, 95, 95, 115, 78])
    t_opp.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_opp)

    # =========================================================================
    # PAGE 5: SENSITIVITY, VALUE DRIVERS, EXPECTED VALUE & DATA GOVERNANCE
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph("8. Value Drivers, Sensitivity & Expected Value Analysis", style_sec_heading)
    )

    be = compute_break_even_appreciation(property_obj, financing_obj, investor_obj, macro)
    ev = compute_probability_weighted(property_obj, financing_obj, investor_obj, macro, 0.25, 0.50, 0.25)
    decomp = compute_return_decomposition(base_res)
    d_total_cap = decomp.get("total_capital_invested", decomp.get("total_equity_invested", 0.0))
    d_net_cg = decomp.get("net_post_tax_capital_gain", decomp.get("capital_gain", 0.0))
    d_cum_noi = decomp.get("cumulative_noi", decomp.get("total_rent_collected", 0.0))
    d_net_yield = decomp.get("net_rental_yield_pct", base_res.year_1_net_yield_pct)
    d_cum_int_tax = decomp.get("cumulative_interest_and_tax", 0.0)
    d_cum_loan_int = decomp.get("cumulative_loan_interest", 0.0)
    d_cum_op_tax = decomp.get("cumulative_operating_tax", 0.0)
    d_net_w = decomp.get("net_wealth_created", base_res.exit.net_exit_proceeds - d_total_cap)

    sens_data = [
        [
            Paragraph("<b>Decision & Sensitivity Metric</b>", style_table_header),
            Paragraph("<b>Value</b>", style_table_header),
            Paragraph("<b>Analytical Prudence & Strategic Context</b>", style_table_header),
        ],
        [
            Paragraph("Break-Even Appreciation vs SBI FD", style_table_cell_bold),
            Paragraph(
                f"{be['break_even_vs_fd_pct']:.2f}% / year"
                if be['break_even_vs_fd_pct'] is not None
                else "> +30%",
                style_table_cell_bold,
            ),
            Paragraph(
                f"Minimum appreciation required to beat risk-free fixed deposit hurdle ({be['fd_post_tax_pct']:.2f}% post-tax).",
                style_table_cell,
            ),
        ],
        [
            Paragraph("Break-Even Appreciation vs Nifty 50 TRI", style_table_cell_bold),
            Paragraph(
                f"{be['break_even_vs_nifty_pct']:.2f}% / year"
                if be['break_even_vs_nifty_pct'] is not None
                else "> +30%",
                style_table_cell_bold,
            ),
            Paragraph(
                f"Appreciation required to beat equity index compounding ({be['nifty_pct']:.2f}% CAGR).",
                style_table_cell,
            ),
        ],
        [
            Paragraph("Probability-Weighted Terminal Wealth", style_table_cell_bold),
            Paragraph(f"{_inr(ev.expected_terminal_wealth)}", style_table_cell_bold),
            Paragraph(
                "Consolidated expected net terminal corpus (25% Best / 50% Base / 25% Worst likelihood).",
                style_table_cell,
            ),
        ],
        [
            Paragraph("Probability-Weighted Equity IRR", style_table_cell_bold),
            Paragraph(
                f"{ev.expected_equity_irr_pct:.2f}%" if ev.expected_equity_irr_pct else "N/A",
                style_table_cell_bold,
            ),
            Paragraph("Blended annualized return on equity across all scenarios.", style_table_cell),
        ],
        [
            Paragraph("Return Decomposition: Capital Invested", style_table_cell),
            Paragraph(f"{_inr(d_total_cap)}", style_table_cell),
            Paragraph("Initial down payment plus ongoing annual cash flow deficits.", style_table_cell),
        ],
        [
            Paragraph("Post-Tax Net Capital Gain (After All Friction & LTCG)", style_table_cell),
            Paragraph(f"{_inr(d_net_cg)}", style_table_cell),
            Paragraph("Terminal appreciation minus all acquisition closing costs, selling brokerage, and 12.5% LTCG tax.", style_table_cell),
        ],
        [
            Paragraph("Cumulative Net Operating Income (NOI)", style_table_cell),
            Paragraph(f"{_inr(d_cum_noi)}", style_table_cell),
            Paragraph(
                f"True cumulative net rent collected after vacancy, society maintenance, and property tax (Y1 net yield: {d_net_yield:.2f}%).",
                style_table_cell,
            ),
        ],
        [
            Paragraph("Less: Cumulative Loan Interest & Operating Tax", style_table_cell),
            Paragraph(f"-{_inr(d_cum_int_tax)}", style_table_cell),
            Paragraph(f"Total mortgage interest paid ({_inr(d_cum_loan_int)}) and annual income tax liability on property ({_inr(d_cum_op_tax)}).", style_table_cell),
        ],
        [
            Paragraph("<b>Net Wealth Created Above Capital</b>", style_table_cell_bold),
            Paragraph(
                f"<b>{_inr(d_net_w)}</b>",
                style_table_cell_bold,
            ),
            Paragraph("Exact additive reconciliation: Post-Tax Gain + NOI - Loan Interest - Operating Tax.", style_table_cell),
        ],
    ]
    t_sens = Table(sens_data, colWidths=[170, 110, 243])
    t_sens.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(t_sens)
    story.append(Spacer(1, 8))

    # VISUALS 5 & 6: Tornado Chart & Sensitivity Heatmap
    story.append(
        Paragraph("<b>Visual Proof: Value Drivers (Tornado) & Appreciation × Escalation Sensitivity</b>", style_sec_heading_sm)
    )
    tornado_items = compute_tornado_data(property_obj, financing_obj, investor_obj, macro)
    im_tornado = _fig_to_reportlab_image(
        create_tornado_chart(tornado_items),
        width=255,
        height=165,
        fallback_fn=lambda: render_tornado_png(tornado_items),
    )
    appr_axis, esc_axis, irr_grid = compute_sensitivity_grid(property_obj, financing_obj, investor_obj, macro)
    im_heatmap = _fig_to_reportlab_image(
        create_sensitivity_heatmap(
            appr_axis,
            esc_axis,
            irr_grid,
            current_appr_pct=property_obj.appreciation_rate * 100.0,
            current_esc_pct=financing_obj.annual_rent_escalation_pct * 100.0,
        ),
        width=255,
        height=165,
        fallback_fn=lambda: render_heatmap_png(
            appr_axis,
            esc_axis,
            irr_grid,
            current_appr_pct=property_obj.appreciation_rate * 100.0,
            current_esc_pct=financing_obj.annual_rent_escalation_pct * 100.0,
        ),
    )
    if im_tornado and im_heatmap:
        t_vis_sens = Table([[im_tornado, im_heatmap]], colWidths=[261.5, 261.5])
        t_vis_sens.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 0),
            ])
        )
        story.append(t_vis_sens)
        story.append(Spacer(1, 8))

    # VISUAL 7: Probability-Weighted Expected Value Chart
    im_ev = _fig_to_reportlab_image(
        create_expected_value_chart(ev),
        width=523,
        height=150,
        fallback_fn=lambda: render_expected_value_png(ev),
    )
    if im_ev:
        story.append(im_ev)
        story.append(Spacer(1, 8))

    # VISUAL 8: Pune Micro-Market Yield vs Valuation Scatter Matrix
    try:
        merged_df = get_merged_dataset()
        im_scatter = _fig_to_reportlab_image(
            create_micromarket_scatter_chart(merged_df, property_obj.pincode, property_obj.bhk),
            width=523,
            height=220,
            fallback_fn=lambda: render_scatter_png(merged_df, property_obj.pincode, property_obj.bhk),
        )
        if im_scatter:
            story.append(
                Paragraph("<b>Visual Proof: Pune Micro-Market Yield vs Valuation Matrix</b>", style_sec_heading_sm)
            )
            story.append(im_scatter)
            story.append(Spacer(1, 8))
    except Exception as exc:
        _logger.warning("Skipped micro-market scatter chart in PDF: %s", exc)

    # Data Provenance & Methodology Governance
    heading_sec9 = Paragraph("9. Institutional Data Provenance & Governance Audit", style_sec_heading)
    loc_d = loc_row if isinstance(loc_row, dict) else (loc_row.to_dict() if hasattr(loc_row, "to_dict") else {})
    zone_d = zone_row if isinstance(zone_row, dict) else (zone_row.to_dict() if hasattr(zone_row, "to_dict") else {})
    prov_data = [
        [
            Paragraph("<b>Benchmark Parameter</b>", style_table_header),
            Paragraph("<b>Value</b>", style_table_header),
            Paragraph("<b>Data Basis Tag</b>", style_table_header),
            Paragraph("<b>Institutional Classification</b>", style_table_header),
        ],
        [
            Paragraph("Ready Reckoner Rate (Floor)", style_table_cell),
            Paragraph(f"INR {property_obj.rr_rate_primary_sqft:,.0f}/sq.ft.", style_table_cell_bold),
            Paragraph(str(loc_d.get("rr_rate_primary_basis", "IGR_SL_PRIMARY")), style_table_cell),
            Paragraph("Official Statutory (IGR Maharashtra 2026)", style_table_cell),
        ],
        [
            Paragraph("Inventory Overhang (QTS)", style_table_cell),
            Paragraph(f"{property_obj.quarters_to_sell:.1f} Quarters", style_table_cell_bold),
            Paragraph(str(zone_d.get("overhang_data_basis", "INSTITUTIONAL_REPORT")), style_table_cell),
            Paragraph("Institutional Market Study (Knight Frank H1 2026)", style_table_cell),
        ],
        [
            Paragraph("Under-Construction Delay Risk", style_table_cell),
            Paragraph(f"{property_obj.avg_uc_delay_months:.0f} Months", style_table_cell_bold),
            Paragraph(str(zone_d.get("delay_data_basis", "UNDERWRITING_ASSUMPTION")), style_table_cell),
            Paragraph("Calibrated Risk Assumption", style_table_cell),
        ],
        [
            Paragraph("Tenant Vacancy Downtime", style_table_cell),
            Paragraph(f"{property_obj.vacancy_months_avg:.1f} Months/yr", style_table_cell_bold),
            Paragraph(str(zone_d.get("vacancy_data_basis", "UNDERWRITING_ASSUMPTION")), style_table_cell),
            Paragraph("Calibrated Prudence Assumption", style_table_cell),
        ],
    ]
    t_prov = Table(prov_data, colWidths=[130, 85, 145, 163])
    t_prov.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
            ("GRID", (0, 0), (-1, -1), 0.5, c_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
            ("PADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    sec9_elements = [heading_sec9, Spacer(1, 4), t_prov]

    # Contextual advisories for Cantonment or Sub-1.0 QTS
    if property_obj.jurisdiction.lower() == "cantonment":
        sec9_elements.append(Spacer(1, 3))
        sec9_elements.append(
            Paragraph("<b>Cantonment Title Risk Advisory:</b> Verify freehold vs. Old Grant / leasehold tenure and Pune Cantonment Board FSI/redevelopment restrictions.", style_table_cell)
        )
    if property_obj.quarters_to_sell < 1.0:
        sec9_elements.append(Spacer(1, 3))
        sec9_elements.append(
            Paragraph("<b>Liquidity Proxy Note:</b> Sub-1.0 QTS reflects low primary developer inventory in a mature micro-market; secondary resale liquidation typically takes 3–6 months.", style_table_cell)
        )

    story.append(KeepTogether(sec9_elements))

    # =========================================================================
    # SECTION 10 (CONDITIONAL): ALL-ZONES TOP 3 LOCALITY RECOMMENDER REPORT
    # =========================================================================
    if all_zone_recommendations:
        def _pdf_safe(txt: str) -> str:
            return txt.replace("₹", "INR ")

        story.append(PageBreak())
        story.append(
            Paragraph(
                "10. Smart Micro-Market Finder: Top Ranked Pune Pincodes (All Zones)",
                style_sec_heading,
            )
        )
        story.append(
            Paragraph(
                f"Full-market quantitative screen across {len(all_zone_recommendations)} Pune localities for "
                f"<b>{property_obj.bhk.replace('_', ' ')} ({property_obj.use_case.replace('_', ' ').title()})</b>, "
                f"calibrated to Liquid Savings of <b>{_inr(investor_obj.total_liquid_savings_inr)}</b>, "
                f"Net Monthly Income of <b>{_inr(investor_obj.net_monthly_income_inr)}</b>, "
                f"<b>{investor_obj.tax_regime.upper()}</b> Tax Regime, and a <b>{h}-Year</b> horizon.",
                style_body,
            )
        )
        story.append(Spacer(1, 8))

        top3 = all_zone_recommendations[:3]
        story.append(
            Paragraph("<b>Top 3 Ranked Pincodes & Areas Aligned With Your Finances & Profit Targets</b>", style_sec_heading_sm)
        )
        top3_rows = [
            [
                Paragraph("<b>Rank / Score</b>", style_table_header),
                Paragraph("<b>Pincode & Area</b>", style_table_header),
                Paragraph("<b>Verdict</b>", style_table_header),
                Paragraph("<b>Ticket / Upfront</b>", style_table_header),
                Paragraph("<b>Base / Real IRR</b>", style_table_header),
                Paragraph("<b>Net Profit</b>", style_table_header),
                Paragraph("<b>Yr-1 CF/mo & DSCR</b>", style_table_header),
                Paragraph("<b>Mkt/RR & Liq</b>", style_table_header),
            ]
        ]
        for rec in top3:
            v_col = "#059669" if rec.verdict_status == "GO" else ("#d97706" if rec.verdict_status == "CAUTION" else "#dc2626")
            b_irr = f"{rec.base_irr_pct:.2f}%" if rec.base_irr_pct is not None else "N/A"
            r_irr = f"{rec.real_irr_pct:.2f}%" if rec.real_irr_pct is not None else "N/A"
            dscr_s = f"{rec.dscr:.2f}x" if rec.dscr is not None else "Cash"
            top3_rows.append([
                Paragraph(f"<b>#{rec.rank}</b> ({rec.alignment_score:.0f}/100)", style_table_cell_bold),
                Paragraph(f"<b>{rec.pincode} - {rec.locality_name}</b><br/>{rec.pune_zone} ({rec.jurisdiction})", style_table_cell),
                Paragraph(f"<font color='{v_col}'><b>{rec.verdict_status}</b></font>", style_table_cell_bold),
                Paragraph(f"{_inr(rec.ticket_price_inr)}<br/>Eq: {_inr(rec.upfront_cash_inr)}", style_table_cell_right),
                Paragraph(f"<b>{b_irr}</b><br/>({r_irr} real)", style_table_cell_right),
                Paragraph(f"<b>{_inr(rec.net_wealth_created_inr)}</b>", style_table_cell_right_bold),
                Paragraph(f"{_inr(rec.year_1_monthly_cf_inr)}/mo<br/>DSCR: {dscr_s}", style_table_cell_right),
                Paragraph(f"{rec.market_to_rr_ratio:.2f}x RR<br/>{rec.liquidity_tier.upper()} ({rec.quarters_to_sell:.1f}Q)", style_table_cell_right),
            ])

        t_top3 = Table(top3_rows, colWidths=[56, 108, 44, 72, 58, 63, 66, 56])
        t_top3.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
                ("GRID", (0, 0), (-1, -1), 0.5, c_border),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        story.append(t_top3)
        story.append(Spacer(1, 10))

        story.append(
            Paragraph("<b>Detailed Financial & Market Research Rationale (Top 3)</b>", style_sec_heading_sm)
        )
        for rec in top3:
            rationale_html = "<br/>".join(f"&bull; {_pdf_safe(r)}" for r in rec.alignment_reasons)
            card_tbl = Table(
                [
                    [
                        Paragraph(
                            f"<b>Rank #{rec.rank}: {rec.locality_name} (PIN {rec.pincode} · {rec.pune_zone} Zone) — "
                            f"Score {rec.alignment_score:.1f}/100 · Verdict: {rec.verdict_status}</b>",
                            style_table_cell_bold,
                        )
                    ],
                    [Paragraph(rationale_html, style_table_cell)],
                ],
                colWidths=[523],
            )
            card_tbl.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.75, c_border),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, c_border),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ])
            )
            story.append(KeepTogether([card_tbl, Spacer(1, 6)]))

        if len(all_zone_recommendations) > 3:
            story.append(Spacer(1, 4))
            story.append(
                Paragraph("<b>Top 10 Pune Shortlist Summary (Out of All Evaluated Pincodes)</b>", style_sec_heading_sm)
            )
            short_rows = [
                [
                    Paragraph("<b>Rank</b>", style_table_header),
                    Paragraph("<b>Pincode</b>", style_table_header),
                    Paragraph("<b>Locality / Area</b>", style_table_header),
                    Paragraph("<b>Zone</b>", style_table_header),
                    Paragraph("<b>Score</b>", style_table_header),
                    Paragraph("<b>Verdict</b>", style_table_header),
                    Paragraph("<b>Ticket Price</b>", style_table_header),
                    Paragraph("<b>Base IRR</b>", style_table_header),
                    Paragraph("<b>Net Wealth</b>", style_table_header),
                ]
            ]
            for rec in all_zone_recommendations[:10]:
                b_irr = f"{rec.base_irr_pct:.2f}%" if rec.base_irr_pct is not None else "N/A"
                short_rows.append([
                    Paragraph(f"#{rec.rank}", style_table_cell_bold),
                    Paragraph(rec.pincode, style_table_cell),
                    Paragraph(rec.locality_name, style_table_cell),
                    Paragraph(rec.pune_zone, style_table_cell),
                    Paragraph(f"{rec.alignment_score:.1f}", style_table_cell_right),
                    Paragraph(rec.verdict_status, style_table_cell_bold),
                    Paragraph(_inr(rec.ticket_price_inr), style_table_cell_right),
                    Paragraph(b_irr, style_table_cell_right),
                    Paragraph(_inr(rec.net_wealth_created_inr), style_table_cell_right),
                ])
            t_short = Table(short_rows, colWidths=[32, 46, 130, 56, 38, 46, 65, 48, 62])
            t_short.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), c_slate_dark),
                    ("GRID", (0, 0), (-1, -1), 0.5, c_border),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_slate_light]),
                    ("PADDING", (0, 0), (-1, -1), 3.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ])
            )
            story.append(t_short)

    doc.build(story)
    return buf.getvalue()
