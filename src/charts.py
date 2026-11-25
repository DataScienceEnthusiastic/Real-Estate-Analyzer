from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.models import (
    AcquisitionCostBreakdown,
    CashFlowYear,
    ScenarioMatrixResult,
    UnderwritingResult,
)


def format_inr(value: float, precision: int = 2) -> str:
    """Formats values in standard Indian denominations (Crores, Lakhs, or Thousands)."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 10_000_000.0:
        cr = abs_val / 10_000_000.0
        return f"{sign}₹{cr:.{precision}f} Cr"
    elif abs_val >= 100_000.0:
        lakh = abs_val / 100_000.0
        return f"{sign}₹{lakh:.{precision}f} L"
    elif abs_val >= 1_000.0:
        return f"{sign}₹{abs_val:,.0f}"
    else:
        return f"{sign}₹{abs_val:.{precision}f}"


def create_acquisition_waterfall_chart(acquisition: AcquisitionCostBreakdown) -> go.Figure:
    """Generates an executive waterfall chart for statutory and all-in property acquisition costs."""
    labels = ["Agreement Value", "Stamp Duty", "Registration Fee"]
    values = [
        acquisition.agreement_value_inr,
        acquisition.stamp_duty_inr,
        acquisition.registration_fee_inr,
    ]

    if acquisition.gst_inr > 0:
        labels.append("GST")
        values.append(acquisition.gst_inr)

    if acquisition.buy_brokerage_inr > 0:
        labels.append("Brokerage (+GST)")
        values.append(acquisition.buy_brokerage_inr)

    if acquisition.interiors_and_misc_inr > 0:
        labels.append("Interiors & Misc")
        values.append(acquisition.interiors_and_misc_inr)

    labels.append("All-In Total")
    values.append(acquisition.all_in_acquisition_cost)

    measure = ["relative"] * (len(labels) - 1) + ["total"]
    text_labels = [format_inr(v) for v in values]

    fig = go.Figure(
        go.Waterfall(
            name="Acquisition Costs",
            orientation="v",
            measure=measure,
            x=labels,
            textposition="outside",
            text=text_labels,
            textfont={"color": "#f8fafc", "size": 11, "family": "Inter, system-ui, sans-serif"},
            y=values,
            connector={"line": {"color": "#475569", "width": 1.5}},
            decreasing={"marker": {"color": "#f43f5e"}},
            increasing={"marker": {"color": "#38bdf8"}},
            totals={"marker": {"color": "#34d399"}},
            hovertemplate="<b>%{x}</b><br>Amount: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "<b>All-In Acquisition Cost Waterfall</b>", "font": {"size": 16, "color": "#f8fafc"}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        yaxis={
            "title": "Cost (INR)",
            "tickformat": ",.0f",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        xaxis={"title": "", "color": "#94a3b8", "tickfont": {"color": "#cbd5e1", "size": 11}},
        showlegend=False,
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=380,
    )
    return fig


def create_cash_flow_trajectory_chart(annual_cfs: list[CashFlowYear]) -> go.Figure:
    """Generates a combined bar and line chart of annual net cash flows and cumulative deficit/surplus."""
    years = [f"Yr {cf.year}" for cf in annual_cfs]
    cfs = [cf.net_cash_flow for cf in annual_cfs]

    cumulative_drain: list[float] = []
    cum = 0.0
    for cf in cfs:
        cum += cf
        cumulative_drain.append(cum)

    colors = ["#34d399" if cf >= 0 else "#f43f5e" for cf in cfs]
    text_cfs = [format_inr(cf) for cf in cfs]

    fig = go.Figure()

    # Annual bar chart
    fig.add_trace(
        go.Bar(
            x=years,
            y=cfs,
            name="Annual Net Cash Flow",
            marker_color=colors,
            text=text_cfs,
            textposition="auto",
            textfont={"color": "#ffffff", "size": 11, "family": "Inter, system-ui, sans-serif"},
            hovertemplate="<b>%{x} Net Cash Flow</b>: %{text}<extra></extra>",
        )
    )

    # Cumulative line chart
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative_drain,
            name="Cumulative Cash Flow",
            mode="lines+markers",
            line={"color": "#818cf8", "width": 3},
            marker={"size": 7, "color": "#a5b4fc"},
            hovertemplate="<b>%{x} Cumulative</b>: INR %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "<b>Operating Cash Flow & Cumulative Trajectory</b>", "font": {"size": 16, "color": "#f8fafc"}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        yaxis={
            "title": "INR",
            "tickformat": ",.0f",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        xaxis={"color": "#94a3b8", "tickfont": {"color": "#cbd5e1", "size": 11}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#cbd5e1", "size": 11},
            "bgcolor": "rgba(17, 24, 39, 0.7)",
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=360,
    )
    return fig


def create_amortization_chart(annual_sched: list[dict[str, float]]) -> go.Figure:
    """Visualizes principal vs interest amortisation along with remaining loan balance."""
    years = [f"Yr {row['year']}" for row in annual_sched]
    interest = [row["interest_paid"] for row in annual_sched]
    principal = [row["principal_paid"] for row in annual_sched]
    balance = [row["closing_balance"] for row in annual_sched]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=principal,
            name="Principal Paid",
            marker_color="#38bdf8",
            hovertemplate="Principal: INR %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=years,
            y=interest,
            name="Interest Paid",
            marker_color="#fbbf24",
            hovertemplate="Interest: INR %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=balance,
            name="Remaining Loan Balance",
            mode="lines+markers",
            line={"color": "#f43f5e", "width": 2.5, "dash": "dot"},
            marker={"size": 6, "color": "#fda4af"},
            yaxis="y2",
            hovertemplate="Loan Balance: INR %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "<b>Loan Amortization & Principal Reduction</b>", "font": {"size": 16, "color": "#f8fafc"}},
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 40, "t": 50, "b": 30},
        yaxis={
            "title": "Annual Outflow (INR)",
            "tickformat": ",.0f",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        yaxis2={
            "title": "Remaining Debt (INR)",
            "overlaying": "y",
            "side": "right",
            "tickformat": ",.0f",
            "showgrid": False,
            "color": "#f87171",
            "tickfont": {"color": "#f87171", "size": 11},
        },
        xaxis={"color": "#94a3b8", "tickfont": {"color": "#cbd5e1", "size": 11}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#cbd5e1", "size": 11},
            "bgcolor": "rgba(17, 24, 39, 0.7)",
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=360,
    )
    return fig


def create_wealth_trajectory_chart(matrix: ScenarioMatrixResult, holding_period_years: int) -> go.Figure:
    """
    Plots the wealth trajectory comparison:
    Real Estate Base Net Equity vs SBI Fixed Deposit (Post-tax) vs Nifty 50 TRI (Post-tax).
    """
    years = [f"Yr {y}" for y in range(0, holding_period_years + 1)]
    base_res = matrix.base

    # RE Net realizable equity: Year 0 = upfront equity, Year 1..H = net_realizable_equity
    if hasattr(base_res, "net_realizable_equity_trajectory") and base_res.net_realizable_equity_trajectory:
        re_equity = base_res.net_realizable_equity_trajectory
    else:
        re_equity = [base_res.loan.required_upfront_cash] + [cf.net_realizable_equity for cf in base_res.annual_cash_flows]

    fd_wealth = base_res.fd_benchmark.trajectory
    nifty_wealth = base_res.nifty_benchmark.trajectory

    # Worst case RE net realizable equity for downside envelope
    if hasattr(matrix.worst, "net_realizable_equity_trajectory") and matrix.worst.net_realizable_equity_trajectory:
        worst_equity = matrix.worst.net_realizable_equity_trajectory
    else:
        worst_equity = [matrix.worst.loan.required_upfront_cash] + [cf.net_realizable_equity for cf in matrix.worst.annual_cash_flows]

    fig = go.Figure()

    # Worst case envelope
    fig.add_trace(
        go.Scatter(
            x=years,
            y=worst_equity,
            name="Real Estate (Worst Case Stress)",
            mode="lines",
            line={"color": "#f87171", "width": 1.8, "dash": "dash"},
            hovertemplate="RE Worst: INR %{y:,.0f}<extra></extra>",
        )
    )

    # Base case RE
    fig.add_trace(
        go.Scatter(
            x=years,
            y=re_equity,
            name="Real Estate (Base Case)",
            mode="lines+markers",
            line={"color": "#34d399", "width": 3.5},
            marker={"size": 8, "color": "#10b981"},
            hovertemplate="RE Base Equity: INR %{y:,.0f}<extra></extra>",
        )
    )

    # SBI FD
    fig.add_trace(
        go.Scatter(
            x=years,
            y=fd_wealth,
            name="SBI FD (Post-Tax Parity)",
            mode="lines+markers",
            line={"color": "#fbbf24", "width": 2.5, "dash": "dot"},
            marker={"size": 6, "color": "#f59e0b"},
            hovertemplate="SBI FD: INR %{y:,.0f}<extra></extra>",
        )
    )

    # Nifty 50 TRI
    fig.add_trace(
        go.Scatter(
            x=years,
            y=nifty_wealth,
            name="Nifty 50 TRI (Post-Tax Parity)",
            mode="lines+markers",
            line={"color": "#818cf8", "width": 2.5},
            marker={"size": 6, "color": "#6366f1"},
            hovertemplate="Nifty 50: INR %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "<b>Wealth Creation Trajectory: Real Estate vs Liquid Benchmarks</b>", "font": {"size": 16, "color": "#f8fafc"}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        yaxis={
            "title": "Net Wealth (INR)",
            "tickformat": ",.0f",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        xaxis={"color": "#94a3b8", "tickfont": {"color": "#cbd5e1", "size": 11}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#cbd5e1", "size": 11},
            "bgcolor": "rgba(17, 24, 39, 0.7)",
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=420,
    )
    return fig


def create_micromarket_scatter_chart(
    merged_df: pd.DataFrame,
    selected_pincode: str,
    selected_bhk: str,
) -> go.Figure:
    """
    Plots interactive scatter across all 43 Pune localities x 4 BHKs:
    Price/sqft carpet equivalent vs implied gross rental yield, sized by ticket price and colored by zone.
    """
    df = merged_df.copy()
    # Filter out null yields for scatter plot display
    plot_df = df.dropna(subset=["implied_gross_rental_yield_pct", "price_per_sqft_carpet_equiv"]).copy()

    plot_df["ticket_cr"] = plot_df["est_ticket_price_inr"] / 10_000_000.0
    plot_df["formatted_ticket"] = plot_df["est_ticket_price_inr"].apply(format_inr)
    plot_df["formatted_rent"] = plot_df["median_monthly_rent"].apply(lambda v: format_inr(v) if pd.notna(v) else "N/A")

    fig = px.scatter(
        plot_df,
        x="price_per_sqft_carpet_equiv",
        y="implied_gross_rental_yield_pct",
        color="pune_zone",
        size="est_ticket_price_inr",
        size_max=22,
        hover_name="locality_name",
        hover_data={
            "bhk": True,
            "pune_zone": True,
            "jurisdiction": True,
            "formatted_ticket": True,
            "price_per_sqft_carpet_equiv": ":,.0f",
            "implied_gross_rental_yield_pct": ":.2f",
            "formatted_rent": True,
            "est_ticket_price_inr": False,
        },
        labels={
            "price_per_sqft_carpet_equiv": "Carpet Price (₹ / sq.ft.)",
            "implied_gross_rental_yield_pct": "Gross Rental Yield (%)",
            "pune_zone": "Pune Zone",
        },
        color_discrete_map={
            "Central": "#c084fc",
            "East": "#38bdf8",
            "West": "#34d399",
            "South": "#fbbf24",
            "North_PCMC": "#fb7185",
        },
    )

    # Highlight currently selected unit
    pincode_clean = str(selected_pincode).strip().zfill(6)
    sel = plot_df[(plot_df["pincode"] == pincode_clean) & (plot_df["bhk"] == selected_bhk)]
    if not sel.empty:
        row = sel.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row["price_per_sqft_carpet_equiv"]],
                y=[row["implied_gross_rental_yield_pct"]],
                mode="markers+text",
                name="SELECTED DEAL",
                text=[f"★ {row['locality_name']} ({row['bhk']})"],
                textposition="top center",
                textfont={"size": 13, "color": "#facc15", "family": "Inter, system-ui, sans-serif"},
                marker={
                    "size": 26,
                    "color": "#facc15",
                    "symbol": "star",
                    "line": {"color": "#ffffff", "width": 2.5},
                },
                hovertemplate=(
                    f"<b>SELECTED DEAL: {row['locality_name']} ({row['bhk']})</b><br>"
                    f"Price/sqft: ₹{row['price_per_sqft_carpet_equiv']:,.0f}<br>"
                    f"Gross Yield: {row['implied_gross_rental_yield_pct']:.2f}%<br>"
                    f"Ticket: {format_inr(row['est_ticket_price_inr'])}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": "<b>Pune Micro-Market Yield vs Valuation Matrix (43 Localities × 4 BHKs)</b>", "font": {"size": 16, "color": "#f8fafc"}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        xaxis={
            "title": "Carpet Price (₹ / sq.ft.)",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "tickformat": ",.0f",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        yaxis={
            "title": "Gross Rental Yield (%)",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "ticksuffix": "%",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#cbd5e1", "size": 11},
            "bgcolor": "rgba(17, 24, 39, 0.7)",
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=500,
    )
    return fig


def create_sensitivity_heatmap(
    appr_vals_pct: np.ndarray,
    esc_vals_pct: np.ndarray,
    irr_matrix: np.ndarray,
) -> go.Figure:
    """Heat-map of equity IRR across appreciation × rent escalation."""
    zmin = float(np.nanmin(irr_matrix)) if not np.all(np.isnan(irr_matrix)) else 0.0
    zmax = float(np.nanmax(irr_matrix)) if not np.all(np.isnan(irr_matrix)) else 1.0
    
    # High-contrast color scale: Red (loss/low) -> Amber -> Green (high returns)
    colorscale = [
        [0.0, "#991b1b"],
        [0.2, "#dc2626"],
        [0.4, "#f59e0b"],
        [0.6, "#84cc16"],
        [0.8, "#10b981"],
        [1.0, "#059669"],
    ]

    fig = go.Figure(
        go.Heatmap(
            x=esc_vals_pct,
            y=appr_vals_pct,
            z=irr_matrix,
            colorscale=colorscale,
            zmid=0.0,
            zmin=zmin,
            zmax=zmax,
            colorbar={
                "title": {"text": "Equity IRR (%)", "font": {"color": "#f8fafc", "size": 12}},
                "tickformat": ".1f",
                "tickfont": {"color": "#94a3b8", "size": 11},
            },
            hovertemplate="Appreciation %{y:.1f}% · Rent esc. %{x:.1f}%<br>IRR: <b>%{z:.2f}%</b><extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "<b>Equity IRR Sensitivity: Appreciation × Rent Escalation</b>",
               "font": {"size": 16, "color": "#f8fafc"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 50, "b": 60},
        xaxis={
            "title": "Rent Escalation (%)",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        yaxis={
            "title": "Annual Appreciation (%)",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=460,
    )
    return fig


def create_tornado_chart(items: list[dict]) -> go.Figure:
    """Tornado / impact chart of single-driver IRR swings."""
    base_val = items[0]["base"] if items else 0.0
    fig = go.Figure()
    for it in items:
        fig.add_trace(
            go.Bar(
                y=[it["label"]],
                x=[it["high"] - it["low"]],
                base=[it["low"]],
                orientation="h",
                marker_color="#38bdf8",
                opacity=0.85,
                text=[f"{it['low']:.1f}% ↔ {it['high']:.1f}%"],
                textposition="outside",
                textfont={"color": "#f8fafc", "size": 11, "family": "Inter, system-ui, sans-serif"},
                hovertemplate=f"<b>{it['label']}</b><br>Low: {it['low']:.2f}% → High: {it['high']:.2f}%<extra></extra>",
            )
        )
    fig.add_vline(x=base_val, line_dash="dash", line_color="#f8fafc", line_width=2)
    fig.add_annotation(
        x=base_val, y=1.05, text=f"Base IRR {base_val:.2f}%",
        showarrow=False, font={"size": 11, "color": "#f8fafc", "family": "Inter, system-ui, sans-serif"},
        xanchor="left", yanchor="bottom",
    )
    fig.update_layout(
        barmode="overlay",
        title={"text": "<b>Value Drivers — One-at-a-Time Impact (Equity IRR)</b>",
               "font": {"size": 16, "color": "#f8fafc"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 40, "t": 50, "b": 30},
        xaxis={
            "title": "Equity IRR (%)",
            "ticksuffix": "%",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        yaxis={"autorange": "reversed", "color": "#f8fafc", "tickfont": {"color": "#f8fafc", "size": 11}},
        showlegend=False,
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=300,
    )
    return fig


def create_expected_value_chart(ev) -> go.Figure:
    """Bar chart of terminal wealth across Best / Base / Worst / Expected."""
    labels = ["Worst case", "Base case", "Best case", "★ Expected value"]
    values = [
        ev.worst_terminal_wealth,
        ev.base_terminal_wealth,
        ev.best_terminal_wealth,
        ev.expected_terminal_wealth,
    ]
    colors = ["#f43f5e", "#38bdf8", "#34d399", "#a855f7"]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[format_inr(v) for v in values],
            textposition="outside",
            textfont={"color": "#f8fafc", "size": 12, "family": "Inter, system-ui, sans-serif"},
            hovertemplate="%{y}: <b>%{text}</b><extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "<b>Probability-Weighted Expected Terminal Wealth</b>",
               "font": {"size": 16, "color": "#f8fafc"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 40, "t": 50, "b": 30},
        xaxis={
            "title": "Net Terminal Wealth (INR)",
            "tickformat": ",.0f",
            "gridcolor": "#1e293b",
            "zerolinecolor": "#334155",
            "color": "#94a3b8",
            "tickfont": {"color": "#94a3b8", "size": 11},
        },
        yaxis={"autorange": "reversed", "color": "#f8fafc", "tickfont": {"color": "#f8fafc", "size": 12}},
        showlegend=False,
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#ffffff", "family": "Inter, system-ui, sans-serif"}},
        height=280,
    )
    return fig


def _format_inr_tick(val: float) -> str:
    """Formats numeric values into concise INR tick labels for charts (e.g., INR 1.2 Cr, INR 40 L)."""
    abs_v = abs(val)
    sign = "-" if val < 0 else ""
    if abs_v >= 10_000_000:
        cr = abs_v / 10_000_000
        return f"{sign}INR {cr:.2f}".rstrip("0").rstrip(".") + " Cr"
    elif abs_v >= 100_000:
        lakh = abs_v / 100_000
        return f"{sign}INR {lakh:.1f}".rstrip("0").rstrip(".") + " L"
    elif abs_v >= 1_000:
        k = abs_v / 1_000
        return f"{sign}INR {k:.0f}k"
    else:
        return f"{sign}INR {abs_v:.0f}"


def prepare_fig_for_pdf(fig: go.Figure) -> go.Figure:
    """Prepares a Plotly Figure for high-resolution, unclipped PDF export with calibrated margins and typography."""
    fig_pdf = go.Figure(fig)

    title_text = ""
    if fig_pdf.layout.title and fig_pdf.layout.title.text:
        title_text = str(fig_pdf.layout.title.text)

    is_tornado = "Tornado" in title_text or any(
        isinstance(t, go.Bar) and getattr(t, "orientation", None) == "h" and "↔" in str(getattr(t, "text", ""))
        for t in fig_pdf.data
    )
    is_expected_value = "Expected" in title_text or "Terminal Wealth" in title_text or any(
        isinstance(t, go.Bar) and getattr(t, "orientation", None) == "h" and any("Expected" in str(lbl) for lbl in (getattr(t, "y", None) or []))
        for t in fig_pdf.data
    )
    is_waterfall = "Acquisition" in title_text or any(isinstance(t, go.Waterfall) for t in fig_pdf.data)

    if is_tornado or is_expected_value:
        fig_pdf.update_layout(
            width=960,
            height=480,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#111827",
            font={"family": "Helvetica, Arial, sans-serif", "color": "#f8fafc", "size": 11},
            margin=dict(l=180, r=85, t=75, b=60),
            legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
        )
        max_x = 0.0
        for t in fig_pdf.data:
            x_vals = getattr(t, "x", None)
            base_vals = getattr(t, "base", None)
            if x_vals is not None:
                for idx, x_val in enumerate(x_vals):
                    try:
                        b = base_vals[idx] if (base_vals is not None and idx < len(base_vals)) else (base_vals or 0.0)
                        max_x = max(max_x, float(b) + float(x_val))
                    except Exception:
                        max_x = max(max_x, float(x_val))
        if max_x > 0:
            fig_pdf.update_xaxes(range=[0, max_x * 1.18])
            if max_x >= 100_000:
                tick_vals = np.linspace(0, max_x * 1.18, 6)
                fig_pdf.update_xaxes(
                    tickmode="array",
                    tickvals=tick_vals.tolist(),
                    ticktext=[_format_inr_tick(v) for v in tick_vals],
                )
    else:
        fig_pdf.update_layout(
            width=960,
            height=480,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#111827",
            font={"family": "Helvetica, Arial, sans-serif", "color": "#f8fafc", "size": 11},
            margin=dict(l=115, r=65, t=95, b=85),
            legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
        )
        if is_waterfall:
            fig_pdf.update_layout(
                title={"text": "<b>All-In Acquisition Cost Waterfall</b>", "font": {"size": 16, "color": "#f8fafc"}}
            )

        max_y = 0.0
        min_y = 0.0
        for t in fig_pdf.data:
            y_vals = getattr(t, "y", None)
            if y_vals is not None:
                for y_val in y_vals:
                    try:
                        f_y = float(y_val)
                        max_y = max(max_y, f_y)
                        min_y = min(min_y, f_y)
                    except Exception:
                        pass
        if max_y >= 100_000:
            start_y = 0.0 if min_y >= 0 else min_y
            tick_vals = np.linspace(start_y, max_y, 6)
            fig_pdf.update_yaxes(
                tickmode="array",
                tickvals=tick_vals.tolist(),
                ticktext=[_format_inr_tick(v) for v in tick_vals],
            )

    return fig_pdf

