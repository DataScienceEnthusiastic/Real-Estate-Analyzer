"""Pure-Python Matplotlib fallback chart renderer for PDF generation.

Used automatically when Kaleido/Chromium is unavailable (e.g. Streamlit Cloud
Linux containers without a system Chrome binary), guaranteeing all 8 dashboard
charts render cleanly into the institutional PDF memorandum.
"""
from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import (
    AcquisitionCostBreakdown,
    CashFlowYear,
    ExpectedValueResult,
    ScenarioMatrixResult,
)


def _fmt_inr_short(val: float) -> str:
    abs_v = abs(val)
    sign = "-" if val < 0 else ""
    if abs_v >= 10_000_000:
        return f"{sign}INR {abs_v / 10_000_000:.2f} Cr"
    if abs_v >= 100_000:
        return f"{sign}INR {abs_v / 100_000:.1f} L"
    if abs_v >= 1_000:
        return f"{sign}INR {abs_v / 1_000:.0f}k"
    return f"{sign}INR {abs_v:.0f}"


def _styled_fig(figsize=(8.0, 4.0)):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(color="#1e293b", linestyle="-", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout(pad=1.1)
    fig.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def render_waterfall_png(acquisition: AcquisitionCostBreakdown) -> bytes:
    labels = ["Agreement", "Stamp Duty", "Reg. Fee"]
    values = [acquisition.agreement_value_inr, acquisition.stamp_duty_inr, acquisition.registration_fee_inr]
    if acquisition.gst_inr > 0:
        labels.append("GST")
        values.append(acquisition.gst_inr)
    if acquisition.buy_brokerage_inr > 0:
        labels.append("Brokerage")
        values.append(acquisition.buy_brokerage_inr)
    if acquisition.interiors_and_misc_inr > 0:
        labels.append("Interiors")
        values.append(acquisition.interiors_and_misc_inr)
    labels.append("All-In Total")
    values.append(acquisition.all_in_acquisition_cost)

    bottoms = []
    running = 0.0
    for i, v in enumerate(values):
        if i == len(values) - 1:
            bottoms.append(0.0)
        else:
            bottoms.append(running)
            running += v

    bar_colors = ["#38bdf8"] * (len(values) - 1) + ["#34d399"]
    fig, ax = _styled_fig()
    x = np.arange(len(labels))
    bars = ax.bar(x, values, bottom=bottoms, color=bar_colors, width=0.55)
    for idx, (bar, v) in enumerate(zip(bars, values)):
        top = bottoms[idx] + v
        ax.text(bar.get_x() + bar.get_width() / 2, top, _fmt_inr_short(v), ha="center", va="bottom", color="#f8fafc", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#cbd5e1", fontsize=8)
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))
    ax.set_title("All-In Acquisition Cost Waterfall", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_cash_flow_png(annual_cfs: list[CashFlowYear]) -> bytes:
    years = [f"Yr {c.year}" for c in annual_cfs]
    cfs = [c.net_cash_flow for c in annual_cfs]
    cum = np.cumsum(cfs).tolist()
    colors = ["#34d399" if v >= 0 else "#f43f5e" for v in cfs]

    fig, ax = _styled_fig()
    x = np.arange(len(years))
    ax.bar(x, cfs, color=colors, width=0.55, label="Yearly Cash Flow")
    ax.plot(x, cum, color="#818cf8", marker="o", linewidth=2.2, markersize=4.5, label="Cumulative Cash Flow")
    ax.axhline(0, color="#475569", linewidth=0.9)
    be_idx = next((i for i, v in enumerate(cfs) if v >= 0), None)
    if be_idx is not None:
        ax.axvline(be_idx, color="#34d399", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(years, color="#cbd5e1", fontsize=7.5)
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))
    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7.5, loc="upper left")
    ax.set_title("Operating Cash Flow & Cumulative Trajectory", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_amortization_png(annual_sched: list[dict[str, float]]) -> bytes:
    years = [f"Yr {r['year']}" for r in annual_sched]
    prin = [r["principal_paid"] for r in annual_sched]
    intr = [r["interest_paid"] for r in annual_sched]
    bal = [r["closing_balance"] for r in annual_sched]

    fig, ax = _styled_fig()
    x = np.arange(len(years))
    ax.bar(x, prin, color="#38bdf8", width=0.55, label="Principal Paid")
    ax.bar(x, intr, bottom=prin, color="#fbbf24", width=0.55, label="Interest Paid")
    ax.set_xticks(x)
    ax.set_xticklabels(years, color="#cbd5e1", fontsize=7.5)
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))

    ax2 = ax.twinx()
    ax2.plot(x, bal, color="#f43f5e", linestyle=":", marker="o", linewidth=2.0, markersize=4, label="Loan Balance")
    ax2.tick_params(colors="#f87171", labelsize=7.5)
    ax2.yaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))
    ax2.spines["right"].set_color("#334155")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#111827", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7, loc="upper right")
    ax.set_title("Loan Amortization & Principal Reduction", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_wealth_trajectory_png(matrix: ScenarioMatrixResult, holding_years: int) -> bytes:
    years = [f"Yr {y}" for y in range(holding_years + 1)]
    base = matrix.base
    re_base = base.net_realizable_equity_trajectory or ([base.loan.required_upfront_cash] + [c.net_realizable_equity for c in base.annual_cash_flows])
    re_worst = matrix.worst.net_realizable_equity_trajectory or ([matrix.worst.loan.required_upfront_cash] + [c.net_realizable_equity for c in matrix.worst.annual_cash_flows])
    fd = base.fd_benchmark.trajectory
    nifty = base.nifty_benchmark.trajectory

    fig, ax = _styled_fig((8.5, 4.1))
    x = np.arange(len(years))
    ax.plot(x, re_worst, color="#f87171", linestyle="--", linewidth=1.6, label=f"RE Worst ({_fmt_inr_short(re_worst[-1])})")
    ax.plot(x, re_base, color="#34d399", marker="o", linewidth=2.6, markersize=4.5, label=f"RE Base ({_fmt_inr_short(re_base[-1])})")
    ax.plot(x, fd, color="#fbbf24", linestyle=":", marker="s", linewidth=2.0, markersize=4, label=f"SBI FD ({_fmt_inr_short(fd[-1])})")
    ax.plot(x, nifty, color="#818cf8", marker="^", linewidth=2.2, markersize=4.5, label=f"Nifty 50 ({_fmt_inr_short(nifty[-1])})")

    ax.set_xticks(x)
    ax.set_xticklabels(years, color="#cbd5e1", fontsize=7.5)
    ax.yaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))
    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7.5, loc="upper left")
    ax.set_title("Wealth Creation Trajectory: Real Estate vs Liquid Benchmarks", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_tornado_png(items: list[dict[str, Any]]) -> bytes:
    fig, ax = _styled_fig()
    if items:
        labels = [it["label"] for it in items]
        lows = [it["low"] for it in items]
        widths = [it["high"] - it["low"] for it in items]
        base_val = items[0]["base"]
        y = np.arange(len(labels))
        bars = ax.barh(y, widths, left=lows, color="#38bdf8", height=0.55, alpha=0.9)
        ax.axvline(base_val, color="#f8fafc", linestyle="--", linewidth=1.5, label=f"Base IRR {base_val:.2f}%")
        for bar, it in zip(bars, items):
            ax.text(it["high"] + 0.15, bar.get_y() + bar.get_height() / 2, f"{it['low']:.1f}% - {it['high']:.1f}%", va="center", color="#f8fafc", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color="#f8fafc", fontsize=7.5)
        ax.invert_yaxis()
        max_x = max(it["high"] for it in items)
        min_x = min(it["low"] for it in items)
        ax.set_xlim(min_x - 1.0, max_x + 3.5)
        ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7.5)
    ax.set_xlabel("Equity IRR (%)", color="#94a3b8", fontsize=8)
    ax.set_title("Value Drivers — One-at-a-Time Impact (Equity IRR)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_heatmap_png(
    appr_axis: np.ndarray,
    esc_axis: np.ndarray,
    irr_grid: np.ndarray,
    current_appr_pct: Optional[float] = None,
    current_esc_pct: Optional[float] = None,
) -> bytes:
    fig, ax = _styled_fig()
    im = ax.imshow(irr_grid, origin="lower", aspect="auto", cmap="RdYlGn")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7.5)
    cbar.set_label("Equity IRR (%)", color="#f8fafc", fontsize=8)

    ax.set_xticks(np.arange(len(esc_axis)))
    ax.set_xticklabels([f"{v:.1f}%" for v in esc_axis], color="#cbd5e1", fontsize=7.5)
    ax.set_yticks(np.arange(len(appr_axis)))
    ax.set_yticklabels([f"{v:.1f}%" for v in appr_axis], color="#cbd5e1", fontsize=7.5)
    if current_appr_pct is not None and current_esc_pct is not None and len(esc_axis) > 1 and len(appr_axis) > 1:
        ix = float(np.interp(current_esc_pct, esc_axis, np.arange(len(esc_axis))))
        iy = float(np.interp(current_appr_pct, appr_axis, np.arange(len(appr_axis))))
        ax.scatter([ix], [iy], marker="*", s=160, color="#facc15", edgecolors="#0f172a", linewidths=1.2, zorder=5)
    ax.set_xlabel("Rent Escalation (%)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Annual Appreciation (%)", color="#94a3b8", fontsize=8)
    ax.set_title("Equity IRR Sensitivity: Appreciation x Rent Escalation", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_expected_value_png(ev: ExpectedValueResult) -> bytes:
    labels = ["Bad Market (Worst)", "Realistic Base", "Optimistic (Best)", "Expected Value"]
    values = [ev.worst_terminal_wealth, ev.base_terminal_wealth, ev.best_terminal_wealth, ev.expected_terminal_wealth]
    colors = ["#f43f5e", "#38bdf8", "#34d399", "#a855f7"]

    fig, ax = _styled_fig((8.5, 3.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.55)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {_fmt_inr_short(v)}", va="center", color="#f8fafc", fontsize=8, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: _fmt_inr_short(v))
    ax.set_title("Probability-Weighted Expected Terminal Wealth", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)


def render_scatter_png(merged_df: pd.DataFrame, selected_pincode: str, selected_bhk: str) -> bytes:
    plot_df = merged_df.dropna(subset=["implied_gross_rental_yield_pct", "price_per_sqft_carpet_equiv"]).copy()
    zone_colors = {
        "Central": "#c084fc",
        "East": "#38bdf8",
        "West": "#34d399",
        "South": "#fbbf24",
        "North_PCMC": "#fb7185",
    }

    fig, ax = _styled_fig((8.5, 4.0))
    for zone_name, grp in plot_df.groupby("pune_zone"):
        ax.scatter(
            grp["price_per_sqft_carpet_equiv"],
            grp["implied_gross_rental_yield_pct"],
            s=np.clip(grp["est_ticket_price_inr"] / 400_000.0, 20, 140),
            color=zone_colors.get(str(zone_name), "#94a3b8"),
            alpha=0.75,
            label=str(zone_name),
        )

    med_p = float(plot_df["price_per_sqft_carpet_equiv"].median())
    med_y = float(plot_df["implied_gross_rental_yield_pct"].median())
    ax.axvline(med_p, color="#475569", linestyle=":", linewidth=1.1)
    ax.axhline(med_y, color="#475569", linestyle=":", linewidth=1.1)

    pcode_clean = str(selected_pincode).strip().zfill(6)
    sel = plot_df[(plot_df["pincode"] == pcode_clean) & (plot_df["bhk"] == selected_bhk)]
    if not sel.empty:
        row = sel.iloc[0]
        ax.scatter(
            [row["price_per_sqft_carpet_equiv"]],
            [row["implied_gross_rental_yield_pct"]],
            marker="*",
            s=260,
            color="#facc15",
            edgecolors="#ffffff",
            linewidths=1.2,
            zorder=6,
            label=f"Selected: {row['locality_name']}",
        )

    ax.set_xlabel("Carpet Price (INR / sq.ft.)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Gross Rental Yield (%)", color="#94a3b8", fontsize=8)
    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7, ncol=3, loc="upper right")
    ax.set_title("Pune Micro-Market Yield vs Valuation Matrix", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    return _to_png_bytes(fig)
