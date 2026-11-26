from __future__ import annotations

import importlib
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Automatically reload any project submodules if they were modified on disk
for _mod_name in list(sys.modules.keys()):
    if _mod_name.startswith("src."):
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            pass

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
)
from src.data_loader import (
    get_all_zones,
    get_localities,
    get_merged_dataset,
    get_property_market_defaults,
    load_locality_master,
    load_macro_constants,
    load_unit_market_data,
    load_zone_risk,
)
from src.advanced_analysis import (
    compute_break_even_appreciation,
    compute_probability_weighted,
    compute_return_decomposition,
    compute_sensitivity_grid,
    compute_tornado_data,
)
from src.decision_gates import evaluate_master_verdict
from src.explainer import (
    render_tab1_explainer,
    render_tab2_explainer,
    render_tab3_explainer,
    render_tab4_explainer,
)
from src.export_engine import generate_full_dashboard_csv
from src.finance_engine import (
    compute_rbi_max_ltv,
    run_full_underwriting_matrix,
)
from src.pdf_generator import generate_underwriting_pdf
from src.recommender import rank_top_localities
from src.methodology import (
    FINANCIAL_GLOSSARY,
    build_verdict_narrative,
    explain_acquisition,
    explain_benchmarks,
    explain_exit,
    explain_gate,
    explain_loan,
    explain_tax_year,
)
from src.models import FinancingInput, InvestorProfile, PropertyInput


# Set page layout and title
st.set_page_config(
    page_title="Pune Residential Real Estate Underwriter",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom executive CSS styling
st.markdown(
    """
    <style>
        /* Top header clearance so all action buttons are 100% visible and unclipped */
        header[data-testid="stHeader"] {
            background: rgba(11, 15, 25, 0.85) !important;
            backdrop-filter: blur(8px);
            z-index: 100 !important;
            height: 2.75rem !important;
        }
        .block-container {
            padding-top: 3.6rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 96% !important;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 22px 26px;
            color: #f8fafc;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }
        /* Strict geometric sizing for metric cards: identical height & flexbox baseline */
        .metric-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-top: 3px solid #3b82f6;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
            text-align: left;
            min-height: 120px;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 1.60rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.80rem;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 500;
        }
        /* Symmetrical cards for Scenarios and Stress Tests */
        .scenario-card {
            background: #111827;
            border-left: 1px solid #1f2937;
            border-right: 1px solid #1f2937;
            border-bottom: 1px solid #1f2937;
            border-radius: 10px;
            padding: 20px;
            min-height: 345px;
            height: 345px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            margin-bottom: 16px;
        }
        .stress-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 18px 20px;
            min-height: 310px;
            height: 310px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .statutory-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-top: 3px solid #3b82f6;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            min-height: 380px;
            height: 380px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        }
        .opp-cost-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            box-sizing: border-box;
        }
        /* Buttons styling */
        div[data-testid="stDownloadButton"] {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            width: 100%;
        }
        div[data-testid="stDownloadButton"] > button {
            min-height: 44px !important;
            height: 44px !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.02em !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid #3b82f6 !important;
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 18px !important;
            width: 100% !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
            border-color: #60a5fa !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-right: 6px;
            letter-spacing: 0.02em;
        }
        .badge-green { background-color: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }
        .badge-amber { background-color: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.4); }
        .badge-red { background-color: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.4); }
        .badge-slate { background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
        .badge-blue { background-color: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.4); }
        .verdict-box {
            border-radius: 10px;
            padding: 16px 24px;
            margin-bottom: 0px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .verdict-go {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(5, 150, 105, 0.35) 100%);
            border: 2px solid #10b981;
            color: #34d399;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
        }
        .verdict-caution {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.22) 0%, rgba(217, 119, 6, 0.35) 100%);
            border: 2px solid #f59e0b;
            color: #fbbf24;
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
        }
        .verdict-nogo {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.35) 100%);
            border: 2px solid #ef4444;
            color: #f87171;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 18px;
            border-radius: 8px;
            font-weight: 600;
            color: #94a3b8;
            background-color: #111827;
            border: 1px solid #1f2937;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border-color: #3b82f6 !important;
        }
        div[data-testid="stExpander"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_macro():
    return load_macro_constants()


@st.cache_data
def get_cached_merged():
    return get_merged_dataset()


@st.cache_data
def get_cached_zones():
    return get_all_zones()


@st.cache_data
def get_cached_zone_df():
    return load_zone_risk()


@st.cache_data
def get_cached_loc_df():
    return load_locality_master()


macro = get_cached_macro()
merged_df = get_cached_merged()
zones = get_cached_zones()
zone_df = get_cached_zone_df()
loc_df = get_cached_loc_df()

# ---------------------------------------------------------
# SIDEBAR: 4-TIER HIERARCHICAL INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Pune Deal Underwriter")
    st.caption("Quantitative Decision Support Engine")
    view_mode = st.radio(
        "Experience Mode",
        ["Simple Buyer Mode", "Pro Analyst Mode"],
        index=0,
        horizontal=True,
        help="Simple Mode highlights plain-English answers and compact tables. Pro Mode unlocks full 12-column accounting schedules and live mathematical proofs.",
    )
    is_pro_mode = view_mode == "Pro Analyst Mode"

    # Tier 1: Property Selection & Market Lookup
    st.markdown("#### 1. Micro-Market & Configuration")
    zone_options = ["All Zones"] + zones
    selected_zone_filter = st.selectbox("Zone Filter", zone_options, index=0)

    available_localities = get_localities(selected_zone_filter)
    loc_display_map = {f"{r['pincode']} - {r['locality_name']}": r for r in available_localities}
    loc_keys = list(loc_display_map.keys())

    # Default to Baner/Balewadi if available; preserve programmatic jump from recommender
    default_idx = 0
    for idx, key in enumerate(loc_keys):
        if "Baner" in key:
            default_idx = idx
            break

    if "selected_loc_key" not in st.session_state or st.session_state["selected_loc_key"] not in loc_keys:
        st.session_state["selected_loc_key"] = loc_keys[default_idx]

    selected_loc_key = st.selectbox("Locality & Pincode", loc_keys, key="selected_loc_key")
    selected_loc_record = loc_display_map[selected_loc_key]
    sel_pincode = selected_loc_record["pincode"]
    sel_loc_name = selected_loc_record["locality_name"]

    bhk_options = ["1_BHK", "2_BHK", "3_BHK", "4_PLUS_BHK"]
    selected_bhk = st.selectbox("Configuration", bhk_options, index=1)

    # Look up benchmark defaults
    defaults = get_property_market_defaults(sel_pincode, sel_loc_name, selected_bhk)

    # Display micro-market badges
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <span class="badge badge-slate">Zone: {defaults.pune_zone}</span>
            <span class="badge badge-slate">Govt: {defaults.jurisdiction}</span>
            <span class="badge badge-slate">Liquidity: {defaults.liquidity_tier.upper()}</span><br>
            <span class="badge badge-slate" style="margin-top: 4px;">RR Rate: ₹{defaults.rr_rate_primary_sqft:,.0f}/sqft</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _default_ptax_for(prop_def) -> float:
        rr_val = prop_def.carpet_area_sqft * prop_def.rr_rate_primary_sqft
        return float(round((rr_val if rr_val > 0 else prop_def.agreement_value_inr) * 0.0025))

    def _sync_benchmarks_to_state(prop_def) -> None:
        st.session_state["override_agreement"] = float(prop_def.agreement_value_inr)
        st.session_state["override_carpet"] = float(prop_def.carpet_area_sqft)
        st.session_state["override_rent"] = float(prop_def.monthly_rent_inr)
        st.session_state["override_appreciation"] = float(prop_def.appreciation_rate * 100.0)
        st.session_state["override_maint"] = float(round(prop_def.expected_sba_sqft * 4.0))
        st.session_state["override_ptax"] = _default_ptax_for(prop_def)

    # Refresh overrides whenever the user switches locality/pincode or BHK tier
    benchmark_key = f"{sel_pincode}_{selected_bhk}"
    if st.session_state.get("active_benchmark_key") != benchmark_key:
        st.session_state["active_benchmark_key"] = benchmark_key
        _sync_benchmarks_to_state(defaults)

    # Reset button
    if st.button("↺ Reset to Market Benchmarks", width="stretch"):
        _sync_benchmarks_to_state(defaults)
        st.rerun()

    # Editable property overrides (bound to keys so switching localities always refreshes values)
    agreement_val_input = st.number_input(
        "Agreement Value (INR)",
        min_value=500_000.0,
        max_value=250_000_000.0,
        step=50_000.0,
        format="%.0f",
        key="override_agreement",
        help="Base purchase price agreed with developer/seller before statutory closing costs.",
    )

    carpet_area_input = st.number_input(
        "Carpet Area (sq. ft.)",
        min_value=150.0,
        max_value=15_000.0,
        step=10.0,
        format="%.1f",
        key="override_carpet",
    )

    # Handle missing rent notice for Chakan 4_PLUS_BHK
    is_rent_missing = defaults.monthly_rent_inr <= 0.0
    if is_rent_missing:
        st.warning("⚠️ No rental comps found for this BHK tier. Please enter estimated monthly rent.")

    monthly_rent_input = st.number_input(
        "Expected Monthly Rent (INR)",
        min_value=0.0,
        max_value=2_000_000.0,
        step=1_000.0,
        format="%.0f",
        key="override_rent",
        help="Gross monthly rent achievable. Used for cash flow in investment and imputed savings in self-use.",
    )

    appreciation_pct_input = st.number_input(
        "Expected Annual Appreciation (%)",
        min_value=0.0,
        max_value=25.0,
        step=0.25,
        format="%.2f",
        key="override_appreciation",
        help=f"Locality trailing 3–5Y median: {defaults.appreciation_rate * 100:.2f}%.",
    )

    st.markdown("---")

    # Tier 2: Purchase Purpose & Asset Branching
    st.markdown("#### 2. Purpose & Legal Structuring")
    use_case = st.selectbox(
        "Purchase Goal",
        ["investment_rental", "self_use"],
        format_func=lambda x: "Investment / Rental Property" if x == "investment_rental" else "Primary Residence (Self-Use)",
        index=0,
    )

    prop_status = st.selectbox(
        "Construction Status",
        ["ready_resale", "under_construction"],
        format_func=lambda x: "Ready / Resale (No GST)" if x == "ready_resale" else "Under-Construction (GST Applicable)",
        index=0,
    )

    expected_possession = 24
    if prop_status == "under_construction":
        expected_possession = st.slider("Expected Possession (Months)", min_value=6, max_value=60, value=24, step=6)
        st.caption(f"Zone avg completion delay: +{defaults.avg_uc_delay_months:.0f} months")

    buyer_ownership = st.selectbox(
        "Buyer Ownership Structure",
        ["sole_male", "sole_female", "female_joint", "male_female_50_50"],
        format_func=lambda x: {
            "sole_male": "Sole Male (7% Urban / 4% Rural)",
            "sole_female": "Sole Female (6% Urban / 3% Rural)",
            "female_joint": "Joint All-Female (6% Urban / 3% Rural)",
            "male_female_50_50": "Joint 50:50 Male-Female (6.5% Urban / 3.5% Rural)",
        }[x],
        index=0,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        buy_brokerage_enabled = st.checkbox("Buy Brokerage", value=(prop_status == "ready_resale"), help="1.5% + 18% GST")
    with col_b2:
        sell_brokerage_enabled = st.checkbox("Exit Brokerage", value=True, help="1.5% + 18% GST on exit")

    st.markdown("---")

    # Tier 3: Investor Profile
    st.markdown("#### 3. Investor Financial Profile")
    net_income = st.number_input(
        "Monthly Take-Home Income (INR)",
        min_value=30_000.0,
        max_value=10_000_000.0,
        value=250_000.0,
        step=10_000.0,
        format="%.0f",
    )
    living_expenses = st.number_input(
        "Monthly Living Expenses (INR)",
        min_value=10_000.0,
        max_value=5_000_000.0,
        value=75_000.0,
        step=5_000.0,
        format="%.0f",
    )
    existing_emis = st.number_input(
        "Existing Monthly EMIs (INR)",
        min_value=0.0,
        max_value=2_000_000.0,
        value=0.0,
        step=5_000.0,
        format="%.0f",
    )
    liquid_savings = st.number_input(
        "Total Liquid Savings (INR)",
        min_value=100_000.0,
        max_value=200_000_000.0,
        value=6_000_000.0,
        step=100_000.0,
        format="%.0f",
        help="Cash, FDs, mutual funds available for down payment and emergency reserves.",
    )
    upcoming_cash_need = st.number_input(
        "Planned Non-Property Cash Need (INR)",
        min_value=0.0,
        max_value=50_000_000.0,
        value=0.0,
        step=100_000.0,
        format="%.0f",
        help="Large cash requirements in next 1-3 years (e.g. child education, wedding).",
    )
    emergency_months_target = st.slider("Target Emergency Buffer (Months)", min_value=3, max_value=18, value=6, step=1)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tax_slab = st.selectbox(
            "Tax Slab (%)",
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
            index=5,
            format_func=lambda x: f"{int(x * 100)}%",
        )
    with col_t2:
        tax_regime = st.selectbox(
            "Tax Regime",
            ["new", "old"],
            index=0,
            format_func=lambda x: "New (No set-off)" if x == "new" else "Old (₹2L loss set-off)",
        )

    surcharge_rate = st.selectbox(
        "Income Tax Surcharge Tier",
        [0.0, 0.10, 0.15, 0.25],
        index=0,
        format_func=lambda x: (
            "0% (Annual Income ≤ ₹50L)" if x == 0.0 else (
                "10% (₹50L – ₹1 Cr)" if x == 0.10 else (
                    "15% (₹1 Cr – ₹2 Cr)" if x == 0.15 else "25% (> ₹2 Cr)"
                )
            )
        ),
        help="Applies statutory surcharge on income tax, FD tax, and property/equity capital gains tax.",
    )

    st.markdown("---")

    # Tier 4: Financing & Horizon Assumptions (Expander)
    with st.expander("⚙️ 4. Mortgage, Friction & Macro Assumptions", expanded=False):
        rbi_limit = compute_rbi_max_ltv(agreement_val_input, macro)
        desired_ltv = st.slider(
            f"Desired LTV (RBI Cap: {int(rbi_limit * 100)}%)",
            min_value=0.0,
            max_value=rbi_limit,
            value=min(0.75, rbi_limit),
            step=0.05,
            format="%.0f%%",
        )
        home_loan_rate = st.number_input(
            "Home Loan Interest Rate (%)",
            min_value=4.0,
            max_value=16.0,
            value=float(macro["base_home_loan_rate_pct"] * 100.0),
            step=0.10,
            format="%.2f",
        ) / 100.0

        loan_tenure = st.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=1)
        holding_period = st.slider("Holding Period (Years)", min_value=2, max_value=25, value=10, step=1)

        rent_escalation = st.number_input(
            "Annual Rent Escalation (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            format="%.1f",
        ) / 100.0

        society_maintenance = st.number_input(
            "Monthly Maintenance (INR)",
            min_value=0.0,
            max_value=200_000.0,
            step=500.0,
            format="%.0f",
            key="override_maint",
        )

        property_tax = st.number_input(
            "Annual Property Tax (INR)",
            min_value=0.0,
            max_value=500_000.0,
            step=1_000.0,
            format="%.0f",
            key="override_ptax",
            help="Anchored to 0.25% of Ready Reckoner value by default (PMC/PCMC assessment basis).",
        )

        interiors_misc = st.number_input(
            "Interiors & Upfront Misc (INR)",
            min_value=0.0,
            max_value=20_000_000.0,
            value=0.0,
            step=50_000.0,
            format="%.0f",
        )

        cpi_inflation = st.number_input(
            "CPI Inflation Rate (% / yr)",
            min_value=2.0,
            max_value=12.0,
            value=5.5,
            step=0.25,
            format="%.2f",
            help="Drives opex escalation and computes Real (inflation-adjusted) Equity IRR.",
        ) / 100.0

        capex_reserve_pct = st.number_input(
            "CapEx & Re-Letting Reserve (% of EGI)",
            min_value=0.0,
            max_value=15.0,
            value=5.0 if use_case == "investment_rental" else 2.0,
            step=0.5,
            format="%.1f",
            help="Annual reserve for periodic painting, major repairs, and tenant re-letting brokerage.",
        ) / 100.0

        loan_motd_pct = st.number_input(
            "Loan Processing & MOTD Franking (% of Loan)",
            min_value=0.0,
            max_value=2.0,
            value=0.35,
            step=0.05,
            format="%.2f",
            help="Maharashtra MOTD mortgage stamp duty (~0.3%) + bank legal/processing fees.",
        ) / 100.0

        mean_revert_appr = st.checkbox(
            "Normalize Appreciation After Yr 5 (→ 6.5%)",
            value=True,
            help="Mitigates 3–5Y post-COVID boom extrapolation bias by reverting 75% of excess appreciation toward Pune's 6.5% long-run equilibrium over Years 6–10.",
        )

        nifty_benchmark = st.number_input(
            "Nifty 50 TRI CAGR (%)",
            min_value=6.0,
            max_value=20.0,
            value=float(macro["liquid_benchmark_nifty50_cagr_pct"] * 100.0),
            step=0.25,
            format="%.2f",
        ) / 100.0


# ---------------------------------------------------------
# ASSEMBLE MODEL INSTANCES & EXECUTE UNDERWRITING ENGINE
# ---------------------------------------------------------
property_obj = PropertyInput(
    pincode=sel_pincode,
    locality_name=sel_loc_name,
    pune_zone=defaults.pune_zone,
    jurisdiction=defaults.jurisdiction,
    bhk=selected_bhk,
    use_case=use_case,
    property_status=prop_status,
    buyer_ownership=buyer_ownership,
    agreement_value_inr=agreement_val_input,
    carpet_area_sqft=carpet_area_input,
    expected_sba_sqft=defaults.expected_sba_sqft,
    monthly_rent_inr=monthly_rent_input,
    appreciation_rate=appreciation_pct_input / 100.0,
    rr_rate_primary_sqft=defaults.rr_rate_primary_sqft,
    liquidity_tier=defaults.liquidity_tier,
    quarters_to_sell=defaults.quarters_to_sell,
    exit_discount_stress_pct=defaults.exit_discount_stress_pct,
    avg_uc_delay_months=defaults.avg_uc_delay_months,
    vacancy_months_best=defaults.vacancy_months_best,
    vacancy_months_avg=defaults.vacancy_months_avg,
    vacancy_months_worst=defaults.vacancy_months_worst,
    expected_possession_months=expected_possession,
    brokerage_enabled=buy_brokerage_enabled,
    sell_brokerage_enabled=sell_brokerage_enabled,
)

financing_obj = FinancingInput(
    desired_ltv_pct=desired_ltv,
    home_loan_interest_rate_pct=home_loan_rate,
    loan_tenure_years=loan_tenure,
    holding_period_years=holding_period,
    annual_rent_escalation_pct=rent_escalation,
    monthly_society_maintenance_inr=society_maintenance,
    annual_property_tax_inr=property_tax,
    interiors_and_misc_inr=interiors_misc,
    nifty_benchmark_cagr_pct=nifty_benchmark,
    cpi_inflation_pct=cpi_inflation,
    capex_and_turnover_reserve_pct=capex_reserve_pct,
    loan_origination_and_motd_pct=loan_motd_pct,
    mean_revert_appreciation_after_yr5=mean_revert_appr,
    long_term_appreciation_anchor_pct=0.065,
)

investor_obj = InvestorProfile(
    net_monthly_income_inr=net_income,
    monthly_living_expenses_inr=living_expenses,
    existing_monthly_emis_inr=existing_emis,
    total_liquid_savings_inr=liquid_savings,
    upcoming_cash_need_inr=upcoming_cash_need,
    emergency_buffer_months_target=emergency_months_target,
    marginal_tax_slab_pct=tax_slab,
    tax_regime=tax_regime,
    surcharge_rate_pct=surcharge_rate,
)

# Run full matrix across 3 scenarios and stress modules
matrix = run_full_underwriting_matrix(property_obj, financing_obj, investor_obj, macro)
verdict = evaluate_master_verdict(investor_obj, property_obj, financing_obj, matrix)
base_res = matrix.base

# Look up selected locality, unit, and zone reference rows for glass-box explainers
loc_records = loc_df[(loc_df["pincode"] == sel_pincode) & (loc_df["locality_name"] == sel_loc_name)]
loc_row = loc_records.iloc[0].to_dict() if not loc_records.empty else selected_loc_record
unit_records = merged_df[
    (merged_df["pincode"] == sel_pincode)
    & (merged_df["locality_name"] == sel_loc_name)
    & (merged_df["bhk"] == selected_bhk)
]
unit_row = unit_records.iloc[0].to_dict() if not unit_records.empty else {}
zone_records = zone_df[zone_df["pune_zone"] == defaults.pune_zone]
zone_row = zone_records.iloc[0].to_dict() if not zone_records.empty else {}

# Pre-compute All-Zones top recommendations once so both PDF export and UI panel include them
all_ranked = (
    rank_top_localities(
        bhk=selected_bhk,
        use_case=use_case,
        property_status=prop_status,
        buyer_ownership=buyer_ownership,
        brokerage_enabled=buy_brokerage_enabled,
        sell_brokerage_enabled=sell_brokerage_enabled,
        expected_possession_months=expected_possession,
        financing_template=financing_obj,
        investor=investor_obj,
        macro=macro,
        loc_df=loc_df,
        top_n=3,
        include_all=True,
    )
    if selected_zone_filter == "All Zones"
    else None
)

# ---------------------------------------------------------
# TOP ACTION BAR: EXECUTIVE INSTITUTIONAL EXPORT (PDF)
# ---------------------------------------------------------
col_top_title, col_top_action = st.columns([6.2, 3.8])
with col_top_title:
    st.markdown(
        """
        <div style="padding: 2px 0 6px 0;">
            <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.10em;">
                🏢 Pune Residential Real Estate Financial Underwriter
            </div>
            <div style="font-size:1.24rem; font-weight:800; color:#ffffff; letter-spacing:-0.01em;">
                Institutional Multi-Scenario Investment Memorandum
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_top_action:
    pdf_report_bytes = generate_underwriting_pdf(
        property_obj=property_obj,
        financing_obj=financing_obj,
        investor_obj=investor_obj,
        matrix=matrix,
        verdict=verdict,
        loc_row=loc_row,
        unit_row=unit_row,
        zone_row=zone_row,
        macro=macro,
        all_zone_recommendations=all_ranked,
    )
    st.download_button(
        label="📄 Download Underwriting Report (PDF)",
        data=pdf_report_bytes,
        file_name=f"pune_underwriting_memorandum_{sel_loc_name}_{selected_bhk}.pdf",
        mime="application/pdf",
        width="stretch",
        help="Export complete multi-page institutional PDF memorandum: Deal Parameters, Statutory Waterfall, Loan Amortization, Stress Tests, Audits, and All-Zones Top 3 Finder (when All Zones is active).",
    )

# ---------------------------------------------------------
# SMART MICRO-MARKET FINDER (WHEN "ALL ZONES" IS SELECTED)
# ---------------------------------------------------------
if selected_zone_filter == "All Zones" and all_ranked:
    with st.expander("🧭 Smart Micro-Market Finder: Top 3 Pune Pincodes Aligned With Your Profile & Profitability", expanded=True):
        st.caption(
            f"Automatically underwrote all 43 Pune pincodes for your {selected_bhk.replace('_', ' ')} ({use_case.replace('_', ' ')}) "
            f"profile — matching your ₹{liquid_savings/100_000:.1f}L liquid savings, ₹{net_income/1000:.0f}k/mo income, {tax_regime.upper()} tax regime, and {holding_period}y horizon."
        )
        top_3_recs = all_ranked[:3]
        if top_3_recs:
            def _select_recommended_locality(target_key: str) -> None:
                st.session_state["selected_loc_key"] = target_key
                st.session_state["active_benchmark_key"] = None

            rec_cols = st.columns(len(top_3_recs))
            for r_col, rec in zip(rec_cols, top_3_recs):
                v_badge = "badge-green" if rec.verdict_status == "GO" else ("badge-amber" if rec.verdict_status == "CAUTION" else "badge-red")
                border_top = "#10b981" if rec.verdict_status == "GO" else ("#f59e0b" if rec.verdict_status == "CAUTION" else "#ef4444")
                rec_irr = f"{rec.base_irr_pct:.2f}%" if rec.base_irr_pct is not None else "N/A"
                rec_real_irr = f"{rec.real_irr_pct:.2f}%" if rec.real_irr_pct is not None else "N/A"
                rec_dscr = f"{rec.dscr:.2f}x" if rec.dscr is not None else "All-Cash"
                with r_col:
                    st.markdown(
                        f"""
                        <div style="background:#111827; border:1px solid #1f2937; border-top:4px solid {border_top}; border-radius:10px; padding:16px 18px; min-height:375px; display:flex; flex-direction:column; justify-content:space-between; box-sizing:border-box;">
                            <div>
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <span class="badge badge-blue">Rank #{rec.rank} · Score {rec.alignment_score:.0f}/100</span>
                                    <span class="badge {v_badge}">{rec.verdict_status}</span>
                                </div>
                                <div style="font-size:1.15rem; font-weight:800; color:#ffffff; margin-bottom:2px;">
                                    {rec.locality_name}
                                </div>
                                <div style="font-size:0.80rem; color:#94a3b8; margin-bottom:10px;">
                                    PIN: <b style="color:#cbd5e1;">{rec.pincode}</b> · {rec.pune_zone} Zone · {rec.jurisdiction}
                                </div>
                                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; background:#0b0f19; border:1px solid #1f2937; border-radius:8px; padding:10px; margin-bottom:10px; font-size:0.82rem;">
                                    <div><span style="color:#94a3b8;">Ticket:</span> <b style="color:#ffffff;">{format_inr(rec.ticket_price_inr)}</b></div>
                                    <div><span style="color:#94a3b8;">Upfront:</span> <b style="color:#ffffff;">{format_inr(rec.upfront_cash_inr)}</b></div>
                                    <div><span style="color:#94a3b8;">Base IRR:</span> <b style="color:#34d399;">{rec_irr}</b> <span style="color:#64748b;">({rec_real_irr} real)</span></div>
                                    <div><span style="color:#94a3b8;">Net Profit:</span> <b style="color:{'#38bdf8' if rec.net_wealth_created_inr >= 0 else '#f87171'};">{format_inr(rec.net_wealth_created_inr)}</b></div>
                                    <div><span style="color:#94a3b8;">Yr-1 CF/mo:</span> <b style="color:{'#34d399' if rec.year_1_monthly_cf_inr >= 0 else '#fbbf24'};">{format_inr(rec.year_1_monthly_cf_inr)}</b></div>
                                    <div><span style="color:#94a3b8;">DSCR / RR:</span> <b style="color:#cbd5e1;">{rec_dscr} / {rec.market_to_rr_ratio:.2f}x</b></div>
                                </div>
                                <ul style="margin:0; padding-left:16px; font-size:0.78rem; color:#cbd5e1; line-height:1.42;">
                                    {''.join(f'<li style="margin-bottom:4px;">{r}</li>' for r in rec.alignment_reasons)}
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    target_loc_key = f"{rec.pincode} - {rec.locality_name}"
                    if target_loc_key in loc_display_map:
                        st.button(
                            f"🔍 Inspect PIN {rec.pincode} ({rec.locality_name})",
                            key=f"inspect_rec_{rec.rank}_{rec.pincode}",
                            width="stretch",
                            on_click=_select_recommended_locality,
                            args=(target_loc_key,),
                        )

            with st.expander("📊 View All 43 Ranked Pune Pincodes for Your Inputs", expanded=False):
                rank_df = pd.DataFrame([
                    {
                        "Rank": r.rank,
                        "Pincode": r.pincode,
                        "Locality": r.locality_name,
                        "Zone": r.pune_zone,
                        "Score": f"{r.alignment_score:.1f}",
                        "Verdict": r.verdict_status,
                        "Ticket Price": format_inr(r.ticket_price_inr),
                        "Upfront Cash": format_inr(r.upfront_cash_inr),
                        "Base IRR": f"{r.base_irr_pct:.2f}%" if r.base_irr_pct is not None else "N/A",
                        "Real IRR": f"{r.real_irr_pct:.2f}%" if r.real_irr_pct is not None else "N/A",
                        "Net Wealth Created": format_inr(r.net_wealth_created_inr),
                        "Net Yield": f"{r.net_yield_pct:.2f}%",
                        "DSCR": f"{r.dscr:.2f}x" if r.dscr is not None else "N/A",
                        "Mkt/RR": f"{r.market_to_rr_ratio:.2f}x",
                        "Liquidity": f"{r.liquidity_tier.upper()} ({r.quarters_to_sell:.1f}Q)",
                    }
                    for r in all_ranked
                ])
                st.dataframe(rank_df, width="stretch", hide_index=True)

# ---------------------------------------------------------
# TOP HERO BANNER, MASTER VERDICT & 10-SECOND BOTTOM-LINE BRIEF
# ---------------------------------------------------------
irr_display = f"{base_res.equity_irr_pct:.2f}%" if base_res.equity_irr_pct is not None else "N/A"
real_irr_display = f"{base_res.real_equity_irr_pct:.2f}%" if base_res.real_equity_irr_pct is not None else "N/A"
dscr_display = f"{base_res.year_1_dscr:.2f}x" if base_res.year_1_dscr is not None else "All-Cash"
rent_cov_str = (
    f"Rent Covers {base_res.year_1_dscr*100:.0f}% of EMI"
    if base_res.year_1_dscr is not None
    else "Zero-Debt All-Cash"
)
foir_pct = (investor_obj.existing_monthly_emis_inr + base_res.loan.monthly_emi) / investor_obj.net_monthly_income_inr * 100.0
post_cash_hero = investor_obj.total_liquid_savings_inr - base_res.loan.required_upfront_cash
m_obligations_hero = investor_obj.monthly_living_expenses_inr + investor_obj.existing_monthly_emis_inr + base_res.loan.monthly_emi
buf_months_hero = max(0.0, post_cash_hero / m_obligations_hero) if m_obligations_hero > 0 else 0.0
y1_monthly_cf = base_res.annual_cash_flows[0].net_cash_flow / 12.0
fd_adv_hero = base_res.fd_benchmark.net_wealth_advantage_inr
nifty_adv_hero = base_res.nifty_benchmark.net_wealth_advantage_inr

if post_cash_hero < 0:
    watchout_msg = f"Upfront cash shortfall of <b>{format_inr(abs(post_cash_hero))}</b> — reduce price/LTV or save more before buying."
elif foir_pct > 50.0:
    watchout_msg = f"High EMI burden (<b>{foir_pct:.1f}%</b> of salary > 50% ceiling) — vulnerable to interest rate hikes."
elif use_case == "investment_rental" and base_res.year_1_dscr is not None and base_res.year_1_dscr < 0.50:
    watchout_msg = f"Rent covers only <b>{base_res.year_1_dscr*100:.0f}%</b> of your EMI; you must subsidize <b>{format_inr(abs(y1_monthly_cf))}/mo</b> from salary in Year 1."
elif property_obj.liquidity_tier.lower() == "low":
    watchout_msg = f"Slow resale market in {defaults.pune_zone} Zone (~<b>{defaults.quarters_to_sell:.1f} quarters</b> inventory overhang) — plan to hold 7+ years."
elif (base_res.equity_irr_pct or 0.0) <= base_res.fd_benchmark.annual_return_pct:
    watchout_msg = f"Returns ({irr_display}) lag a safe bank FD ({base_res.fd_benchmark.annual_return_pct:.2f}%/yr) — only buy for personal use or negotiate a lower price."
else:
    watchout_msg = "Financially balanced deal — monitor floating home loan rate changes and tenant occupancy."

wallet_pocket_str = (
    f"pay <b style='color:#fbbf24;'>{format_inr(abs(y1_monthly_cf))}/mo</b> out-of-pocket"
    if y1_monthly_cf < 0
    else f"pocket <b style='color:#34d399;'>+{format_inr(y1_monthly_cf)}/mo</b> surplus"
)
fd_cmp_str = (
    f"<b style='color:#34d399;'>+{format_inr(fd_adv_hero)} more</b> than SBI FD"
    if fd_adv_hero >= 0
    else f"<b style='color:#f87171;'>{format_inr(abs(fd_adv_hero))} less</b> than SBI FD"
)
nifty_cmp_str = (
    f"<b style='color:#34d399;'>+{format_inr(nifty_adv_hero)} more</b> than Nifty 50"
    if nifty_adv_hero >= 0
    else f"<b style='color:#f87171;'>{format_inr(abs(nifty_adv_hero))} less</b> than Nifty 50"
)

st.markdown(
    f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <span class="badge badge-slate">PIN: {sel_pincode}</span>
                <span class="badge badge-slate">{defaults.pune_zone} Zone</span>
                <span class="badge badge-slate">{defaults.jurisdiction}</span>
                <span class="badge badge-slate">{selected_bhk.replace('_', ' ')}</span>
                <h1 style="margin: 10px 0 6px 0; font-size: 2.05rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                    {sel_loc_name} <span style="font-weight: 400; font-size: 1.35rem; color: #94a3b8;">({use_case.replace('_', ' ').title()})</span>
                </h1>
                <p style="margin: 0; font-size: 0.94rem; color: #cbd5e1; line-height: 1.6;">
                    Total Cost (incl. Taxes): <b style="color: #ffffff;">{format_inr(base_res.acquisition.all_in_acquisition_cost)}</b> &nbsp;|&nbsp;
                    Upfront Cash Needed: <b style="color: #ffffff;">{format_inr(base_res.loan.required_upfront_cash)}</b> &nbsp;|&nbsp;
                    EMI: <b style="color: #ffffff;">{format_inr(base_res.loan.monthly_emi)}/mo</b> (<b style="color: {'#34d399' if foir_pct <= 40 else ('#fbbf24' if foir_pct <= 50 else '#f87171')}">{foir_pct:.1f}%</b> of income) &nbsp;|&nbsp;
                    Annual Return: <b style="color: {'#34d399' if (base_res.equity_irr_pct or 0) > 8 else '#ffffff'}">{irr_display}</b> (<span style="color:#94a3b8;">After Inflation:</span> <b style="color:#cbd5e1;">{real_irr_display}</b>) &nbsp;|&nbsp;
                    <b style="color:#cbd5e1;">{rent_cov_str}</b>
                </p>
            </div>
            <div style="margin-top: 4px;">
                <div class="verdict-box verdict-{verdict.status.lower().replace('-', '')}" style="padding: 14px 28px; min-width: 230px; text-align: center;">
                    <div>
                        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800; opacity: 0.9;">Buy / No-Buy Verdict</div>
                        <div style="font-size: 1.75rem; font-weight: 900; letter-spacing: 0.03em;">
                            {'🟢' if verdict.status == 'GO' else ('🟡' if verdict.status == 'CAUTION' else '🔴')} {verdict.status}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid #334155; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 12px 16px; margin-top: 14px; font-size: 0.88rem; color: #e2e8f0; line-height: 1.55;">
            <div style="font-size: 0.74rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; margin-bottom: 4px;">⚡ 10-Second Plain-English Bottom Line</div>
            <div>• <b>Wallet Check:</b> You need <b style="color:#ffffff;">{format_inr(base_res.loan.required_upfront_cash)}</b> upfront (leaving <b style="color:#ffffff;">{format_inr(max(0.0, post_cash_hero))}</b> emergency buffer = <b>{buf_months_hero:.1f} months</b>) and will {wallet_pocket_str} in Year 1.</div>
            <div>• <b>Profit Check:</b> Selling after {holding_period} yrs leaves <b style="color:#ffffff;">{format_inr(base_res.exit.net_exit_proceeds)}</b> in hand — {fd_cmp_str} and {nifty_cmp_str} on identical cash invested.</div>
            <div>• <b>#1 Watch-Out:</b> {watchout_msg}</div>
        </div>
        <div style="display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;">
            <span class="badge badge-{'green' if verdict.gates[0].status == 'PASS' else ('amber' if verdict.gates[0].status == 'WARN' else 'red')}">
                1. Can You Afford It?: {verdict.gates[0].status}
            </span>
            <span class="badge badge-{'green' if verdict.gates[1].status == 'PASS' else ('amber' if verdict.gates[1].status == 'WARN' else 'red')}">
                2. Survives Bad Market?: {verdict.gates[1].status}
            </span>
            <span class="badge badge-{'green' if verdict.gates[2].status == 'PASS' else ('amber' if verdict.gates[2].status == 'WARN' else 'red')}">
                3. Fair Price & Rent?: {verdict.gates[2].status}
            </span>
            <span class="badge badge-{'green' if verdict.gates[3].status == 'PASS' else ('amber' if verdict.gates[3].status == 'WARN' else 'red')}">
                4. Beats FD & Nifty?: {verdict.gates[3].status}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# WORKSPACE TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary & 4-Gate Audit",
    "📈 Cash Flow, Amortization & Tax Engine",
    "⚖️ Scenarios, Stress Tests & Opportunity Cost",
    "🗺️ Pune Micro-Market Explorer & Comparables",
    "🔬 Sensitivity, Break-Even & Expected Value",
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY & 4-GATE AUDIT
# ---------------------------------------------------------
with tab1:
    # Plain-English transparency layer — explains every headline number
    with st.expander("🧠 Plain-English Summary — How Each Figure Is Calculated", expanded=False):
        st.markdown(build_verdict_narrative(verdict))
        st.markdown("---")
        st.markdown("#### 🏗️ Acquisition structure")
        st.markdown(explain_acquisition(base_res.acquisition))
        st.markdown("---")
        st.markdown("#### 🏦 Financing")
        st.markdown(explain_loan(base_res.loan, investor_obj, investor_obj.net_monthly_income_inr))
        st.markdown("---")
        st.markdown("#### 💰 Exit & capital-gains tax")
        st.markdown(explain_exit(base_res.exit))
        st.markdown("---")
        st.markdown("#### ⚖️ Opportunity cost vs liquid benchmarks")
        st.markdown(explain_benchmarks(base_res))

    # KPI Metric Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)

    with kpi_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Property Cost</div>
                <div class="metric-value">{format_inr(base_res.acquisition.all_in_acquisition_cost)}</div>
                <div class="metric-sub">Builder Price: {format_inr(base_res.acquisition.agreement_value_inr)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Upfront Cash Needed</div>
                <div class="metric-value">{format_inr(base_res.loan.required_upfront_cash)}</div>
                <div class="metric-sub">Bank Loan: {format_inr(base_res.loan.loan_amount)} ({base_res.loan.effective_ltv*100:.0f}% LTV)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Monthly EMI Burden</div>
                <div class="metric-value">{format_inr(base_res.loan.monthly_emi)}</div>
                <div class="metric-sub">{foir_pct:.1f}% of Salary (Safe ≤40%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_c4:
        post_cash = max(0.0, investor_obj.total_liquid_savings_inr - base_res.loan.required_upfront_cash)
        m_obligations = investor_obj.monthly_living_expenses_inr + investor_obj.existing_monthly_emis_inr + base_res.loan.monthly_emi
        buf_months = (post_cash / m_obligations) if m_obligations > 0 else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Emergency Cash Left</div>
                <div class="metric-value">{buf_months:.1f} Mo</div>
                <div class="metric-sub">Buffer Left: {format_inr(post_cash)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_c5:
        rent_cov_pct_txt = f"Covers {base_res.year_1_dscr*100:.0f}% EMI" if base_res.year_1_dscr is not None else "All-Cash"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Pure Rent Return (Yr 1)</div>
                <div class="metric-value">{base_res.year_1_net_yield_pct:.2f}%</div>
                <div class="metric-sub">Gross: {base_res.year_1_gross_yield_pct:.2f}% · {rent_cov_pct_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_c6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Your Annual Return (IRR)</div>
                <div class="metric-value" style="color: {'#34d399' if (base_res.equity_irr_pct or 0) > 8 else '#ffffff'};">{irr_display}</div>
                <div class="metric-sub">After CPI: {real_irr_display} · NPV: {format_inr(base_res.npv_inr)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4-Gate Decision Audit Expandable Breakdown
    st.markdown("### 🛡️ 4-Gate Decision Engine Audit (Plain-English Checklist)")
    gate_cols = st.columns(4)
    gate_plain_titles = {
        1: "Can You Afford It?",
        2: "Survives Bad Market?",
        3: "Fair Price & Rent?",
        4: "Beats FD & Nifty?",
    }

    for i, gate in enumerate(verdict.gates):
        badge_style = "badge-green" if gate.status == "PASS" else ("badge-amber" if gate.status == "WARN" else "badge-red")
        plain_title = gate_plain_titles.get(gate.gate_number, gate.name)
        with gate_cols[i]:
            with st.expander(f"Gate {gate.gate_number}: {plain_title} ({gate.status})", expanded=True):
                st.markdown(f"<span class='badge {badge_style}'>{gate.status}</span> <span style='font-size:0.78rem; color:#94a3b8;'>{gate.name}</span>", unsafe_allow_html=True)
                for reason in gate.reasons:
                    st.markdown(f"- {reason}")

    # Phase 3: Actionable Fix Suggestions when Verdict is CAUTION or NO-GO
    if verdict.status != "GO":
        fix_tips: list[str] = []
        g1, g2, g3, g4 = verdict.gates
        if g1.status != "PASS":
            if post_cash_hero < 0:
                fix_tips.append(
                    f"**Fix Affordability (Gate 1):** You are short **{format_inr(abs(post_cash_hero))}** upfront — "
                    f"consider a lower-priced area from the Top 3 Finder or increase your liquid down payment budget."
                )
            elif foir_pct > 40.0:
                fix_tips.append(
                    f"**Reduce EMI Burden (Gate 1):** EMI takes **{foir_pct:.1f}%** of salary — increasing down payment or extending loan tenure reduces monthly EMI."
                )
            else:
                fix_tips.append(
                    f"**Build Safety Cushion (Gate 1):** Post-purchase cash covers **{buf_months_hero:.1f} mo** — keep an extra reserve to hit your **{emergency_months_target:.0f}-month** target."
                )
        if g2.status != "PASS":
            yr3_rec = float(g2.metrics.get("yr3_capital_recovery_pct", 100.0))
            if yr3_rec < 75.0:
                fix_tips.append(
                    f"**Avoid Early Exit Loss (Gate 2):** Selling in Year 3 recovers only **{yr3_rec:.0f}%** of invested equity — commit to holding **5+ years** so appreciation covers stamp duty."
                )
            if float(g2.metrics.get("stressed_foir_200bps", 0.0)) > 50.0:
                fix_tips.append(
                    f"**Rate-Hike Protection (Gate 2):** A +2% loan rate hike pushes EMI to **{g2.metrics.get('stressed_foir_200bps', 0.0):.1f}%** of salary — keep extra cash reserves or prepay early."
                )
        if g3.status != "PASS":
            mkt_rr_val = float(g3.metrics.get("market_to_rr_ratio", 1.0))
            if mkt_rr_val > 2.50:
                target_price = base_res.acquisition.rr_market_value * 2.50
                disc_needed = max(0.0, (1.0 - target_price / base_res.acquisition.agreement_value_inr) * 100.0) if base_res.acquisition.agreement_value_inr > 0 else 0.0
                fix_tips.append(
                    f"**Negotiate Price (Gate 3):** Asking price is **{mkt_rr_val:.2f}×** Govt RR — negotiating **~{disc_needed:.0f}%** down toward {format_inr(target_price)} brings valuation into fair territory."
                )
            if use_case == "investment_rental" and base_res.year_1_net_yield_pct < 2.0:
                fix_tips.append(
                    f"**Boost Rental Yield (Gate 3):** Net yield is **{base_res.year_1_net_yield_pct:.2f}%** (< 2.0%) — compare higher-yield IT corridors in Tab 4 or use the All-Zones Top 3 Finder."
                )
        if g4.status != "PASS":
            fd_r = base_res.fd_benchmark.annual_return_pct
            hurdle_r = float(g4.metrics.get("risk_adjusted_hurdle_pct", fd_r + 4.0))
            if (base_res.equity_irr_pct or 0.0) <= fd_r:
                fix_tips.append(
                    f"**Beat Bank FD (Gate 4):** Return (**{irr_display}**) trails safe SBI FD (**{fd_r:.2f}%**) — extend holding period, negotiate a lower price, or pick a higher-growth micro-market."
                )
            else:
                fix_tips.append(
                    f"**Improve Risk Premium (Gate 4):** Return (**{irr_display}**) beats FD (**{fd_r:.2f}%**) but is below the **{hurdle_r:.2f}%** risk-adjusted target — lower loan rate or longer hold improves spread."
                )
        if fix_tips:
            with st.expander("🛠️ How Can You Turn This Deal Into a GO? (Actionable Fixes)", expanded=(verdict.status == "NO-GO")):
                for tip in fix_tips:
                    st.markdown(f"- {tip}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Acquisition Waterfall Chart & TDS Callout
    wf_col1, wf_col2 = st.columns([6.6, 3.4])
    with wf_col1:
        waterfall_fig = create_acquisition_waterfall_chart(base_res.acquisition)
        st.plotly_chart(waterfall_fig, width="stretch")
        extra_closing = base_res.acquisition.all_in_acquisition_cost - base_res.acquisition.agreement_value_inr
        extra_pct = (extra_closing / base_res.acquisition.agreement_value_inr * 100.0) if base_res.acquisition.agreement_value_inr > 0 else 0.0
        st.caption(
            f"💡 **Takeaway:** Stamp duty, registration, GST, and fees add **+{format_inr(extra_closing)} (+{extra_pct:.1f}%)** "
            f"above the {format_inr(base_res.acquisition.agreement_value_inr)} seller price. (1% TDS is deducted from the seller, not added to your cost.)"
        )

    with wf_col2:
        st.markdown("#### 📋 Statutory Checklist")
        st.markdown(
            f"""
            <div class="statutory-card">
                <div>
                    <p style="margin:0 0 10px 0; font-size:0.92rem; color:#cbd5e1;"><span style="color:#94a3b8;">Statutory Floor:</span> <b style="color:#ffffff;">₹{defaults.rr_rate_primary_sqft:,.0f}/sqft</b></p>
                    <p style="margin:0 0 10px 0; font-size:0.92rem; color:#cbd5e1;"><span style="color:#94a3b8;">Ready Reckoner Value:</span> <b style="color:#ffffff;">{format_inr(base_res.acquisition.rr_market_value)}</b></p>
                    <p style="margin:0 0 10px 0; font-size:0.92rem; color:#cbd5e1;"><span style="color:#94a3b8;">Stamp Duty Base:</span> <b style="color:#ffffff;">{format_inr(base_res.acquisition.stamp_duty_base)}</b></p>
                    <p style="margin:0 0 10px 0; font-size:0.92rem; color:#cbd5e1;"><span style="color:#94a3b8;">Effective Stamp Duty:</span> <b style="color:#ffffff;">{base_res.acquisition.effective_stamp_duty_pct*100:.2f}%</b></p>
                    <p style="margin:0 0 10px 0; font-size:0.92rem; color:#cbd5e1;"><span style="color:#94a3b8;">Registration Fee:</span> <b style="color:#ffffff;">{format_inr(base_res.acquisition.registration_fee_inr)} (Cap: ₹30k)</b></p>
                </div>
                <div style="background: {'rgba(245, 158, 11, 0.15)' if base_res.acquisition.tds_applicable else 'rgba(16, 185, 129, 0.15)'}; border: 1px solid {'rgba(251, 191, 36, 0.4)' if base_res.acquisition.tds_applicable else 'rgba(52, 211, 153, 0.4)'}; border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <div style="font-size:0.80rem; font-weight:700; color:{'#fbbf24' if base_res.acquisition.tds_applicable else '#34d399'};">
                        {'⚠️ TDS COMPLIANCE APPLICABLE (Sec 194-IA / 393(1))' if base_res.acquisition.tds_applicable else '✅ NO TDS (Below ₹50 Lakhs Threshold)'}
                    </div>
                    <div style="font-size:0.84rem; margin-top:5px; color:{'#fde68a' if base_res.acquisition.tds_applicable else '#a7f3d0'}; line-height: 1.4;">
                        {f'Mandatory 1% TDS: <b style="color:#ffffff;">{format_inr(base_res.acquisition.tds_amount_inr)}</b> deducted from seller payout and deposited with IT Department. <i style="color:#e2e8f0;">Not an extra cost to buyer.</i>' if base_res.acquisition.tds_applicable else 'Transaction value is below ₹50 Lakhs.'}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Glass-box calculation proof & decision audit
    render_tab1_explainer(
        result=base_res,
        decision=verdict,
        prop_input=property_obj,
        investor_profile=investor_obj,
        loc_row=loc_row,
        unit_row=unit_row,
        macro=macro,
    )


# ---------------------------------------------------------
# TAB 2: CASH FLOW, AMORTIZATION & TAX ENGINE
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📅 Year-by-Year Cash Flow & Loan Schedule")
    st.caption("Plain-English yearly summary of rent, EMI, taxes, your monthly pocket impact, and realizable profit if sold.")

    show_full_accounting = st.checkbox(
        "Show Full 12-Column Institutional Accounting Schedule",
        value=is_pro_mode,
        help="Switch between a simple 6-column buyer summary and the full institutional operating/amortization ledger.",
    )

    # Full 12-column DataFrame (used for CSV download and Pro View)
    cf_records = []
    compact_records = []
    for row in base_res.annual_cash_flows:
        cf_records.append({
            "Year": f"Yr {row.year}",
            "Gross/Imputed Rent": row.gross_annual_rent,
            "Vacancy Loss": row.vacancy_loss,
            "Opex": row.opex,
            "NOI": row.noi,
            "Annual EMI": row.annual_emi,
            "Interest Paid": row.interest_paid,
            "Principal Paid": row.principal_paid,
            "Tax Impact": row.tax_impact,
            "Net Cash Flow": row.net_cash_flow,
            "Closing Loan Balance": row.remaining_loan_balance,
            "Net Equity": row.net_equity,
        })
        tax_label_val = -row.tax_impact  # positive = tax saved, negative = tax paid
        compact_records.append({
            "Year": f"Yr {row.year}",
            "Rent Collected": format_inr(row.egi, precision=2),
            "Total EMI Paid": format_inr(row.annual_emi, precision=2),
            "Tax Saved / (Paid)": format_inr(tax_label_val, precision=2),
            "Pocket Impact (Yearly)": format_inr(row.net_cash_flow, precision=2),
            "Pocket Impact (Monthly)": f"{format_inr(row.net_cash_flow / 12.0, precision=1)}/mo",
            "Cash In Hand If Sold": format_inr(row.net_realizable_equity, precision=2),
        })
    cf_df = pd.DataFrame(cf_records)

    if show_full_accounting:
        display_df = cf_df.copy()
        for col in display_df.columns:
            if col != "Year":
                display_df[col] = display_df[col].apply(lambda x: format_inr(x, precision=2))
        st.dataframe(display_df, width="stretch", hide_index=True)
    else:
        st.dataframe(pd.DataFrame(compact_records), width="stretch", hide_index=True)

    # Download CSV button
    csv_bytes = cf_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cash Flow Schedule (CSV)",
        data=csv_bytes,
        file_name=f"pune_real_estate_cashflow_{sel_loc_name}_{selected_bhk}.csv",
        mime="text/csv",
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Charts: Cash Flow Trajectory & Amortization Stack
    cf_chart_col, amort_chart_col = st.columns(2)
    with cf_chart_col:
        cf_fig = create_cash_flow_trajectory_chart(base_res.annual_cash_flows)
        st.plotly_chart(cf_fig, width="stretch")
        pos_yr = next((cf.year for cf in base_res.annual_cash_flows if cf.net_cash_flow >= 0), None)
        tot_drain = sum(min(0.0, cf.net_cash_flow) for cf in base_res.annual_cash_flows)
        be_msg = (
            f"Cash flow turns positive in **Year {pos_yr}** (rent catches up to EMI)."
            if pos_yr is not None
            else f"Rent stays below EMI across all {holding_period} yrs (total top-up: **{format_inr(abs(tot_drain))}**)."
        )
        st.caption(f"💡 **Takeaway:** Year-1 pocket impact is **{format_inr(y1_monthly_cf)}/mo**. {be_msg}")

    with amort_chart_col:
        amort_fig = create_amortization_chart(base_res.loan.annual_schedule)
        st.plotly_chart(amort_fig, width="stretch")
        tot_int = sum(cf.interest_paid for cf in base_res.annual_cash_flows)
        tot_prin = sum(cf.principal_paid for cf in base_res.annual_cash_flows)
        st.caption(
            f"💡 **Takeaway:** Over {holding_period} yrs, you pay **{format_inr(tot_int)}** in bank interest and "
            f"repay **{format_inr(tot_prin)}** of loan principal (remaining loan at exit: **{format_inr(base_res.loan.remaining_loan_balance_H)}**)."
        )

    # Glass-box calculation proof & decision audit
    render_tab2_explainer(
        result=base_res,
        prop_input=property_obj,
        investor_profile=investor_obj,
        macro=macro,
    )


# ---------------------------------------------------------
# TAB 3: SCENARIOS, STRESS TESTS & OPPORTUNITY COST
# ---------------------------------------------------------
with tab3:
    st.markdown("### ⚖️ Multi-Scenario Matrix & Liquid Opportunity Cost")
    st.caption("Side-by-side analysis under Best, Base, and Worst (Stress) conditions against SBI Term Deposits and Nifty 50 TRI.")

    # Side-by-side scenario comparison table
    sc_col1, sc_col2, sc_col3 = st.columns(3)

    scenarios_meta = [
        ("🌟 Best Case", matrix.best, "#059669", "Appreciation +1.5%, low vacancy, rate -50 bps, 0% haircut"),
        ("⚖️ Base Case", matrix.base, "#2563eb", "Benchmark appreciation & vacancy, baseline loan rate"),
        ("⚡ Worst Case (Stress)", matrix.worst, "#dc2626", "Appreciation -2.5%, worst vacancy, rate +200 bps, exit discount"),
    ]

    for col, (title, sc_res, border_col, desc) in zip([sc_col1, sc_col2, sc_col3], scenarios_meta):
        irr_val = f"{sc_res.equity_irr_pct:.2f}%" if sc_res.equity_irr_pct is not None else "N/A"
        cum_drain = sum(min(0.0, cf.net_cash_flow) for cf in sc_res.annual_cash_flows)
        with col:
            st.markdown(
                f"""
                <div class="scenario-card" style="border-top: 4px solid {border_col};">
                    <div>
                        <h3 style="margin:0 0 6px 0; font-size:1.18rem; color:#ffffff; font-weight:700;">{title}</h3>
                        <p style="margin:0 0 12px 0; font-size:0.82rem; color:#94a3b8; line-height:1.4;">{desc}</p>
                        <div style="margin-bottom:12px;"><span style="color:#94a3b8; font-size:0.88rem;">Equity IRR:</span> <span style="font-size:1.35rem; font-weight:800; color:{border_col}; margin-left:6px;">{irr_val}</span></div>
                    </div>
                    <div>
                        <div style="margin-bottom:6px; font-size:0.88rem; color:#cbd5e1;"><span style="color:#94a3b8;">Terminal Net Equity:</span> <b style="color:#ffffff;">{format_inr(sc_res.exit.net_exit_proceeds)}</b></div>
                        <div style="margin-bottom:6px; font-size:0.88rem; color:#cbd5e1;"><span style="color:#94a3b8;">Total Operational Drain:</span> <b style="color:#ffffff;">{format_inr(abs(cum_drain))}</b></div>
                        <div style="margin-bottom:6px; font-size:0.88rem; color:#cbd5e1;"><span style="color:#94a3b8;">Year-1 Net Cash Flow:</span> <b style="color:#ffffff;">{format_inr(sc_res.annual_cash_flows[0].net_cash_flow)}</b></div>
                        <div style="margin-bottom:6px; font-size:0.88rem; color:#cbd5e1;"><span style="color:#94a3b8;">NPV (@ Post-Tax FD):</span> <b style="color:#ffffff;">{format_inr(sc_res.npv_inr)}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Wealth Trajectory Chart
    wealth_fig = create_wealth_trajectory_chart(matrix, financing_obj.holding_period_years)
    st.plotly_chart(wealth_fig, width="stretch")
    re_end = base_res.exit.net_exit_proceeds
    fd_end = base_res.fd_benchmark.terminal_wealth_post_tax
    nifty_end = base_res.nifty_benchmark.terminal_wealth_post_tax
    winner_name, winner_val = max(
        [("Real Estate (Base)", re_end), ("SBI FD", fd_end), ("Nifty 50 TRI", nifty_end)],
        key=lambda x: x[1],
    )
    st.caption(
        f"💡 **Takeaway:** At Year {holding_period}, **{winner_name}** finishes highest at **{format_inr(winner_val)}** in hand "
        f"(Real Estate: **{format_inr(re_end)}** · SBI FD: **{format_inr(fd_end)}** · Nifty 50: **{format_inr(nifty_end)}**)."
    )

    # Opportunity Cost Summary Box
    fd_adv = base_res.fd_benchmark.net_wealth_advantage_inr
    nifty_adv = base_res.nifty_benchmark.net_wealth_advantage_inr
    st.markdown(
        f"""
        <div class="opp-cost-card">
            <h4 style="margin:0 0 12px 0; color:#ffffff; font-size:1.1rem; font-weight:700;">🍎 Apples-to-Apples Opportunity Cost Takeaway</h4>
            <div style="display:flex; gap:36px; flex-wrap:wrap;">
                <div>
                    <span style="color:#94a3b8; font-size:0.9rem;">Advantage vs SBI Term Deposit:</span><br>
                    <span style="color: {'#34d399' if fd_adv >= 0 else '#f87171'}; font-weight:800; font-size:1.35rem;">
                        {('+' if fd_adv >= 0 else '') + format_inr(fd_adv)}
                    </span>
                    <span style="font-size:0.84rem; color:#94a3b8; margin-left:6px;"> (IRR Spread: <b style="color:{'#34d399' if base_res.fd_benchmark.irr_spread_pct >= 0 else '#f87171'}">{base_res.fd_benchmark.irr_spread_pct:+.2f}%</b>)</span>
                </div>
                <div>
                    <span style="color:#94a3b8; font-size:0.9rem;">Advantage vs Nifty 50 TRI:</span><br>
                    <span style="color: {'#34d399' if nifty_adv >= 0 else '#f87171'}; font-weight:800; font-size:1.35rem;">
                        {('+' if nifty_adv >= 0 else '') + format_inr(nifty_adv)}
                    </span>
                    <span style="font-size:0.84rem; color:#94a3b8; margin-left:6px;"> (IRR Spread: <b style="color:{'#34d399' if base_res.nifty_benchmark.irr_spread_pct >= 0 else '#f87171'}">{base_res.nifty_benchmark.irr_spread_pct:+.2f}%</b>)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dedicated Stress Test Cards
    st.markdown("### 🧪 Dedicated Stress Test Cards")
    st_col1, st_col2, st_col3 = st.columns(3)

    with st_col1:
        s1 = matrix.stress_interest_rate_shock
        s1_badge = "badge-green" if s1.status == "PASS" else ("badge-amber" if s1.status == "WARN" else "badge-red")
        st.markdown(
            f"""
            <div class="stress-card" style="border-top:3px solid #38bdf8;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; font-size:1.05rem; color:#ffffff; font-weight:700;">{s1.name}</h4>
                        <span class="badge {s1_badge}">{s1.status}</span>
                    </div>
                    <p style="font-size:0.82rem; color:#94a3b8; margin:0 0 12px 0; line-height:1.4;">{s1.description}</p>
                    <div style="font-size:0.86rem; margin-bottom:5px; color:#cbd5e1;"><span style="color:#94a3b8;">+100 bps EMI:</span> <b style="color:#ffffff;">{format_inr(s1.metrics['emi_plus_100bps'])}</b> <span style="color:#38bdf8;">(+{format_inr(s1.metrics['delta_emi_100bps'])})</span></div>
                    <div style="font-size:0.86rem; margin-bottom:5px; color:#cbd5e1;"><span style="color:#94a3b8;">+200 bps EMI:</span> <b style="color:#ffffff;">{format_inr(s1.metrics['emi_plus_200bps'])}</b> <span style="color:#f87171;">(+{format_inr(s1.metrics['delta_emi_200bps'])})</span></div>
                    <div style="font-size:0.86rem; margin-bottom:8px; color:#cbd5e1;"><span style="color:#94a3b8;">Stressed FOIR:</span> <b style="color:{'#34d399' if s1.metrics['stressed_foir_200bps'] <= 40 else ('#fbbf24' if s1.metrics['stressed_foir_200bps'] <= 50 else '#f87171')};">{s1.metrics['stressed_foir_200bps']:.1f}%</b></div>
                </div>
                <ul style="margin:8px 0 0 0; padding-left:18px; font-size:0.80rem; color:#94a3b8; line-height:1.4;">
                    {''.join(f'<li>{n}</li>' for n in s1.notes)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st_col2:
        s2 = matrix.stress_vacancy_shock
        s2_badge = "badge-green" if s2.status == "PASS" else ("badge-amber" if s2.status == "WARN" else "badge-red")
        st.markdown(
            f"""
            <div class="stress-card" style="border-top:3px solid #fbbf24;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; font-size:1.05rem; color:#ffffff; font-weight:700;">{s2.name}</h4>
                        <span class="badge {s2_badge}">{s2.status}</span>
                    </div>
                    <p style="font-size:0.82rem; color:#94a3b8; margin:0 0 12px 0; line-height:1.4;">{s2.description}</p>
                    <div style="font-size:0.86rem; margin-bottom:5px; color:#cbd5e1;"><span style="color:#94a3b8;">Additional Cash Drain:</span> <b style="color:#f87171;">{format_inr(s2.metrics['vacancy_cash_drain'])}</b></div>
                    <div style="font-size:0.86rem; margin-bottom:8px; color:#cbd5e1;"><span style="color:#94a3b8;">Post-Shock Buffer:</span> <b style="color:{'#34d399' if s2.metrics['remaining_buffer_months'] >= 3 else '#f87171'};">{s2.metrics['remaining_buffer_months']:.1f} Months</b></div>
                </div>
                <ul style="margin:8px 0 0 0; padding-left:18px; font-size:0.80rem; color:#94a3b8; line-height:1.4;">
                    {''.join(f'<li>{n}</li>' for n in s2.notes)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st_col3:
        s3 = matrix.stress_forced_exit_year_3
        s3_badge = "badge-green" if s3.status == "PASS" else ("badge-amber" if s3.status == "WARN" else "badge-red")
        st.markdown(
            f"""
            <div class="stress-card" style="border-top:3px solid #f87171;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; font-size:1.05rem; color:#ffffff; font-weight:700;">{s3.name}</h4>
                        <span class="badge {s3_badge}">{s3.status}</span>
                    </div>
                    <p style="font-size:0.82rem; color:#94a3b8; margin:0 0 12px 0; line-height:1.4;">{s3.description}</p>
                    <div style="font-size:0.86rem; margin-bottom:5px; color:#cbd5e1;"><span style="color:#94a3b8;">Total Cash Invested (Yr 0-3):</span> <b style="color:#ffffff;">{format_inr(s3.metrics['total_cash_invested_3yr'])}</b></div>
                    <div style="font-size:0.86rem; margin-bottom:5px; color:#cbd5e1;"><span style="color:#94a3b8;">Net Recovered:</span> <b style="color:#ffffff;">{format_inr(s3.metrics['net_recovered_3yr'])}</b></div>
                    <div style="font-size:0.86rem; margin-bottom:8px; color:#cbd5e1;"><span style="color:#94a3b8;">Capital Recovery Ratio:</span> <b style="color:{'#34d399' if s3.metrics['capital_recovery_pct'] >= 80 else ('#fbbf24' if s3.metrics['capital_recovery_pct'] >= 60 else '#f87171')};">{s3.metrics['capital_recovery_pct']:.1f}%</b></div>
                </div>
                <ul style="margin:8px 0 0 0; padding-left:18px; font-size:0.80rem; color:#94a3b8; line-height:1.4;">
                    {''.join(f'<li>{n}</li>' for n in s3.notes)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Glass-box calculation proof & decision audit
    render_tab3_explainer(
        result=base_res,
        scenarios=matrix,
        stress_results=[
            matrix.stress_interest_rate_shock,
            matrix.stress_vacancy_shock,
            matrix.stress_forced_exit_year_3,
        ],
        prop_input=property_obj,
        investor_profile=investor_obj,
        zone_row=zone_row,
        macro=macro,
    )


# ---------------------------------------------------------
# TAB 4: PUNE MICRO-MARKET EXPLORER & COMPARABLES
# ---------------------------------------------------------
with tab4:
    st.markdown("### 🗺️ Pune Micro-Market Yield vs Valuation Matrix")
    st.caption("Interactive institutional benchmarking across all 43 Pune localities and 4 residential configurations.")

    scatter_fig = create_micromarket_scatter_chart(merged_df, sel_pincode, selected_bhk)
    st.plotly_chart(scatter_fig, width="stretch")
    mkt_rr_cur = (
        base_res.acquisition.agreement_value_inr / base_res.acquisition.rr_market_value
        if base_res.acquisition.rr_market_value > 0
        else 1.0
    )
    st.caption(
        f"💡 **How to Read This:** Top-left is best (higher rental yield at a lower markup over Govt Ready Reckoner). "
        f"**{sel_loc_name}** sits at **{base_res.year_1_gross_yield_pct:.2f}%** gross yield and **{mkt_rr_cur:.2f}×** Govt RR."
    )

    st.markdown("---")

    # Zone Risk Overview Table with INSTITUTIONAL_REPORT vs UNDERWRITING_ASSUMPTION tags
    st.markdown("#### 🛡️ Institutional Zone Risk Benchmarks (`pune_zone_risk_v2.csv`)")
    st.caption("Knight Frank H1 2026 overhang metrics joined with conservative underwriting haircut and delay models.")

    zone_display = zone_df.copy()
    zone_cols = [
        "pune_zone", "quarters_to_sell", "liquidity_tier", "exit_discount_stress_pct",
        "avg_uc_delay_months", "vacancy_months_best", "vacancy_months_avg", "vacancy_months_worst",
        "overhang_data_basis", "delay_data_basis"
    ]
    st.dataframe(zone_display[zone_cols], width="stretch", hide_index=True)

    # Macro parameters reference
    with st.expander("📜 Official Pune Statutory Macro Parameters (`pune_macro_constants_v2.csv`)", expanded=False):
        macro_df = pd.DataFrame(list(macro.items()), columns=["Parameter Name", "Value"])
        st.dataframe(macro_df, width="stretch", hide_index=True)

    # Glass-box calculation proof & decision audit
    render_tab4_explainer(
        loc_row=loc_row,
        unit_row=unit_row,
        zone_row=zone_row,
    )


# ---------------------------------------------------------
# TAB 5: SENSITIVITY, BREAK-EVEN & EXPECTED VALUE
# ---------------------------------------------------------
with tab5:
    st.markdown("### 🔬 Sensitivity, Break-Even & Expected Value")
    st.caption(
        "Institutional depth: what actually drives the outcome, where the deal ties "
        "its liquid alternatives, and a probability-weighted 'expected' answer you can "
        "present with confidence."
    )

    # Break-even appreciation vs liquid benchmarks
    be = compute_break_even_appreciation(property_obj, financing_obj, investor_obj, macro)
    be_c1, be_c2, be_c3 = st.columns(3)
    with be_c1:
        be_val = be["break_even_vs_fd_pct"]
        be_txt = f"{be_val:.2f}%" if be_val is not None else "> +30%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Break-even Appreciation · vs Post-Tax SBI FD</div>
                <div class="metric-value" style="color:#fbbf24;">{be_txt}</div>
                <div class="metric-sub">FD hurdle: {be['fd_post_tax_pct']:.2f}%/yr</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with be_c2:
        be_nifty = be["break_even_vs_nifty_pct"]
        be_nifty_txt = f"{be_nifty:.2f}%" if be_nifty is not None else "> +30%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Break-even Appreciation · vs Nifty 50 TRI</div>
                <div class="metric-value" style="color:#818cf8;">{be_nifty_txt}</div>
                <div class="metric-sub">Nifty post-tax IRR: {be['nifty_pct']:.2f}%/yr</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with be_c3:
        cur_appr = property_obj.appreciation_rate * 100.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Your Locality Expected Appreciation</div>
                <div class="metric-value">{cur_appr:.2f}%</div>
                <div class="metric-sub">Used as the Base-case assumption</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "If the locality's expected appreciation is *above* the break-even, the deal "
        "is statistically likely to beat that benchmark on a cash-flow-parity basis."
    )

    st.markdown("---")

    # Tornado: what drives the outcome
    st.markdown("### 🔀 Value Drivers — Relative Impact")
    tornado_items = compute_tornado_data(property_obj, financing_obj, investor_obj, macro)
    st.plotly_chart(create_tornado_chart(tornado_items), width="stretch")
    if tornado_items:
        top_drv = tornado_items[0]
        st.caption(
            f"💡 **Takeaway:** Your **#1 profit driver is {top_drv['label']}**, which swings your annual return from "
            f"**{top_drv['low']:.2f}%** (downside) to **{top_drv['high']:.2f}%** (upside)."
        )

    st.markdown("---")

    # Sensitivity heatmap
    st.markdown("### 🌡️ Appreciation × Rent Escalation Sensitivity")
    appr_axis, esc_axis, irr_grid = compute_sensitivity_grid(
        property_obj, financing_obj, investor_obj, macro
    )
    st.plotly_chart(
        create_sensitivity_heatmap(
            appr_axis,
            esc_axis,
            irr_grid,
            current_appr_pct=property_obj.appreciation_rate * 100.0,
            current_esc_pct=financing_obj.annual_rent_escalation_pct * 100.0,
        ),
        width="stretch",
    )
    st.caption(
        "Find your locality's expected appreciation on the vertical axis and your "
        "assumed rent growth on the horizontal axis — the colour is the equity IRR "
        "you should expect across that combination."
    )

    st.markdown("---")

    # Probability-weighted expected value
    st.markdown("### 🎲 Probability-Weighted Expected Value")
    st.caption(
        "Assign your own likelihood to the Best / Base / Worst scenarios; the engine "
        "folds them into one expected answer."
    )
    pr_c1, pr_c2, pr_c3 = st.columns(3)
    with pr_c1:
        p_best = st.slider("Probability · Best case (%)", 0, 100, 25, 5)
    with pr_c2:
        p_base = st.slider("Probability · Base case (%)", 0, 100, 50, 5)
    with pr_c3:
        p_worst = st.slider("Probability · Worst case (%)", 0, 100, 25, 5)

    ev = compute_probability_weighted(
        property_obj, financing_obj, investor_obj, macro,
        p_best / 100.0, p_base / 100.0, p_worst / 100.0,
    )

    ev_cols = st.columns(4)
    with ev_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Terminal Wealth</div>
                <div class="metric-value">{format_inr(ev.expected_terminal_wealth)}</div>
                <div class="metric-sub">vs {format_inr(ev.expected_total_cash_invested)} invested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ev_cols[1]:
        ev_irr = f"{ev.expected_equity_irr_pct:.2f}%" if ev.expected_equity_irr_pct is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Equity IRR</div>
                <div class="metric-value">{ev_irr}</div>
                <div class="metric-sub">FD {ev.fd_post_tax_pct:.2f}% · Nifty {ev.nifty_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ev_cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">P(Beat Post-Tax FD)</div>
                <div class="metric-value" style="color:{'#34d399' if ev.p_beat_fd >= 0.5 else '#f87171'};">{ev.p_beat_fd*100:.0f}%</div>
                <div class="metric-sub">Probability-weighted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ev_cols[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">P(Beat Nifty 50 TRI)</div>
                <div class="metric-value" style="color:{'#34d399' if ev.p_beat_nifty >= 0.5 else '#f87171'};">{ev.p_beat_nifty*100:.0f}%</div>
                <div class="metric-sub">Probability-weighted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.plotly_chart(create_expected_value_chart(ev), width="stretch")
    ev_irr_cap = f"{ev.expected_equity_irr_pct:.2f}%/yr" if ev.expected_equity_irr_pct is not None else "N/A"
    st.caption(
        f"💡 **Takeaway:** Weighting Best ({p_best}%), Base ({p_base}%), and Worst ({p_worst}%) outcomes gives an expected return of "
        f"**{ev_irr_cap}** and **{format_inr(ev.expected_terminal_wealth)}** in hand."
    )

    st.markdown("---")

    # Return decomposition
    try:
        import src.advanced_analysis as _adv_mod
        importlib.reload(_adv_mod)
    except Exception:
        pass
    decomp = compute_return_decomposition(base_res)
    total_cap_inv = decomp.get("total_capital_invested", decomp.get("total_equity_invested", 0.0))
    net_cg = decomp.get("net_post_tax_capital_gain", decomp.get("capital_gain", 0.0))
    cum_noi = decomp.get("cumulative_noi", decomp.get("total_rent_collected", 0.0))
    cum_int_tax = decomp.get("cumulative_interest_and_tax", 0.0)
    net_wealth = decomp.get("net_wealth_created", base_res.exit.net_exit_proceeds - total_cap_inv)

    st.markdown("### 🧮 Where Your Return Comes From (Exact Net Wealth Bridge)")
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    with rc1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Capital Invested</div>
                <div class="metric-value">{format_inr(total_cap_inv)}</div>
                <div class="metric-sub">Upfront + cumulative deficits</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Post-Tax Capital Gain</div>
                <div class="metric-value" style="color:{'#34d399' if net_cg >= 0 else '#f87171'};">{format_inr(net_cg)}</div>
                <div class="metric-sub">After all friction & LTCG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rc3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Cumulative NOI</div>
                <div class="metric-value" style="color:#34d399;">{format_inr(cum_noi)}</div>
                <div class="metric-sub">Net rent after vacancy & opex</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rc4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Less: Interest & Tax</div>
                <div class="metric-value" style="color:#f87171;">-{format_inr(cum_int_tax)}</div>
                <div class="metric-sub">Loan interest + property tax</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rc5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Net Wealth Created</div>
                <div class="metric-value" style="color:#38bdf8; font-weight:800;">{format_inr(net_wealth)}</div>
                <div class="metric-sub">Additive reconciliation (±₹0)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Methodology glossary collapsible
    with st.expander("📖 Financial Glossary — What Every Term Means", expanded=False):
        for term, definition in FINANCIAL_GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")
