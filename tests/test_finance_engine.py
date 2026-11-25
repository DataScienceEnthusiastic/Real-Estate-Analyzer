from __future__ import annotations

import math
import numpy as np
import pytest

from src.data_loader import (
    load_locality_master,
    load_macro_constants,
    load_unit_market_data,
    load_zone_risk,
    get_property_market_defaults,
)
from src.finance_engine import (
    calculate_monthly_emi,
    compute_acquisition_costs,
    compute_loan_breakdown,
    compute_rbi_max_ltv,
    run_full_underwriting_matrix,
    run_scenario_underwriting,
)
from src.decision_gates import evaluate_master_verdict
from src.models import FinancingInput, InvestorProfile, PropertyInput


# Test 1: CSV Integrity
def test_csv_integrity():
    macro = load_macro_constants()
    assert len(macro) == 21, f"Expected 21 macro constants, got {len(macro)}"
    assert "stamp_duty_urban_male_pct" in macro
    assert "liquid_benchmark_nifty50_cagr_pct" in macro

    zones = load_zone_risk()
    assert len(zones) == 5, f"Expected 5 zones, got {len(zones)}"
    assert set(zones["pune_zone"].unique()) == {"Central", "East", "West", "South", "North_PCMC"}

    localities = load_locality_master()
    assert len(localities) == 43, f"Expected 43 localities, got {len(localities)}"
    assert all(isinstance(p, str) and len(p) == 6 for p in localities["pincode"]), "All pincodes must be 6-digit strings"

    units = load_unit_market_data()
    assert len(units) == 172, f"Expected 172 units, got {len(units)}"
    assert all(isinstance(p, str) and len(p) == 6 for p in units["pincode"]), "All unit pincodes must be strings"
    assert set(units["bhk"].unique()) == {"1_BHK", "2_BHK", "3_BHK", "4_PLUS_BHK"}


# Test 2: Stamp Duty & Registration Floor
def test_stamp_duty_and_registration_floor():
    macro = load_macro_constants()
    financing = FinancingInput()

    # Case A: Urban Male with agreement value > RR value
    # Agreement 1 Cr, RR rate 5,000, carpet 1000 sqft -> RR value 50L. Base is 1 Cr.
    prop_urban_male = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=1000.0,
        rr_rate_primary_sqft=5000.0,
        buyer_ownership="sole_male",
        brokerage_enabled=False,
    )
    acq_a = compute_acquisition_costs(prop_urban_male, financing, macro)
    assert acq_a.stamp_duty_base == 10_000_000.0
    assert acq_a.effective_stamp_duty_pct == pytest.approx(0.07, rel=1e-4)
    assert acq_a.stamp_duty_inr == pytest.approx(700_000.0, rel=1e-4)
    # Registration fee: 1% of 1 Cr is 1,00,000, capped at 30,000
    assert acq_a.registration_fee_inr == 30_000.0

    # Case B: Ready Reckoner statutory floor binds
    # Agreement 40L, RR rate 10,000, carpet 500 sqft -> RR value 50L. Base is 50L.
    prop_floor = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="1_BHK",
        agreement_value_inr=4_000_000.0,
        carpet_area_sqft=500.0,
        rr_rate_primary_sqft=10_000.0,
        buyer_ownership="sole_male",
        brokerage_enabled=False,
    )
    acq_b = compute_acquisition_costs(prop_floor, financing, macro)
    assert acq_b.stamp_duty_base == 5_000_000.0, "Stamp duty base must equal RR market value when it exceeds agreement value"
    assert acq_b.stamp_duty_inr == pytest.approx(5_000_000.0 * 0.07, rel=1e-4)

    # Case C: Sole Female rebate (7% - 1% = 6%)
    prop_female = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=1000.0,
        rr_rate_primary_sqft=5000.0,
        buyer_ownership="sole_female",
        brokerage_enabled=False,
    )
    acq_c = compute_acquisition_costs(prop_female, financing, macro)
    assert acq_c.effective_stamp_duty_pct == pytest.approx(0.06, rel=1e-4)
    assert acq_c.stamp_duty_inr == pytest.approx(600_000.0, rel=1e-4)

    # Case D: Joint Male-Female 50:50 rebate (7% - 0.5% = 6.5%)
    prop_joint = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=1000.0,
        rr_rate_primary_sqft=5000.0,
        buyer_ownership="male_female_50_50",
        brokerage_enabled=False,
    )
    acq_d = compute_acquisition_costs(prop_joint, financing, macro)
    assert acq_d.effective_stamp_duty_pct == pytest.approx(0.065, rel=1e-4)
    assert acq_d.stamp_duty_inr == pytest.approx(650_000.0, rel=1e-4)

    # Case E: Rural jurisdiction (PMRDA_Rural) male -> 4%
    prop_rural = PropertyInput(
        pincode="410501",
        locality_name="Chakan",
        pune_zone="North_PCMC",
        jurisdiction="PMRDA_Rural",
        bhk="2_BHK",
        agreement_value_inr=5_000_000.0,
        carpet_area_sqft=800.0,
        rr_rate_primary_sqft=2000.0,
        buyer_ownership="sole_male",
        brokerage_enabled=False,
    )
    acq_e = compute_acquisition_costs(prop_rural, financing, macro)
    assert acq_e.effective_stamp_duty_pct == pytest.approx(0.04, rel=1e-4)
    assert acq_e.stamp_duty_inr == pytest.approx(200_000.0, rel=1e-4)

    # Case F: Registration fee cap for small property <= 30L (1% uncapped)
    prop_small = PropertyInput(
        pincode="410501",
        locality_name="Chakan",
        pune_zone="North_PCMC",
        jurisdiction="PMRDA_Rural",
        bhk="1_BHK",
        agreement_value_inr=2_000_000.0,
        carpet_area_sqft=400.0,
        rr_rate_primary_sqft=2000.0,
        buyer_ownership="sole_male",
        brokerage_enabled=False,
    )
    acq_f = compute_acquisition_costs(prop_small, financing, macro)
    assert acq_f.registration_fee_inr == 20_000.0, "1% of 20L is 20,000, under the 30k cap"


# Test 3: TDS Non-Addition Rule
def test_tds_non_addition_rule():
    macro = load_macro_constants()
    financing = FinancingInput(interiors_and_misc_inr=0.0)

    # 1 Crore property: TDS is 1% = 1,00,000
    prop = PropertyInput(
        pincode="411001",
        locality_name="Camp",
        pune_zone="Central",
        jurisdiction="Cantonment",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=800.0,
        rr_rate_primary_sqft=8000.0,
        buyer_ownership="sole_male",
        brokerage_enabled=False,
    )
    acq = compute_acquisition_costs(prop, financing, macro)
    assert acq.tds_applicable is True
    assert acq.tds_amount_inr == 100_000.0

    # all-in cost MUST NOT include TDS
    expected_all_in = (
        acq.agreement_value_inr
        + acq.stamp_duty_inr
        + acq.registration_fee_inr
        + acq.gst_inr
        + acq.buy_brokerage_inr
        + acq.interiors_and_misc_inr
    )
    assert acq.all_in_acquisition_cost == expected_all_in
    assert acq.all_in_acquisition_cost != expected_all_in + acq.tds_amount_inr


# Test 4: RBI LTV Caps
def test_rbi_ltv_caps():
    macro = load_macro_constants()

    # <= 30 Lakhs -> 90%
    assert compute_rbi_max_ltv(2_500_000.0, macro) == 0.90
    assert compute_rbi_max_ltv(3_000_000.0, macro) == 0.90

    # > 30 Lakhs and <= 75 Lakhs -> 80%
    assert compute_rbi_max_ltv(3_000_001.0, macro) == 0.80
    assert compute_rbi_max_ltv(5_000_000.0, macro) == 0.80
    assert compute_rbi_max_ltv(7_500_000.0, macro) == 0.80

    # > 75 Lakhs -> 75%
    assert compute_rbi_max_ltv(7_500_001.0, macro) == 0.75
    assert compute_rbi_max_ltv(15_000_000.0, macro) == 0.75


# Test 5: Amortization Closure & Zero Loan Handling
def test_amortization_closure_and_zero_loan():
    macro = load_macro_constants()

    # Subtest 5A: Zero loan amount
    prop = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=1000.0,
        rr_rate_primary_sqft=5000.0,
    )
    financing_zero_ltv = FinancingInput(desired_ltv_pct=0.0, loan_tenure_years=20, holding_period_years=10)
    acq = compute_acquisition_costs(prop, financing_zero_ltv, macro)
    loan_zero = compute_loan_breakdown(acq, financing_zero_ltv, prop, macro)

    assert loan_zero.loan_amount == 0.0
    assert loan_zero.monthly_emi == 0.0
    assert loan_zero.remaining_loan_balance_H == 0.0
    assert all(row["annual_emi"] == 0.0 for row in loan_zero.annual_schedule)

    # Subtest 5B: Standard loan amortizes to 0 at final month
    loan_amt = 5_000_000.0
    rate = 0.075
    tenure = 20
    sched, annual_sched, emi, _, _ = compute_loan_breakdown(
        acq,
        FinancingInput(desired_ltv_pct=0.50, home_loan_interest_rate_pct=rate, loan_tenure_years=tenure, holding_period_years=tenure),
        prop,
        macro,
    ).monthly_schedule, None, None, None, None

    # Check that loan closing balance at month 240 is 0 (± 1 INR due to floating rounding)
    assert sched[239]["month"] == 240
    assert sched[239]["closing_balance"] == pytest.approx(0.0, abs=1.0)


# Test 6: Chakan 4_PLUS_BHK NULL Rent Handling
def test_chakan_null_rent_handling():
    # Load defaults for Chakan 4_PLUS_BHK
    prop = get_property_market_defaults(pincode="410501", locality_name="Chakan", bhk="4_PLUS_BHK")
    assert prop.monthly_rent_inr == 0.0, "Missing rent should safely default to 0.0 without throwing NaN exception"

    macro = load_macro_constants()
    financing = FinancingInput()
    investor = InvestorProfile(
        net_monthly_income_inr=150_000.0,
        monthly_living_expenses_inr=50_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=5_000_000.0,
    )

    # Underwriting should complete cleanly without throwing NaN or ZeroDivisionError
    res = run_scenario_underwriting(
        property_input=prop,
        financing_input=financing,
        investor=investor,
        macro=macro,
        scenario_appreciation_rate=prop.appreciation_rate,
        scenario_vacancy_months=prop.vacancy_months_avg,
        scenario_loan_rate=0.075,
        scenario_exit_discount_pct=0.0,
    )
    assert res.year_1_gross_yield_pct == 0.0
    assert res.annual_cash_flows[0].gross_annual_rent == 0.0
    assert not math.isnan(res.acquisition.all_in_acquisition_cost)


# Test 7: Tax Regime Branching
def test_tax_regime_branching():
    macro = load_macro_constants()
    financing = FinancingInput(holding_period_years=5)

    # Create an investment deal with heavy interest paid creating a house property loss
    prop = PropertyInput(
        pincode="411004",
        locality_name="Deccan/Erandwane",
        pune_zone="Central",
        jurisdiction="PMC",
        bhk="2_BHK",
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=800.0,
        monthly_rent_inr=20_000.0,  # 2.4L gross rent
        use_case="investment_rental",
    )

    investor_new = InvestorProfile(
        net_monthly_income_inr=200_000.0,
        monthly_living_expenses_inr=60_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=5_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    investor_old = InvestorProfile(
        net_monthly_income_inr=200_000.0,
        monthly_living_expenses_inr=60_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=5_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="old",
    )

    res_new = run_scenario_underwriting(prop, financing, investor_new, macro, 0.06, 1.0, 0.075, 0.0)
    res_old = run_scenario_underwriting(prop, financing, investor_old, macro, 0.06, 1.0, 0.075, 0.0)

    # In Year 1, interest is ~5.6L, rent is ~2.4L. House property loss is substantial.
    # New regime blocks salary offset -> tax impact >= 0 (cannot be negative)
    assert res_new.annual_cash_flows[0].tax_impact >= 0.0

    # Old regime allows salary loss offset up to INR 2,00,000 -> tax impact is negative (saving)
    # 2,00,000 * 30% * 1.04 = -62,400
    assert res_old.annual_cash_flows[0].tax_impact == pytest.approx(-62_400.0, abs=10.0)


# Test 8: Scenario Ordering
def test_scenario_ordering():
    macro = load_macro_constants()
    prop = get_property_market_defaults(pincode="411045", locality_name="Baner/Balewadi", bhk="2_BHK")
    financing = FinancingInput(holding_period_years=10)
    investor = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=70_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )

    matrix = run_full_underwriting_matrix(prop, financing, investor, macro)

    best_irr = matrix.best.equity_irr_pct
    base_irr = matrix.base.equity_irr_pct
    worst_irr = matrix.worst.equity_irr_pct

    assert best_irr is not None and base_irr is not None and worst_irr is not None
    assert best_irr >= base_irr >= worst_irr, f"Expected Best ({best_irr:.2f}%) >= Base ({base_irr:.2f}%) >= Worst ({worst_irr:.2f}%)"

    # Terminal proceeds should also be ordered
    assert matrix.best.exit.net_exit_proceeds >= matrix.base.exit.net_exit_proceeds >= matrix.worst.exit.net_exit_proceeds


# Test 9: Decision Gates Evaluation
def test_decision_gates_master_verdict():
    macro = load_macro_constants()
    prop = get_property_market_defaults(pincode="411045", locality_name="Baner/Balewadi", bhk="2_BHK")
    financing = FinancingInput(holding_period_years=10)

    # Solid profile -> Should pass or caution
    investor_strong = InvestorProfile(
        net_monthly_income_inr=400_000.0,
        monthly_living_expenses_inr=80_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=15_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    matrix_strong = run_full_underwriting_matrix(prop, financing, investor_strong, macro)
    verdict_strong = evaluate_master_verdict(investor_strong, prop, financing, matrix_strong)
    assert verdict_strong.status in {"GO", "CAUTION"}

    # Broke profile -> Must fail Gate 1 and trigger NO-GO
    investor_broke = InvestorProfile(
        net_monthly_income_inr=50_000.0,
        monthly_living_expenses_inr=40_000.0,
        existing_monthly_emis_inr=20_000.0,
        total_liquid_savings_inr=500_000.0,  # Far below required upfront cash
    )
    matrix_broke = run_full_underwriting_matrix(prop, financing, investor_broke, macro)
    verdict_broke = evaluate_master_verdict(investor_broke, prop, financing, matrix_broke)
    assert verdict_broke.status == "NO-GO"
    assert verdict_broke.color == "red"
    assert verdict_broke.gates[0].status == "FAIL"


# Test 10: Glass-Box Explainer Proof Fields across 4 Edge Cases
def test_glass_box_explainer_proof_fields():
    macro = load_macro_constants()

    # (a) Baner/Balewadi 3_BHK investment_rental
    prop_a = get_property_market_defaults("411045", "Baner/Balewadi", "3_BHK")
    prop_a.use_case = "investment_rental"
    fin_a = FinancingInput(desired_ltv_pct=0.75, loan_tenure_years=20, holding_period_years=10)
    inv_a = InvestorProfile(
        net_monthly_income_inr=350_000.0,
        monthly_living_expenses_inr=70_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=8_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    mat_a = run_full_underwriting_matrix(prop_a, fin_a, inv_a, macro)
    res_a = mat_a.base
    dec_a = evaluate_master_verdict(inv_a, prop_a, fin_a, mat_a)

    assert not math.isnan(res_a.acquisition.rr_market_value)
    assert not math.isnan(res_a.acquisition.stamp_duty_base)
    assert not math.isnan(res_a.annual_cash_flows[0].nav)
    assert not math.isnan(res_a.annual_cash_flows[0].std_deduction)
    assert not math.isnan(res_a.annual_cash_flows[0].net_hp_income)
    assert not math.isnan(res_a.nifty_benchmark.gross_wealth)
    assert not math.isnan(res_a.nifty_benchmark.principal_invested)
    assert dec_a.status in {"GO", "CAUTION", "NO-GO"}

    # (b) self_use + under_construction + old regime
    prop_b = get_property_market_defaults("411045", "Baner/Balewadi", "2_BHK")
    prop_b.use_case = "self_use"
    prop_b.property_status = "under_construction"
    prop_b.expected_possession_months = 24
    fin_b = FinancingInput(desired_ltv_pct=0.75, loan_tenure_years=20, holding_period_years=10)
    inv_b = InvestorProfile(
        net_monthly_income_inr=300_000.0,
        monthly_living_expenses_inr=60_000.0,
        existing_monthly_emis_inr=15_000.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="old",
    )
    mat_b = run_full_underwriting_matrix(prop_b, fin_b, inv_b, macro)
    res_b = mat_b.base
    dec_b = evaluate_master_verdict(inv_b, prop_b, fin_b, mat_b)

    assert res_b.acquisition.gst_inr > 0, "GST must be computed for under-construction"
    assert res_b.loan.monthly_pre_emi > 0, "Pre-EMI must be computed for under-construction"
    assert not math.isnan(res_b.exit.tax_cost_basis)
    assert not math.isnan(res_b.exit.capital_gain)
    assert dec_b.status in {"GO", "CAUTION", "NO-GO"}

    # (c) loan_amount = 0 all-cash purchase
    prop_c = get_property_market_defaults("411045", "Baner/Balewadi", "1_BHK")
    fin_c = FinancingInput(desired_ltv_pct=0.0, loan_tenure_years=20, holding_period_years=10)
    inv_c = InvestorProfile(
        net_monthly_income_inr=500_000.0,
        monthly_living_expenses_inr=50_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=20_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    mat_c = run_full_underwriting_matrix(prop_c, fin_c, inv_c, macro)
    res_c = mat_c.base
    dec_c = evaluate_master_verdict(inv_c, prop_c, fin_c, mat_c)

    assert res_c.loan.loan_amount == 0.0
    assert res_c.loan.monthly_emi == 0.0
    assert res_c.loan.required_upfront_cash == res_c.acquisition.all_in_acquisition_cost
    assert not math.isnan(res_c.annual_cash_flows[0].net_cash_flow)
    assert dec_c.status in {"GO", "CAUTION", "NO-GO"}

    # (d) 410501 Chakan 4_PLUS_BHK NULL rent fallback
    prop_d = get_property_market_defaults("410501", "Chakan", "4_PLUS_BHK")
    assert prop_d.monthly_rent_inr == 0.0, "Chakan 4_PLUS_BHK must have 0.0 default rent"
    fin_d = FinancingInput(desired_ltv_pct=0.75, loan_tenure_years=20, holding_period_years=10)
    inv_d = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=50_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=5_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    mat_d = run_full_underwriting_matrix(prop_d, fin_d, inv_d, macro)
    res_d = mat_d.base
    dec_d = evaluate_master_verdict(inv_d, prop_d, fin_d, mat_d)

    assert not math.isnan(res_d.acquisition.all_in_acquisition_cost)
    assert not math.isnan(res_d.annual_cash_flows[0].noi)
    assert not math.isnan(res_d.exit.net_exit_proceeds)
    assert dec_d.status in {"GO", "CAUTION", "NO-GO"}


# Test 11: Full Dashboard CSV Export Integrity
def test_full_dashboard_csv_export():
    from src.export_engine import generate_full_dashboard_csv

    macro = load_macro_constants()
    prop = get_property_market_defaults("411045", "Baner/Balewadi", "2_BHK")
    fin = FinancingInput(holding_period_years=10)
    inv = InvestorProfile(
        net_monthly_income_inr=300_000.0,
        monthly_living_expenses_inr=60_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=10_000_000.0,
    )
    matrix = run_full_underwriting_matrix(prop, fin, inv, macro)
    verdict = evaluate_master_verdict(inv, prop, fin, matrix)

    csv_text = generate_full_dashboard_csv(
        property_obj=prop,
        financing_obj=fin,
        investor_obj=inv,
        matrix=matrix,
        verdict=verdict,
        loc_row={},
        unit_row={},
        zone_row={},
        macro=macro,
    )

    assert isinstance(csv_text, str)
    assert len(csv_text) > 5000
    lines = csv_text.splitlines()
    assert len(lines) > 100

    expected_sections = [
        "SECTION 1: INVESTOR FINANCIAL PROFILE",
        "SECTION 2: PROPERTY ASSUMPTIONS",
        "SECTION 3: MORTGAGE FINANCING",
        "SECTION 4: STATUTORY ACQUISITION COST WATERFALL",
        "SECTION 5: 4-GATE DECISION ENGINE AUDIT",
        "SECTION 6: YEAR-BY-YEAR OPERATING CASH FLOW",
        "SECTION 7: MULTI-SCENARIO UNDERWRITING MATRIX",
        "SECTION 8: DEDICATED STRESS TESTS AUDIT",
        "SECTION 9: OPPORTUNITY COST & BENCHMARK PARITY",
        "SECTION 10: SENSITIVITY, BREAK-EVEN & EXPECTED VALUE",
    ]
    for sec in expected_sections:
        assert any(sec in line for line in lines), f"Missing section: {sec}"


# Test 12: Institutional PDF Underwriting Report Export Integrity
def test_underwriting_pdf_export():
    from src.pdf_generator import generate_underwriting_pdf

    macro = load_macro_constants()
    prop = get_property_market_defaults("411045", "Baner/Balewadi", "2_BHK")
    fin = FinancingInput(holding_period_years=10)
    inv = InvestorProfile(
        net_monthly_income_inr=300_000.0,
        monthly_living_expenses_inr=60_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=10_000_000.0,
    )
    matrix = run_full_underwriting_matrix(prop, fin, inv, macro)
    verdict = evaluate_master_verdict(inv, prop, fin, matrix)

    pdf_bytes = generate_underwriting_pdf(
        property_obj=prop,
        financing_obj=fin,
        investor_obj=inv,
        matrix=matrix,
        verdict=verdict,
        loc_row={},
        unit_row={},
        zone_row={},
        macro=macro,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 10_000, f"PDF report too small: {len(pdf_bytes)} bytes"

    # Branch test: under-construction scenario with sole_female ownership & self_use
    prop_uc = PropertyInput(
        pincode="411057",
        locality_name="Wakad",
        pune_zone="West",
        jurisdiction="PCMC",
        bhk="3_BHK",
        property_status="under_construction",
        use_case="self_use",
        buyer_ownership="sole_female",
        agreement_value_inr=12_000_000.0,
        carpet_area_sqft=1100.0,
        expected_possession_months=24,
    )
    matrix_uc = run_full_underwriting_matrix(prop_uc, fin, inv, macro)
    verdict_uc = evaluate_master_verdict(inv, prop_uc, fin, matrix_uc)
    pdf_bytes_uc = generate_underwriting_pdf(
        property_obj=prop_uc,
        financing_obj=fin,
        investor_obj=inv,
        matrix=matrix_uc,
        verdict=verdict_uc,
        loc_row={},
        unit_row={},
        zone_row={},
        macro=macro,
    )
    assert isinstance(pdf_bytes_uc, bytes)
    assert pdf_bytes_uc.startswith(b"%PDF")
    assert len(pdf_bytes_uc) > 10_000


# Test 13: Wealth Chart Matches Net Exit Proceeds at Horizon (Patch 1)
def test_wealth_chart_matches_net_exit_at_horizon():
    macro = load_macro_constants()
    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    for loc, pcode in [("Akurdi", "411035"), ("Baner/Balewadi", "411045")]:
        p = get_property_market_defaults(pcode, loc, "2_BHK")
        maint = round(p.expected_sba_sqft * 4.0)
        ptax = round(p.agreement_value_inr * 0.0015)
        fin = FinancingInput(
            holding_period_years=10,
            desired_ltv_pct=0.75,
            monthly_society_maintenance_inr=maint,
            annual_property_tax_inr=ptax,
        )
        mat = run_full_underwriting_matrix(p, fin, inv, macro)
        base = mat.base
        assert abs(base.net_realizable_equity_trajectory[-1] - base.exit.net_exit_proceeds) < 1.0
        if loc == "Akurdi":
            assert base.net_realizable_equity_trajectory[-1] < base.fd_benchmark.terminal_wealth_post_tax


# Test 14: True Post-Tax Nifty 50 IRR on Parity Vector (Patch 2)
def test_nifty_post_tax_irr_below_pretax_cagr():
    macro = load_macro_constants()
    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    p = get_property_market_defaults("411045", "Baner/Balewadi", "2_BHK")
    fin = FinancingInput(holding_period_years=10, desired_ltv_pct=0.75)
    mat = run_full_underwriting_matrix(p, fin, inv, macro)
    base = mat.base

    nifty_post_tax_irr = base.nifty_benchmark.post_tax_irr_pct / 100.0
    assert nifty_post_tax_irr < 0.1348
    assert abs(nifty_post_tax_irr - 0.1233) < 0.015
    expected_spread = base.equity_irr_pct - base.nifty_benchmark.post_tax_irr_pct
    assert abs(base.nifty_benchmark.irr_spread_pct - expected_spread) < 0.01


# Test 15: Exact Reconciling Return Decomposition Net Wealth Bridge (Patch 4)
def test_return_decomposition_exact_reconciliation():
    from src.advanced_analysis import compute_return_decomposition

    macro = load_macro_constants()
    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    five_zones = [
        ("Baner/Balewadi", "411045"),
        ("Hadapsar/Magarpatta/Amanora", "411028"),
        ("Bibwewadi/Market Yard", "411037"),
        ("Camp", "411001"),
        ("Akurdi", "411035"),
    ]
    for loc, pcode in five_zones:
        p = get_property_market_defaults(pcode, loc, "2_BHK")
        maint = round(p.expected_sba_sqft * 4.0)
        ptax = round(p.agreement_value_inr * 0.0015)
        fin = FinancingInput(
            holding_period_years=10,
            desired_ltv_pct=0.75,
            monthly_society_maintenance_inr=maint,
            annual_property_tax_inr=ptax,
        )
        mat = run_full_underwriting_matrix(p, fin, inv, macro)
        d = compute_return_decomposition(mat.base)
        reconciled_sum = (
            d["net_post_tax_capital_gain"]
            + d["cumulative_noi"]
            - d["cumulative_loan_interest"]
            - d["cumulative_operating_tax"]
        )
        assert abs(d["net_wealth_created"] - reconciled_sum) < 1.0


# Test 16: Calibrated 4-Gate Decisions and Master Verdicts Across 5 Zones (Patch 2 & 3)
def test_calibrated_verdicts_across_5_zones():
    macro = load_macro_constants()
    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    test_cases = [
        ("Baner/Balewadi", "411045", "GO", "PASS", "PASS", "PASS", "PASS"),
        ("Hadapsar/Magarpatta/Amanora", "411028", "CAUTION", "PASS", "WARN", "WARN", None),
        ("Bibwewadi/Market Yard", "411037", "CAUTION", "PASS", "PASS", "WARN", None),
        ("Camp", "411001", "NO-GO", "PASS", "WARN", "PASS", "FAIL"),
        ("Akurdi", "411035", "NO-GO", "PASS", "PASS", "WARN", "FAIL"),
    ]
    for loc, pcode, exp_v, exp_g1, exp_g2, exp_g3, exp_g4 in test_cases:
        p = get_property_market_defaults(pcode, loc, "2_BHK")
        maint = round(p.expected_sba_sqft * 4.0)
        ptax = round(p.agreement_value_inr * 0.0015)
        fin = FinancingInput(
            holding_period_years=10,
            desired_ltv_pct=0.75,
            monthly_society_maintenance_inr=maint,
            annual_property_tax_inr=ptax,
        )
        mat = run_full_underwriting_matrix(p, fin, inv, macro)
        v = evaluate_master_verdict(inv, p, fin, mat)

        assert v.status == exp_v, f"{loc} expected {exp_v}, got {v.status}"
        assert v.gates[0].status == exp_g1, f"{loc} G1 expected {exp_g1}, got {v.gates[0].status}"
        assert v.gates[1].status == exp_g2, f"{loc} G2 expected {exp_g2}, got {v.gates[1].status}"
        if exp_g3 is not None:
            assert v.gates[2].status == exp_g3, f"{loc} G3 expected {exp_g3}, got {v.gates[2].status}"
        if exp_g4 is not None:
            assert v.gates[3].status == exp_g4, f"{loc} G4 expected {exp_g4}, got {v.gates[3].status}"
        if loc == "Camp":
            assert any("Cantonment" in r for r in v.gates[1].reasons), "Camp must include Cantonment title risk advisory"


# Test 17: Section 5 Base Haircut and Ready Resale Delay Labels (Patch 5)
def test_sec5_base_haircut_and_ready_resale_delay_labels():
    from src.export_engine import generate_full_dashboard_csv

    macro = load_macro_constants()
    loc_df = load_locality_master()
    zone_df = load_zone_risk()
    unit_df = load_unit_market_data()

    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    p = get_property_market_defaults("411035", "Akurdi", "2_BHK")
    fin = FinancingInput(holding_period_years=10, desired_ltv_pct=0.75)
    mat = run_full_underwriting_matrix(p, fin, inv, macro)
    v = evaluate_master_verdict(inv, p, fin, mat)

    loc_row = loc_df[loc_df["pincode"] == p.pincode].iloc[0]
    unit_row = unit_df[(unit_df["pincode"] == p.pincode) & (unit_df["bhk"] == p.bhk)].iloc[0]
    zone_row = zone_df[zone_df["pune_zone"] == p.pune_zone].iloc[0]

    csv_out = generate_full_dashboard_csv(p, fin, inv, mat, v, loc_row, unit_row, zone_row, macro)
    assert "Exit Haircut Discount,0.0%,0.0%" in csv_out, "Base Case exit haircut must be 0.0%"
    assert "Under-Construction Delay,N/A (Ready Resale),N/A (Ready Resale),N/A (Ready Resale)" in csv_out, (
        "Ready resale must render N/A (Ready Resale) for UC delay"
    )


# Test 18: Institutional PDF Export Smoke & Zero Missing Glyph (Patch 6)
def test_pdf_export_smoke_and_no_missing_glyph():
    import os
    import tempfile
    import pypdf
    from src.pdf_generator import generate_underwriting_pdf

    macro = load_macro_constants()
    loc_df = load_locality_master()
    zone_df = load_zone_risk()
    unit_df = load_unit_market_data()

    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    p = get_property_market_defaults("411035", "Akurdi", "2_BHK")
    maint = round(p.expected_sba_sqft * 4.0)
    ptax = round(p.agreement_value_inr * 0.0015)
    fin = FinancingInput(
        holding_period_years=10,
        desired_ltv_pct=0.75,
        monthly_society_maintenance_inr=maint,
        annual_property_tax_inr=ptax,
    )
    mat = run_full_underwriting_matrix(p, fin, inv, macro)
    v = evaluate_master_verdict(inv, p, fin, mat)

    loc_row = loc_df[loc_df["pincode"] == p.pincode].iloc[0]
    unit_row = unit_df[(unit_df["pincode"] == p.pincode) & (unit_df["bhk"] == p.bhk)].iloc[0]
    zone_row = zone_df[zone_df["pune_zone"] == p.pune_zone].iloc[0]

    pdf_bytes = generate_underwriting_pdf(p, fin, inv, mat, v, loc_row, unit_row, zone_row, macro)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        file_size = os.path.getsize(tmp_path)
        assert file_size > 20_000, f"Expected PDF > 20 KB, got {file_size} bytes"

        reader = pypdf.PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        assert "■" not in full_text, "Found missing glyph '■' in exported PDF text"
        assert "9. Institutional Data Provenance" in full_text
        assert "IGR_SL_PRIMARY" in full_text
        idx_heading = full_text.index("9. Institutional Data Provenance")
        idx_tag = full_text.index("IGR_SL_PRIMARY")
        assert idx_heading < idx_tag, "Section 9 heading must appear before IGR_SL_PRIMARY table content"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)




