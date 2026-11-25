from __future__ import annotations

import pytest

from src.advanced_analysis import (
    compute_break_even_appreciation,
    compute_probability_weighted,
    compute_return_decomposition,
)
from src.data_loader import (
    get_property_market_defaults,
    load_locality_master,
    load_macro_constants,
)
from src.finance_engine import (
    run_full_underwriting_matrix,
    run_scenario_underwriting,
)
from src.models import FinancingInput, InvestorProfile, PropertyInput
from src.recommender import rank_top_localities


def _baner_fixture():
    macro = load_macro_constants()
    prop = get_property_market_defaults("411045", "Baner/Balewadi", "2_BHK")
    fin = FinancingInput(
        holding_period_years=10,
        desired_ltv_pct=0.75,
        monthly_society_maintenance_inr=round(prop.expected_sba_sqft * 4.0),
        annual_property_tax_inr=round(prop.carpet_area_sqft * prop.rr_rate_primary_sqft * 0.0025),
    )
    inv = InvestorProfile(
        net_monthly_income_inr=250_000.0,
        monthly_living_expenses_inr=75_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=6_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="new",
    )
    return macro, prop, fin, inv


def test_advanced_analysis_uses_post_tax_nifty_irr():
    macro, prop, fin, inv = _baner_fixture()
    mat = run_full_underwriting_matrix(prop, fin, inv, macro)
    be = compute_break_even_appreciation(prop, fin, inv, macro)
    ev = compute_probability_weighted(prop, fin, inv, macro)

    expected_nifty_post_tax = mat.base.nifty_benchmark.post_tax_irr_pct
    assert expected_nifty_post_tax < 13.48
    assert be["nifty_pct"] == pytest.approx(expected_nifty_post_tax, abs=1e-4)
    assert ev.nifty_pct == pytest.approx(expected_nifty_post_tax, abs=1e-4)


def test_uc_pre_emi_5yr_amortization_and_sec_71b_carryforward():
    macro = load_macro_constants()
    prop = PropertyInput(
        pincode="411045",
        locality_name="Baner/Balewadi",
        pune_zone="West",
        jurisdiction="PMC",
        bhk="2_BHK",
        use_case="investment_rental",
        property_status="under_construction",
        expected_possession_months=24,
        agreement_value_inr=10_000_000.0,
        carpet_area_sqft=850.0,
        monthly_rent_inr=35_000.0,
        appreciation_rate=0.08,
    )
    fin = FinancingInput(holding_period_years=10, desired_ltv_pct=0.75, annual_rent_escalation_pct=0.08)
    inv_old = InvestorProfile(
        net_monthly_income_inr=300_000.0,
        monthly_living_expenses_inr=70_000.0,
        existing_monthly_emis_inr=0.0,
        total_liquid_savings_inr=8_000_000.0,
        marginal_tax_slab_pct=0.30,
        tax_regime="old",
    )
    res = run_scenario_underwriting(prop, fin, inv_old, macro, 0.08, 1.0, 0.075, 0.0, scenario_uc_delay_months=0)

    # Years 1 and 2 are fully under construction -> no current-year tax deduction
    assert res.annual_cash_flows[0].tax_impact == 0.0
    assert res.annual_cash_flows[1].tax_impact == 0.0
    assert res.annual_cash_flows[0].pre_emi_amort_deduction == 0.0

    # Years 3..7 amortize 1/5th of accumulated pre-construction interest
    total_pre_emi_interest = res.annual_cash_flows[0].interest_paid + res.annual_cash_flows[1].interest_paid
    expected_installment = total_pre_emi_interest / 5.0
    assert res.annual_cash_flows[2].pre_emi_amort_deduction == pytest.approx(expected_installment, abs=1.0)
    assert res.annual_cash_flows[6].pre_emi_amort_deduction == pytest.approx(expected_installment, abs=1.0)
    assert res.annual_cash_flows[7].pre_emi_amort_deduction == 0.0


def test_realistic_friction_surcharge_and_reconciliation():
    macro, prop, fin, inv = _baner_fixture()
    fin.capex_and_turnover_reserve_pct = 0.05
    fin.loan_origination_and_motd_pct = 0.0035
    fin.mean_revert_appreciation_after_yr5 = True
    inv.surcharge_rate_pct = 0.10

    mat = run_full_underwriting_matrix(prop, fin, inv, macro)
    base = mat.base

    # Wealth trajectory end point still reconciles strictly to net_exit_proceeds
    assert base.net_realizable_equity_trajectory[-1] == pytest.approx(base.exit.net_exit_proceeds, abs=1.0)
    # Scenario ordering strictly preserved under mean reversion
    assert mat.best.equity_irr_pct >= mat.base.equity_irr_pct >= mat.worst.equity_irr_pct
    # Real IRR & DSCR populated
    assert base.real_equity_irr_pct is not None and base.real_equity_irr_pct < base.equity_irr_pct
    assert base.year_1_dscr is not None and 0.0 < base.year_1_dscr < 1.0
    # Return decomposition reconciles to 0 discrepancy even with MOTD & CapEx reserve
    d = compute_return_decomposition(base)
    reconciled = d["net_post_tax_capital_gain"] + d["cumulative_noi"] - d["cumulative_loan_interest"] - d["cumulative_operating_tax"]
    assert abs(d["net_wealth_created"] - reconciled) < 1.0


def test_recommender_ranks_top_3_aligned_localities():
    macro, _, fin, inv = _baner_fixture()
    fin.capex_and_turnover_reserve_pct = 0.05
    fin.loan_origination_and_motd_pct = 0.0035
    fin.mean_revert_appreciation_after_yr5 = True

    loc_df = load_locality_master()
    top3 = rank_top_localities(
        bhk="2_BHK",
        use_case="investment_rental",
        property_status="ready_resale",
        buyer_ownership="sole_male",
        brokerage_enabled=True,
        sell_brokerage_enabled=True,
        expected_possession_months=24,
        financing_template=fin,
        investor=inv,
        macro=macro,
        loc_df=loc_df,
        top_n=3,
    )
    assert len(top3) == 3
    assert [r.rank for r in top3] == [1, 2, 3]
    assert top3[0].alignment_score >= top3[1].alignment_score >= top3[2].alignment_score
    for rec in top3:
        assert len(rec.pincode) == 6
        assert rec.upfront_cash_inr <= inv.total_liquid_savings_inr
        assert len(rec.alignment_reasons) == 3


def test_pdf_includes_all_zones_top3_report_when_provided():
    import io
    import pypdf
    from src.decision_gates import evaluate_master_verdict
    from src.pdf_generator import generate_underwriting_pdf

    macro, prop, fin, inv = _baner_fixture()
    fin.capex_and_turnover_reserve_pct = 0.05
    fin.loan_origination_and_motd_pct = 0.0035
    fin.mean_revert_appreciation_after_yr5 = True

    mat = run_full_underwriting_matrix(prop, fin, inv, macro)
    verdict = evaluate_master_verdict(inv, prop, fin, mat)
    loc_df = load_locality_master()
    all_ranked = rank_top_localities(
        bhk="2_BHK",
        use_case="investment_rental",
        property_status="ready_resale",
        buyer_ownership="sole_male",
        brokerage_enabled=True,
        sell_brokerage_enabled=True,
        expected_possession_months=24,
        financing_template=fin,
        investor=inv,
        macro=macro,
        loc_df=loc_df,
        top_n=3,
        include_all=True,
    )

    pdf_bytes = generate_underwriting_pdf(
        property_obj=prop,
        financing_obj=fin,
        investor_obj=inv,
        matrix=mat,
        verdict=verdict,
        loc_row={},
        unit_row={},
        zone_row={},
        macro=macro,
        all_zone_recommendations=all_ranked,
    )
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    full_text = "".join((p.extract_text() or "") for p in reader.pages)
    assert "■" not in full_text
    assert "10. Smart Micro-Market Finder" in full_text
    for rec in all_ranked[:3]:
        assert rec.pincode in full_text
        assert rec.locality_name in full_text

