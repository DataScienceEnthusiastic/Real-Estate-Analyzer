from __future__ import annotations

import numpy as np
import pytest

from src.advanced_analysis import (
    compute_break_even_appreciation,
    compute_probability_weighted,
    compute_return_decomposition,
    compute_sensitivity_grid,
    compute_tornado_data,
)
from src.data_loader import get_property_market_defaults, load_macro_constants
from src.finance_engine import run_full_underwriting_matrix
from src.models import FinancingInput, InvestorProfile


def _fixture():
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
    return macro, prop, financing, investor


def test_break_even_appreciation():
    macro, prop, financing, investor = _fixture()
    be = compute_break_even_appreciation(prop, financing, investor, macro)
    assert set(be) >= {"fd_post_tax_pct", "nifty_pct", "break_even_vs_fd_pct", "break_even_vs_nifty_pct"}
    assert be["fd_post_tax_pct"] > 0.0
    # FD hurdle is low, so a crossing should exist within the band
    assert be["break_even_vs_fd_pct"] is not None
    # Nifty is a high bar; crossing should still exist (or be reported None)
    assert be["break_even_vs_nifty_pct"] is None or be["break_even_vs_nifty_pct"] <= 30.0


def test_sensitivity_grid_shapes():
    macro, prop, financing, investor = _fixture()
    appr, esc, grid = compute_sensitivity_grid(prop, financing, investor, macro, appr_steps=5, esc_steps=5)
    assert appr.shape == (5,)
    assert esc.shape == (5,)
    assert grid.shape == (5, 5)
    finite = grid[np.isfinite(grid)]
    assert finite.size > 0
    # IRR should generally rise with appreciation (monotonic along each column)
    col = grid[:, 0]
    col_finite = col[np.isfinite(col)]
    assert np.all(np.diff(col_finite) >= -0.5)  # near-monotonic tolerance


def test_tornado_data():
    macro, prop, financing, investor = _fixture()
    items = compute_tornado_data(prop, financing, investor, macro)
    assert isinstance(items, list)
    assert len(items) >= 1
    for it in items:
        assert "label" in it and "low" in it and "high" in it and "base" in it
        assert it["high"] >= it["low"] - 1e-9


def test_probability_weighted_extremes():
    macro, prop, financing, investor = _fixture()
    # All mass on Best -> expected terminal equals best terminal
    ev_best = compute_probability_weighted(prop, financing, investor, macro, 1.0, 0.0, 0.0)
    assert ev_best.expected_terminal_wealth == pytest.approx(ev_best.best_terminal_wealth, rel=1e-6)
    assert ev_best.p_beat_fd in (0.0, 1.0)

    ev_mix = compute_probability_weighted(prop, financing, investor, macro, 0.25, 0.50, 0.25)
    # Expected terminal should sit between worst and best
    assert ev_mix.worst_terminal_wealth <= ev_mix.expected_terminal_wealth <= ev_mix.best_terminal_wealth
    assert 0.0 <= ev_mix.p_beat_fd <= 1.0
    assert 0.0 <= ev_mix.p_beat_nifty <= 1.0
    assert ev_mix.expected_total_cash_invested > 0.0


def test_return_decomposition():
    macro, prop, financing, investor = _fixture()
    matrix = run_full_underwriting_matrix(prop, financing, investor, macro)
    decomp = compute_return_decomposition(matrix.base)
    assert decomp["total_equity_invested"] > 0.0
    assert decomp["total_rent_collected"] >= 0.0
    assert 0.0 <= decomp["net_rental_yield_pct"] <= 30.0
    assert -1.0 < decomp["appreciation_cagr_pct"] < 1.0
