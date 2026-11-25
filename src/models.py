from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InvestorProfile:
    """Investor financial profile for underwriting and gate safety checks."""
    net_monthly_income_inr: float
    monthly_living_expenses_inr: float
    existing_monthly_emis_inr: float
    total_liquid_savings_inr: float
    upcoming_cash_need_inr: float = 0.0
    emergency_buffer_months_target: float = 6.0
    marginal_tax_slab_pct: float = 0.30
    tax_regime: str = "new"  # "new" or "old"
    surcharge_rate_pct: float = 0.0  # 0.0, 0.10, 0.15, or 0.25 based on annual income bracket


@dataclass
class PropertyInput:
    """Selected property parameters, market defaults, and override assumptions."""
    pincode: str
    locality_name: str
    pune_zone: str
    jurisdiction: str
    bhk: str
    use_case: str = "investment_rental"  # "investment_rental" | "self_use"
    property_status: str = "ready_resale"  # "ready_resale" | "under_construction"
    buyer_ownership: str = "sole_male"  # "sole_male" | "sole_female" | "female_joint" | "male_female_50_50"
    agreement_value_inr: float = 0.0
    carpet_area_sqft: float = 0.0
    expected_sba_sqft: float = 0.0
    monthly_rent_inr: float = 0.0
    appreciation_rate: float = 0.06
    rr_rate_primary_sqft: float = 0.0
    liquidity_tier: str = "medium"  # "high" | "medium" | "low"
    quarters_to_sell: float = 4.8
    exit_discount_stress_pct: float = 0.08
    avg_uc_delay_months: float = 8.0
    vacancy_months_best: float = 0.5
    vacancy_months_avg: float = 1.0
    vacancy_months_worst: float = 2.5
    expected_possession_months: int = 24
    brokerage_enabled: bool = True
    sell_brokerage_enabled: bool = True


@dataclass
class FinancingInput:
    """Mortgage terms, holding period, and escalation assumptions."""
    desired_ltv_pct: float = 0.75
    home_loan_interest_rate_pct: float = 0.075
    loan_tenure_years: int = 20
    holding_period_years: int = 10
    annual_rent_escalation_pct: float = 0.05
    monthly_society_maintenance_inr: float = 0.0
    annual_property_tax_inr: float = 0.0
    interiors_and_misc_inr: float = 0.0
    nifty_benchmark_cagr_pct: float = 0.1348
    cpi_inflation_pct: float = 0.055
    capex_and_turnover_reserve_pct: float = 0.0
    loan_origination_and_motd_pct: float = 0.0
    mean_revert_appreciation_after_yr5: bool = False
    long_term_appreciation_anchor_pct: float = 0.065


@dataclass
class AcquisitionCostBreakdown:
    """Detailed statutory floor and all-in acquisition costs."""
    agreement_value_inr: float
    rr_market_value: float
    stamp_duty_base: float
    stamp_duty_inr: float
    effective_stamp_duty_pct: float
    registration_fee_inr: float
    gst_inr: float
    gst_rate: float
    is_affordable_gst: bool
    buy_brokerage_inr: float
    interiors_and_misc_inr: float
    all_in_acquisition_cost: float
    tds_applicable: bool
    tds_amount_inr: float


@dataclass
class LoanBreakdown:
    """RBI LTV cap, borrowing amount, upfront equity, and amortization."""
    rbi_max_ltv: float
    effective_ltv: float
    loan_amount: float
    required_upfront_cash: float
    monthly_emi: float
    monthly_pre_emi: float
    monthly_schedule: list[dict[str, float]]
    annual_schedule: list[dict[str, float]]
    remaining_loan_balance_H: float
    loan_origination_and_motd_inr: float = 0.0
    home_loan_interest_rate_pct: float = 0.075
    loan_tenure_years: int = 20
    holding_period_years: int = 10


@dataclass
class CashFlowYear:
    """Annual operating cash flow and tax impact for Year t."""
    year: int
    gross_annual_rent: float
    vacancy_loss: float
    egi: float
    opex: float
    noi: float
    annual_emi: float
    interest_paid: float
    principal_paid: float
    tax_impact: float
    net_cash_flow: float
    remaining_loan_balance: float
    net_equity: float
    net_realizable_equity: float = 0.0
    nav: float = 0.0
    std_deduction: float = 0.0
    net_hp_income: float = 0.0
    capex_reserve: float = 0.0
    pre_emi_amort_deduction: float = 0.0
    loss_cf_used: float = 0.0


@dataclass
class AlternativeBenchmarkResult:
    """Comparative returns and terminal wealth against liquid benchmarks."""
    benchmark_name: str
    annual_return_pct: float
    terminal_wealth_post_tax: float
    net_wealth_advantage_inr: float
    irr_spread_pct: float
    trajectory: list[float]
    gross_wealth: float = 0.0
    principal_invested: float = 0.0
    exit_tax: float = 0.0
    pre_tax_cagr_pct: float = 0.0
    post_tax_irr_pct: Optional[float] = None


@dataclass
class ExitValuation:
    """Holding period terminal exit value, taxes, and net proceeds."""
    terminal_gross_value: float
    exit_sale_price: float
    sell_brokerage_inr: float
    tax_cost_basis: float
    capital_gain: float
    is_ltcg: bool
    cg_tax_inr: float
    remaining_loan_balance_H: float
    net_exit_proceeds: float


@dataclass
class UnderwritingResult:
    """Complete financial underwriting outcome for a specific scenario."""
    acquisition: AcquisitionCostBreakdown
    loan: LoanBreakdown
    annual_cash_flows: list[CashFlowYear]
    exit: ExitValuation
    equity_cash_flow_vector: list[float]
    equity_irr_pct: Optional[float]
    npv_inr: float
    fd_benchmark: AlternativeBenchmarkResult
    nifty_benchmark: AlternativeBenchmarkResult
    year_1_gross_yield_pct: float
    year_1_net_yield_pct: float
    net_realizable_equity_trajectory: list[float] = field(default_factory=list)
    real_equity_irr_pct: Optional[float] = None
    year_1_dscr: Optional[float] = None
    avg_dscr: Optional[float] = None
    scenario_appreciation_rate: float = 0.0
    scenario_loan_rate: float = 0.0
    scenario_vacancy_months: float = 0.0
    scenario_exit_discount_pct: float = 0.0
    scenario_uc_delay_months: int = 0


@dataclass
class GateResult:
    """Evaluation result for one of the 4 hierarchical decision gates."""
    gate_number: int
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class MasterVerdict:
    """Aggregated decision outcome across all 4 gates."""
    status: str  # "GO" | "CAUTION" | "NO-GO"
    color: str   # "green" | "amber" | "red"
    summary: str
    gates: list[GateResult] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Dedicated stress test outcome card."""
    name: str
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "PASS"  # "PASS" | "WARN" | "FAIL"
    notes: list[str] = field(default_factory=list)


@dataclass
class ScenarioMatrixResult:
    """Best, Base, and Worst (Stress) underwriting outcomes."""
    best: UnderwritingResult
    base: UnderwritingResult
    worst: UnderwritingResult
    stress_interest_rate_shock: StressTestResult
    stress_vacancy_shock: StressTestResult
    stress_forced_exit_year_3: StressTestResult


@dataclass
class ExpectedValueResult:
    """Probability-weighted consolidation of Best / Base / Worst scenarios."""
    p_best: float
    p_base: float
    p_worst: float
    expected_terminal_wealth: float
    expected_equity_irr_pct: Optional[float]
    expected_net_exit_proceeds: float
    expected_total_cash_invested: float
    best_irr_pct: Optional[float]
    base_irr_pct: Optional[float]
    worst_irr_pct: Optional[float]
    best_terminal_wealth: float
    base_terminal_wealth: float
    worst_terminal_wealth: float
    p_beat_fd: float
    p_beat_nifty: float
    fd_post_tax_pct: float
    nifty_pct: float


@dataclass
class LocalityRecommendation:
    """Ranked Pune micro-market recommendation aligned with investor profile and profitability."""
    rank: int
    pincode: str
    locality_name: str
    pune_zone: str
    jurisdiction: str
    alignment_score: float
    verdict_status: str
    gate_statuses: list[str]
    ticket_price_inr: float
    all_in_cost_inr: float
    upfront_cash_inr: float
    monthly_emi_inr: float
    foir_pct: float
    base_irr_pct: Optional[float]
    real_irr_pct: Optional[float]
    net_wealth_created_inr: float
    npv_inr: float
    net_yield_pct: float
    dscr: Optional[float]
    market_to_rr_ratio: float
    liquidity_tier: str
    quarters_to_sell: float
    appreciation_rate_pct: float
    year_1_monthly_cf_inr: float
    alignment_reasons: list[str] = field(default_factory=list)
