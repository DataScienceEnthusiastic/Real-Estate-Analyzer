from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.models import PropertyInput


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _read_csv_robust(file_name: str, dtype: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Reads a CSV from the data directory with consistent NA handling and explicit dtypes."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required dataset file not found at: {file_path}")
    return pd.read_csv(
        file_path,
        dtype=dtype,
        na_values=["NULL", "null", "NaN", "nan", ""]
    )


@lru_cache(maxsize=1)
def load_macro_constants() -> dict[str, float]:
    """Loads macro parameters and maps parameter_name -> float value."""
    df = _read_csv_robust("pune_macro_constants_v2.csv")
    expected_cols = {"parameter_name", "value"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Macro constants CSV missing expected columns: {expected_cols - set(df.columns)}")
    
    macro: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["parameter_name"]).strip()
        val = float(row["value"])
        macro[name] = val
    return macro


@lru_cache(maxsize=1)
def load_zone_risk() -> pd.DataFrame:
    """Loads Pune zone risk metrics and assumptions."""
    df = _read_csv_robust("pune_zone_risk_v2.csv")
    required_cols = {
        "pune_zone", "quarters_to_sell", "liquidity_tier", "exit_discount_stress_pct",
        "avg_uc_delay_months", "vacancy_months_best", "vacancy_months_avg", "vacancy_months_worst"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Zone risk CSV missing required columns: {missing}")
    return df


@lru_cache(maxsize=1)
def load_locality_master() -> pd.DataFrame:
    """Loads the 43 Pune localities with primary ready reckoner rates and appreciation."""
    df = _read_csv_robust("pune_locality_master_v3.csv", dtype={"pincode": str})
    required_cols = {
        "pincode", "locality_name", "pune_zone", "jurisdiction",
        "rr_rate_primary_sqft", "appreciation_pct_median"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Locality master CSV missing required columns: {missing}")
    # Normalize pincode to 6-digit string
    df["pincode"] = df["pincode"].astype(str).str.strip().str.zfill(6)
    return df


@lru_cache(maxsize=1)
def load_unit_market_data() -> pd.DataFrame:
    """Loads the 172 unit configurations across Pune localities."""
    df = _read_csv_robust("pune_unit_market_data_v3.csv", dtype={"pincode": str})
    required_cols = {
        "pincode", "locality_name", "asset_type", "bhk",
        "median_carpet_area_sqft", "expected_sba_sqft", "est_ticket_price_inr",
        "median_monthly_rent", "implied_gross_rental_yield_pct"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Unit market data CSV missing required columns: {missing}")
    df["pincode"] = df["pincode"].astype(str).str.strip().str.zfill(6)
    return df


def get_merged_dataset() -> pd.DataFrame:
    """Joins unit market data with locality master and zone risk into a flat analytic table."""
    unit_df = load_unit_market_data()
    loc_df = load_locality_master()
    zone_df = load_zone_risk()

    # Join unit -> locality on (pincode, locality_name)
    merged = unit_df.merge(
        loc_df[["pincode", "locality_name", "pune_zone", "jurisdiction", "rr_rate_primary_sqft", "appreciation_pct_median", "appreciation_pct_avg"]],
        on=["pincode", "locality_name"],
        how="left"
    )

    # Join with zone risk on pune_zone
    merged = merged.merge(
        zone_df,
        on="pune_zone",
        how="left"
    )
    return merged


def get_all_zones() -> list[str]:
    """Returns sorted unique zones from zone risk dataset."""
    zone_df = load_zone_risk()
    return sorted(zone_df["pune_zone"].unique().tolist())


def get_localities(zone_filter: Optional[str] = None) -> list[dict[str, str]]:
    """Returns unique localities, optionally filtered by zone."""
    loc_df = load_locality_master()
    if zone_filter and zone_filter != "All Zones":
        loc_df = loc_df[loc_df["pune_zone"] == zone_filter]
    
    unique_locs = loc_df[["pincode", "locality_name", "pune_zone", "jurisdiction"]].drop_duplicates()
    unique_locs = unique_locs.sort_values(by=["locality_name"])
    return unique_locs.to_dict("records")


def get_property_market_defaults(pincode: str, locality_name: str, bhk: str) -> PropertyInput:
    """
    Looks up official benchmark data for a selected property and returns pre-populated PropertyInput.
    Handles the Chakan 4_PLUS_BHK edge case where rent is NaN without throwing exceptions.
    """
    merged = get_merged_dataset()
    pincode_clean = str(pincode).strip().zfill(6)
    
    match = merged[
        (merged["pincode"] == pincode_clean) &
        (merged["locality_name"] == locality_name) &
        (merged["bhk"] == bhk)
    ]

    if match.empty:
        raise ValueError(f"No unit data found for pincode '{pincode}', locality '{locality_name}', BHK '{bhk}'")

    row = match.iloc[0]

    raw_rent = row.get("median_monthly_rent")
    # Graceful handling for missing rent (Chakan 4_PLUS_BHK)
    monthly_rent = 0.0 if pd.isna(raw_rent) else float(raw_rent)

    raw_appreciation = row.get("appreciation_pct_median")
    appreciation_rate = float(raw_appreciation) if not pd.isna(raw_appreciation) else 0.06

    raw_agreement = row.get("est_ticket_price_inr")
    agreement_val = float(raw_agreement) if not pd.isna(raw_agreement) else 0.0

    raw_carpet = row.get("median_carpet_area_sqft")
    carpet_area = float(raw_carpet) if not pd.isna(raw_carpet) else 0.0

    raw_sba = row.get("expected_sba_sqft")
    sba = float(raw_sba) if not pd.isna(raw_sba) else carpet_area * 1.3

    raw_rr = row.get("rr_rate_primary_sqft")
    rr_rate = float(raw_rr) if not pd.isna(raw_rr) else 0.0

    return PropertyInput(
        pincode=pincode_clean,
        locality_name=str(locality_name),
        pune_zone=str(row.get("pune_zone", "Central")),
        jurisdiction=str(row.get("jurisdiction", "PMC")),
        bhk=str(bhk),
        use_case="investment_rental",
        property_status="ready_resale",
        buyer_ownership="sole_male",
        agreement_value_inr=agreement_val,
        carpet_area_sqft=carpet_area,
        expected_sba_sqft=sba,
        monthly_rent_inr=monthly_rent,
        appreciation_rate=appreciation_rate,
        rr_rate_primary_sqft=rr_rate,
        liquidity_tier=str(row.get("liquidity_tier", "medium")),
        quarters_to_sell=float(row.get("quarters_to_sell", 4.0)),
        exit_discount_stress_pct=float(row.get("exit_discount_stress_pct", 0.08)),
        avg_uc_delay_months=float(row.get("avg_uc_delay_months", 8.0)),
        vacancy_months_best=float(row.get("vacancy_months_best", 0.5)),
        vacancy_months_avg=float(row.get("vacancy_months_avg", 1.0)),
        vacancy_months_worst=float(row.get("vacancy_months_worst", 2.5)),
        expected_possession_months=24,
        brokerage_enabled=True,
        sell_brokerage_enabled=True
    )
