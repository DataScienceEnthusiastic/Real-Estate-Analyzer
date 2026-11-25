# Real Estate Analyzer (Pune Residential Underwriter)

Institutional-grade quantitative underwriting dashboard for Pune residential real estate investments and self-use decisions.

## Key Capabilities
- **Statutory Acquisition & Tax Engine:** Maharashtra urban/rural stamp duty slabs, Ready Reckoner (RR) valuation floors, female ownership rebates, GST (1%/5% UC), Section 194-IA TDS compliance, MOTD/loan origination fees, Section 24(a)/(b) (including 5-year pre-construction interest amortization), Section 71B 8-year loss carry-forward, surcharge tiers, and post-July 2024 LTCG rules (12.5% + surcharge/cess).
- **4-Gate Decision Engine:** Evaluates Affordability & Liquidity Safety (FOIR, emergency buffer), Downside Survival (+200 bps rate shock, 6-month vacancy, Year-3 forced exit), Yield & Valuation Sanity (DSCR, Market-to-RR ratio), and Wealth Creation vs. post-tax SBI FD & cash-flow-parity Nifty 50 TRI.
- **Smart Micro-Market Finder (`All Zones`):** Automatically underwrites all 43 Pune pincodes against your financial profile and returns the Top 3 ranked areas with alignment scores and rationale.
- **Institutional Exports & Glass-Box Proofs:** Step-by-step live formula proofs across every tab, plus multi-page PDF memorandum export (with embedded charts and All-Zones Top 3 report) and full CSV audit export.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests
```bash
pytest tests/ -q
```
