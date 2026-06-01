# Data

The private pipeline uses WRDS BoardEx North America, CRSP, CCM, Compustat, optional I/B/E/S, optional liquidity measures, SEC EDGAR filing timestamps, and FRED macro regime controls.

Public artifacts are aggregate only.

## Sources Used in the Scaled Run

- BoardEx North America role histories and company profiles.
- CRSP monthly returns, daily returns, names, delisting records, and CCM links.
- Compustat annual fundamentals with reporting-lag activation.
- I/B/E/S analyst attention controls.
- Fama-French five factors plus UMD.
- FRED VIX, NBER recession, fed funds, term spread, and credit spread controls.
- SEC EDGAR submissions for the timing-audit overlay.

## Access and Publication Boundaries

Raw WRDS extracts, cached private panels, person-level BoardEx records, and row-level model predictions stay under ignored `data/` paths on the research machine. Public artifacts in `reports/` are aggregate tables, figures, manifests, and dashboard files only.
