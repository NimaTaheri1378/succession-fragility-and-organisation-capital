# Methodology

The core characteristic is Organisation-Capital Fragility, a monthly firm-level composite built from executive-team concentration, succession depth, external load, poaching pressure, bench depth, and team-cohesion decay.

## Timing

Executive states are expanded from BoardEx role start and end dates into month-end firm-team states. Features are formed before the next-month return label, and Compustat fundamentals are activated with a six-month reporting lag. CRSP monthly returns use the actual last trading date of the month; factor and macro controls are aligned by calendar month to avoid missing factors in months where the last trading date is not the calendar month-end.

## Baseline Tests

The transparent benchmark uses Fama-MacBeth cross-sectional regressions and quintile long-short portfolios. Robustness tables add sample exclusions, alternative OCF maps, regime splits, double sorts, turnover, cost haircuts, and FF5+UMD alpha estimates.

## Event Study

Leadership-change events are built from high-criticality BoardEx role appointments and departures. Events are linked to CRSP daily returns and sorted by pre-event OCF buckets. SEC EDGAR submissions are used as a timing audit overlay for CIK-linked events; only aggregate match coverage is published.

## Model Ladder

The model ladder follows the proposal order: Elastic Net, LightGBM, GPU Deep Sets, and a conditional-autoencoder/SDF block. The final Deep Sets run uses variable-sized BoardEx executive sets: each firm-month is represented by up to 32 executives with role criticality, tenure, internal-candidate status, outside-role burden, and prior-employer breadth measured only from roles that started before the formation month. The leakage-safe run fits a separate expanding target-year model: each test year is predicted only by a model trained on prior calendar years. Row-level predictions are not published; only aggregate metrics, portfolio returns, and figure outputs are retained.

## Strategy Sharpening

The strategy-sharpening stage refits the walk-forward LightGBM model and evaluates a small pre-specified portfolio grid: quintile and decile score sorts, equal and value weighting, nonmicrocap filters, analyst-coverage splits, intangibility splits, high-OCF states, and a score-dispersion gate whose threshold is computed only from prior months. The stage writes aggregate return, turnover, alpha, bootstrap interval, and visual-QA outputs; it does not publish row-level predictions.
