# wc26 — 2026 World Cup match prediction

Predicts 2026 FIFA World Cup matches with an L1-penalized (lasso) Poisson
goals model trained on all international matches since 1996, evaluated
leave-one-tournament-out on the seven World Cups 1998–2022, and feeds a
Monte Carlo tournament simulator implementing the full 48-team format.

## Pipeline

```
make pipeline          # or run the steps individually:
python src/build_elo.py        # 1. Elo over 49k internationals (1872->now)
python src/build_features.py   # 2. per-(match, side) feature table
python src/fit_model.py        # 3. lasso Poisson, LOTO CV, final coefficients
python src/simulate_baseline.py 100000   # 4. Elo-based tournament Monte Carlo
```

Stage 4 is currently the *baseline* simulator (hand-calibrated Elo→Poisson
mapping). Wiring the fitted model from stage 3 into the simulator is the
next step (see below).

## Data (`data/raw/`, vendored with source attribution)

| dir | source | role |
|---|---|---|
| `international_results/` | github.com/martj42/international_results (CC0) | all internationals 1872–present incl. scheduled 2026 WC group fixtures; basis for Elo, form, h2h |
| `worldcup/` | github.com/jfjelstul/worldcup (CC-BY 4.0) | World Cup database 1930–2022; evaluation set definition + WC pedigree features |
| `fifa_ranking/` | github.com/Dato-Futbol/fifa-ranking mirror | monthly FIFA rankings 1992–2024-09 |
| `elo_manual_2026.csv` | eloratings.net top-20 (2026-01-19) + estimates | baseline simulator input only |

Refresh with `scripts/fetch_data.sh`. Note the FIFA ranking mirror ends
Sep 2024; the fitted model assigns FIFA rank a zero coefficient, so this
staleness does not affect predictions.

## Model

Two rows per match (one per side); target = goals scored; features are
strictly pre-match team-minus-opponent differences plus venue/competition
indicators. L1-penalized Poisson GLM (statsmodels), features standardized,
intercept unpenalized. Penalty chosen by leave-one-tournament-out CV where
each fold trains only on matches *before* that tournament (temporal
integrity; no information from the future).

**Held-out W/D/L log loss (mean over 7 WCs):** lasso 0.9686 vs Elo-only
baseline 0.9779 (lasso wins 5/7 tournaments).

**Selected coefficients** (standardized): `elo_diff +0.40`,
`opp_host −0.13`, `form_ga_diff +0.11`, `form_pts_diff −0.07` (suppressor —
collinearity artifact, do not read causally), `host +0.07`, `h2h_gd +0.05`.
Zeroed: FIFA rank diff, WC experience, WC-match indicator, neutral flag.

## Tournament simulator (`src/simulate_baseline.py`)

Full 2026 format: 12 groups of 4, FIFA tiebreakers with head-to-head
mini-tables, ranking of third-placed teams, third-place slot allocation by
bipartite matching against the Annex C allowed-group sets, fixed R32→final
bracket, extra time at 1/3 intensity, Elo-weighted shootouts, +100 Elo to
the three hosts.

## Layout

```
data/raw/        vendored third-party data (see table)
data/processed/  matches_elo.csv, elo_current.csv, long_features_raw.csv
src/             pipeline stages
outputs/         cv_results.csv, final_model.csv, baseline_sim_results.csv
scripts/         fetch_data.sh (refresh raw data from upstream)
```

## Next steps

1. `src/predict.py`: team feature snapshot (current Elo, form, h2h matrix)
   + saved coefficients → `predict_lambda(team, opponent, venue)` for
   arbitrary pairings.
2. Swap `lambdas()` in the simulator for the model predictions; group
   fixtures use their exact feature rows, knockouts use the snapshot.
3. Optional: Dixon-Coles low-score dependence; per-sim strength noise
   (`elo + N(0, σ)`) to reflect rating uncertainty; market-implied
   strengths from The Odds API as an alternative input.

## Known limitations

- Goals in `results.csv` include extra time for knockout matches (mild
  target noise); independent Poisson slightly underestimates draws.
- 30 of 48 manual baseline Elo values in `elo_manual_2026.csv` are
  estimates (flagged in the file); the computed Elo in
  `data/processed/elo_current.csv` supersedes them once the model is wired
  into the simulator.
- Squad market value not included (no clean open mirror); historically it
  adds little beyond Elo.
