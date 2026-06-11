"""Fit L1-penalized Poisson GLM on goals; evaluate leave-one-tournament-out
against an Elo-only baseline; report selected coefficients.

Training rows for each held-out World Cup are all internationals strictly
BEFORE that tournament's start date (temporal integrity, not k-fold).
Metric: match-level W/D/L log loss, computed from the two predicted goal
rates via independent Poisson; plus mean Poisson deviance on goals.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"

FEATURES = ["elo_diff", "fifa_rank_diff", "form_pts_diff", "form_gf_diff",
            "form_ga_diff", "rest_diff", "h2h_gd", "wc_exp_diff",
            "host", "opp_host", "neutral",
            "comp_wc", "comp_wc_qual", "comp_cont_final", "comp_cont_qual",
            "comp_nations", "comp_friendly"]

WC_STARTS = {1998: "1998-06-10", 2002: "2002-05-31", 2006: "2006-06-09",
             2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14",
             2022: "2022-11-20"}
WC_ENDS = {1998: "1998-07-12", 2002: "2002-06-30", 2006: "2006-07-09",
           2010: "2010-07-11", 2014: "2014-07-13", 2018: "2018-07-15",
           2022: "2022-12-18"}


def build_design(L: pd.DataFrame) -> pd.DataFrame:
    # self-join to pull opponent's own-row features
    opp = L[["match_id", "team", "fifa_rank", "form_pts", "form_gf",
             "form_ga", "rest", "wc_matches"]].rename(
        columns={"team": "opponent", "fifa_rank": "o_rank",
                 "form_pts": "o_form_pts", "form_gf": "o_form_gf",
                 "form_ga": "o_form_ga", "rest": "o_rest",
                 "wc_matches": "o_wc"})
    d = L.merge(opp, on=["match_id", "opponent"], how="left")

    d["elo_diff"] = (d.elo - d.opp_elo) / 100.0
    d["fifa_rank_diff"] = np.log(d.o_rank) - np.log(d.fifa_rank)
    d["form_pts_diff"] = d.form_pts - d.o_form_pts
    d["form_gf_diff"] = d.form_gf - d.o_form_gf
    d["form_ga_diff"] = d.o_form_ga - d.form_ga  # positive = team better
    d["rest_diff"] = (d.rest - d.o_rest) / 7.0
    d["wc_exp_diff"] = np.log1p(d.wc_matches) - np.log1p(d.o_wc)
    d["opp_host"] = d.groupby("match_id")["host"].transform("sum") - d.host
    for c in ("wc", "wc_qual", "cont_final", "cont_qual", "nations",
              "friendly"):
        d[f"comp_{c}"] = (d.comp == c).astype(int)
    return d


def wdl_logloss(lam_t, lam_o, gt, go, grid=13):
    lam_t = np.asarray(lam_t, dtype=float)
    lam_o = np.asarray(lam_o, dtype=float)
    gt = np.asarray(gt); go = np.asarray(go)
    """Match-level W/D/L log loss given the two team-rows of each match."""
    k = np.arange(grid)
    pt = np.exp(-lam_t[:, None]) * lam_t[:, None] ** k / np.array(
        [__import__("math").factorial(i) for i in k])
    po = np.exp(-lam_o[:, None]) * lam_o[:, None] ** k / np.array(
        [__import__("math").factorial(i) for i in k])
    joint = pt[:, :, None] * po[:, None, :]
    pw = np.triu(np.ones((grid, grid)), 1).T  # i>j
    p_win = (joint * (k[:, None] > k[None, :])).sum(axis=(1, 2))
    p_draw = (joint * (k[:, None] == k[None, :])).sum(axis=(1, 2))
    p_loss = 1 - p_win - p_draw
    out = np.where(gt > go, p_win, np.where(gt == go, p_draw, p_loss))
    return -np.log(np.clip(out, 1e-12, None)).mean()


def fit_pois(X, y, alpha, l1_wt=1.0):
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.GLM(y, Xc, family=sm.families.Poisson())
    if alpha == 0:
        return m.fit()
    pen = np.ones(Xc.shape[1]) * alpha
    pen[0] = 0.0  # never penalize intercept
    return m.fit_regularized(alpha=pen, L1_wt=l1_wt, maxiter=200)


def standardize(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return (train[cols] - mu) / sd, (test[cols] - mu) / sd, mu, sd


def main():
    L = pd.read_csv(PROC / "long_features_raw.csv", parse_dates=["date"])
    D = build_design(L)
    model_rows = D[(D.date >= "1996-01-01") & D.goals.notna()].copy()
    model_rows = model_rows.dropna(subset=FEATURES)

    is_wc = model_rows.comp == "wc"
    alphas = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    records = []
    for year, start in WC_STARTS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(WC_ENDS[year])
        train = model_rows[model_rows.date < start]
        test = model_rows[is_wc & model_rows.date.between(start, end)]
        test = test.sort_values(["match_id", "is_home_listed"],
                                ascending=[True, False])
        Xtr_s, Xte_s, _, _ = standardize(train, test, FEATURES)

        # Elo baseline: single feature, unpenalized
        base = fit_pois(Xtr_s[["elo_diff"]], train.goals.values, 0.0)
        lam_b = base.predict(sm.add_constant(Xte_s[["elo_diff"]],
                                             has_constant="add"))
        for a in alphas:
            fit = fit_pois(Xtr_s, train.goals.values, a)
            lam = fit.predict(sm.add_constant(Xte_s, has_constant="add"))
            t = test.is_home_listed == 1
            ll = wdl_logloss(lam[t.values], lam[~t.values],
                             test.goals[t.values].values,
                             test.goals[~t.values].values)
            records.append((year, a, ll, len(test) // 2))
        llb = wdl_logloss(lam_b[t.values], lam_b[~t.values],
                          test.goals[t.values].values,
                          test.goals[~t.values].values)
        records.append((year, "elo_only", llb, len(test) // 2))

    res = pd.DataFrame(records, columns=["wc", "alpha", "logloss", "n"])
    piv = res.pivot(index="alpha", columns="wc", values="logloss")
    piv["mean"] = piv.mean(axis=1)
    print("W/D/L log loss by held-out tournament (rows: lasso alpha):")
    print(piv.round(4).to_string())
    res.to_csv(OUTD / "cv_results.csv", index=False)

    # final fit at best alpha on all data, report coefficients
    best_a = piv.drop(index="elo_only").mean(axis=1).idxmin()
    train = model_rows
    mu, sd = train[FEATURES].mean(), train[FEATURES].std().replace(0, 1)
    Xs = (train[FEATURES] - mu) / sd
    fit = fit_pois(Xs, train.goals.values, best_a)
    coefs = pd.Series(np.asarray(fit.params)[1:], index=FEATURES)
    print(f"\nfinal fit alpha={best_a}, standardized coefficients:")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index)
          .round(4).to_string())
    pd.DataFrame({"coef": coefs, "mu": mu, "sd": sd}).to_csv(
        OUTD / "final_model.csv")
    np.save(OUTD / "final_intercept.npy", np.asarray(fit.params)[0])


if __name__ == "__main__":
    main()
