"""Build the modeling table: one row per (match, side) with strictly
pre-match features. Window: matches from 1998-01-01 (training/eval) plus
the 72 scheduled 2026 WC group fixtures (prediction rows, y = NaN).

Features (all team-minus-opponent diffs unless noted):
  elo_diff            pre-match Elo difference (incl. nothing for home adv;
                      home/host handled by separate indicators)
  fifa_rank_diff      log(opp_rank) - log(team_rank)  (positive = team better)
  form_pts_diff       points/game over last 10 internationals
  form_gf_diff        goals for/game last 10
  form_ga_diff        goals against/game last 10 (sign flipped: positive good)
  rest_diff           rest days (capped 30) minus opponent's
  h2h_gd              head-to-head goal diff/game vs this opponent, last 15y
  wc_exp_diff         prior men's WC finals matches played (log1p diff)
  host                1 if team plays in own country
  opp_host            1 if opponent plays in own country
  neutral             1 if neutral venue
  comp_*              competition type one-hots: wc, cont_final, wc_qual,
                      cont_qual, nations, friendly (other = baseline)
Target: goals scored by the team (90-minute goals where available; the
results dataset records final score incl. ET for knockouts — acceptable
noise, flagged with `knockout` indicator).
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"

FIFA_ALIASES = {
    "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran",
    "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "China PR": "China", "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde", "Congo DR": "DR Congo",
    "Kyrgyz Republic": "Kyrgyzstan", "USA": "United States",
    "Türkiye": "Turkey", "Curacao": "Curaçao", "Czechia": "Czech Republic",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia", "The Gambia": "Gambia",
    "Hong Kong, China": "Hong Kong", "Macau, China": "Macau",
    "Brunei Darussalam": "Brunei",
}
# results.csv uses "Czech Republic"; 2026 fixtures may use either — normalize.
RESULTS_ALIASES = {"Czechia": "Czech Republic"}


def comp_type(t: str) -> str:
    s = t.lower()
    if "fifa world cup" in s and "qualification" not in s:
        return "wc"
    if "fifa world cup" in s:
        return "wc_qual"
    continental = ("uefa euro", "copa américa", "african cup of nations",
                   "afc asian cup", "gold cup", "concacaf championship",
                   "oceania nations cup", "confederations cup")
    if any(c in s for c in continental):
        return "cont_qual" if "qualification" in s else "cont_final"
    if "nations league" in s:
        return "nations"
    if s == "friendly":
        return "friendly"
    return "other"


def main():
    me = pd.read_csv(PROC / "matches_elo.csv", parse_dates=["date"])
    for c in ("home_team", "away_team"):
        me[c] = me[c].replace(RESULTS_ALIASES)

    # FIFA ranks: monthly releases -> per-team time series for as-of joins
    fr = pd.read_csv(RAW / "fifa_ranking/ranking_fifa_historical.csv",
                     parse_dates=["date"])
    fr["team"] = fr["team"].replace(FIFA_ALIASES).replace(RESULTS_ALIASES)
    fr = (fr.sort_values("date")
            .drop_duplicates(["team", "date"], keep="last"))
    # rank within each release date (file stores points; derive rank)
    fr["rank"] = fr.groupby("date")["total_points"].rank(
        ascending=False, method="min")
    fr = fr[["team", "date", "rank"]].sort_values("date")

    # prior WC finals matches per team, cumulative by date (jfjelstul)
    wc = pd.read_csv(RAW / "worldcup/matches.csv",
                     parse_dates=["match_date"])
    wc = wc[wc.tournament_name.str.contains("Men's")]
    app = pd.concat([
        wc[["match_date", "home_team_name"]].rename(
            columns={"home_team_name": "team"}),
        wc[["match_date", "away_team_name"]].rename(
            columns={"away_team_name": "team"})])
    app["team"] = app["team"].replace(
        {"West Germany": "Germany", **RESULTS_ALIASES})
    app = app.sort_values("match_date")
    app["n"] = app.groupby("team").cumcount()  # matches BEFORE this one

    # ------------------------------------------------- long format, per side
    played = me.dropna(subset=["home_score"]).copy()
    sched = pd.read_csv(RAW / "international_results/results.csv",
                        parse_dates=["date"])
    for c in ("home_team", "away_team"):
        sched[c] = sched[c].replace(RESULTS_ALIASES)
    fixtures = sched[(sched.home_score.isna()) &
                     (sched.tournament == "FIFA World Cup")].copy()
    fixtures["home_elo_pre"] = np.nan  # filled from current ratings below
    fixtures["away_elo_pre"] = np.nan
    cur = pd.read_csv(PROC / "elo_current.csv", index_col=0)["elo"]
    cur.index = pd.Index(cur.index).map(
        lambda t: RESULTS_ALIASES.get(t, t))
    fixtures["home_elo_pre"] = fixtures.home_team.map(cur)
    fixtures["away_elo_pre"] = fixtures.away_team.map(cur)

    allm = pd.concat([played, fixtures], ignore_index=True)
    allm = allm.sort_values("date").reset_index(drop=True)
    allm["match_id"] = np.arange(len(allm))
    allm["comp"] = allm.tournament.map(comp_type)
    allm["knockout"] = 0  # refined below for WC eval matches

    long = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "match_id": allm.match_id, "date": allm.date,
            "team": allm[f"{side}_team"], "opponent": allm[f"{opp}_team"],
            "goals": allm[f"{side}_score"],
            "goals_against": allm[f"{opp}_score"],
            "elo": allm[f"{side}_elo_pre"], "opp_elo": allm[f"{opp}_elo_pre"],
            "venue_country": allm.country, "neutral": allm.neutral.astype(int),
            "comp": allm.comp, "tournament": allm.tournament,
            "is_home_listed": int(side == "home"),
        })
        long.append(d)
    L = pd.concat(long, ignore_index=True).sort_values(
        ["date", "match_id"]).reset_index(drop=True)

    # rolling form & rest, computed per team over its own match history
    L["pts"] = np.select(
        [L.goals > L.goals_against, L.goals == L.goals_against],
        [3.0, 1.0], default=0.0)
    L.loc[L.goals.isna(), "pts"] = np.nan

    L = L.sort_values(["team", "date"])
    g = L.groupby("team")
    L["form_pts"] = g["pts"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    L["form_gf"] = g["goals"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    L["form_ga"] = g["goals_against"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    L["rest"] = g["date"].diff().dt.days.clip(upper=30)

    # head-to-head goal diff per game, last 15 years, before this match
    L["_gd"] = L.goals - L.goals_against
    L = L.sort_values(["team", "opponent", "date"])
    h2h_vals = np.zeros(len(L))
    pos = 0
    for _, grp in L.groupby(["team", "opponent"], sort=False):
        dates = grp.date.to_numpy()
        gds = grp._gd.to_numpy(dtype=float)
        for i in range(len(grp)):
            lo = dates[i] - np.timedelta64(15 * 365, "D")
            mask = (dates[:i] >= lo)
            past = gds[:i][mask]
            past = past[~np.isnan(past)]
            h2h_vals[pos + i] = past.mean() if len(past) else 0.0
        pos += len(grp)
    L["h2h_gd"] = h2h_vals
    L = L.drop(columns="_gd")

    # FIFA rank as-of (most recent release before match date)
    L = L.sort_values("date")
    L = pd.merge_asof(L, fr.rename(columns={"team": "team", "rank": "fifa_rank"}),
                      left_on="date", right_on="date", by="team",
                      allow_exact_matches=False)

    # prior WC matches as-of
    app_s = app.rename(columns={"match_date": "date"}).sort_values("date")
    L = pd.merge_asof(L, app_s.rename(columns={"n": "wc_matches"}),
                      on="date", by="team", allow_exact_matches=True)
    L["wc_matches"] = L.groupby("team")["wc_matches"].ffill().fillna(0)

    # host indicator
    L["host"] = (L.team == L.venue_country).astype(int)

    L.to_csv(PROC / "long_features_raw.csv", index=False)
    print(L.shape)
    print("FIFA rank coverage 1998+ rows:",
          L[L.date >= "1998-01-01"].fifa_rank.notna().mean().round(3))


if __name__ == "__main__":
    main()
