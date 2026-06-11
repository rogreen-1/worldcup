"""Compute eloratings.net-style Elo over martj42 internationals.

Outputs matches_elo.csv: every match with pre-match Elo for both sides.
Algorithm: R' = R + K*G*(W - We), We = 1/(1+10^(-d/400)), d includes +100
home advantage when not neutral. K by competition tier, G by goal margin.
Team identities unified via former_names.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"

K_TIERS = [
    (("FIFA World Cup",), 60),
    (("Copa América", "African Cup of Nations", "UEFA Euro", "AFC Asian Cup",
      "CONCACAF Championship", "Gold Cup", "Confederations Cup",
      "Oceania Nations Cup", "Intercontinental"), 50),
    (("qualification",), 40),
    (("Friendly",), 20),
]
DEFAULT_K = 30  # other tournaments (Nations League, King's Cup, etc.)


def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    if "qualification" in t:
        return 40
    for names, k in K_TIERS[1:2]:
        for n in names:
            if n.lower() in t:
                return 50
    if t == "friendly":
        return 20
    return DEFAULT_K


def margin_mult(diff: int) -> float:
    d = abs(diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return 1.75 + (d - 3) / 8.0


def main():
    r = pd.read_csv(RAW / "international_results/results.csv")
    r["date"] = pd.to_datetime(r["date"])

    fn = pd.read_csv(RAW / "international_results/former_names.csv")
    rename = dict(zip(fn.former, fn.current))
    for col in ("home_team", "away_team"):
        r[col] = r[col].replace(rename)

    played = r.dropna(subset=["home_score", "away_score"]).sort_values(
        ["date"]).reset_index(drop=True)

    elo: dict[str, float] = {}
    rows = []
    for m in played.itertuples():
        h, a = m.home_team, m.away_team
        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)
        d = eh - ea + (0.0 if m.neutral else 100.0)
        we = 1.0 / (1.0 + 10 ** (-d / 400.0))
        gd = int(m.home_score) - int(m.away_score)
        w = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
        delta = k_factor(m.tournament) * margin_mult(gd) * (w - we)
        rows.append((m.date, h, a, int(m.home_score), int(m.away_score),
                     m.tournament, m.country, m.neutral, eh, ea))
        elo[h] = eh + delta
        elo[a] = ea - delta

    out = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "country", "neutral", "home_elo_pre", "away_elo_pre"])
    out.to_csv(PROC / "matches_elo.csv", index=False)

    final = pd.Series(elo).sort_values(ascending=False)
    final.to_csv(PROC / "elo_current.csv", header=["elo"])
    print(f"{len(out)} matches rated, {len(elo)} teams")
    print("\nTop 15 current:")
    print(final.head(15).round(0).to_string())


if __name__ == "__main__":
    main()
