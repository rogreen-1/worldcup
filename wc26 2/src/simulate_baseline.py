"""Monte Carlo simulator for the 2026 FIFA World Cup.

Model
-----
- Team strength: World Football Elo (eloratings.net snapshot, see elo.csv).
- Match model: Elo expected score -> expected goal difference -> independent
  Poisson goals for each side. Total expected goals MU_TOTAL, expected GD
  is GD_SCALE * (E - 0.5) where E = 1 / (1 + 10^(-d/400)).
- Home advantage: +HOME_BONUS Elo to USA/Mexico/Canada in every match
  (entire tournament is on home soil).
- Knockouts: 90' with the same model; if level, extra time at 1/3 intensity;
  if still level, penalties as an Elo-weighted coin flip (mild edge).
- Group tiebreakers: points, GD, GF, head-to-head mini-table, then random
  (proxy for fair play / drawing of lots).
- Third-place allocation: FIFA fixes which groups' thirds each winner may
  face (Regulations Annex C). We solve the per-simulation assignment as a
  bipartite matching against those allowed sets via backtracking. Where the
  matching is non-unique, FIFA's published table picks one specific solution;
  we pick the first found in a fixed slot order. This can differ from Annex C
  in which *allowed* third a winner draws, never in *whether* the pairing is
  legal — negligible effect on advancement probabilities.

Usage: python sim.py [n_sims]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- parameters
MU_TOTAL = 2.65        # expected total goals per match
GD_SCALE = 4.5         # expected goal diff per unit of (E - 0.5); calibrated
                       # so sim W + D/2 tracks Elo expected score for d<=200
MIN_LAMBDA = 0.15      # floor on per-team expected goals
HOME_BONUS = 100.0     # Elo bonus for host nations
HOSTS = {"United States", "Mexico", "Canada"}
ET_FACTOR = 1 / 3      # extra-time scoring intensity vs 90'
PENALTY_ELO_SCALE = 1000.0  # softens Elo edge in shootouts (d=200 -> ~55%)

# ---------------------------------------------------------------- tournament
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Round of 32, FIFA matches 73-88. ("W",g)=winner, ("R",g)=runner-up,
# ("T",slot)=third-place team assigned to that winner's slot.
R32 = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("T", "E")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("T", "I")),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("T", "A")),
    80: (("W", "L"), ("T", "L")),
    81: (("W", "D"), ("T", "D")),
    82: (("W", "G"), ("T", "G")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("T", "B")),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("T", "K")),
    88: (("R", "D"), ("R", "G")),
}

# Allowed source groups for the third-place opponent of each group winner
# (FIFA Regulations, round-of-32 schedule).
THIRD_SLOTS = {
    "E": set("ABCDF"),
    "I": set("CDFGH"),
    "A": set("CEFHI"),
    "L": set("EHIJK"),
    "D": set("BEFIJ"),
    "G": set("AEHIJ"),
    "B": set("EFGIJ"),
    "K": set("DEIJL"),
}
SLOT_ORDER = ["E", "I", "A", "L", "D", "G", "B", "K"]

R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = (101, 102)

# ---------------------------------------------------------------- match model

def load_elo(path: Path) -> dict[str, float]:
    with open(path) as f:
        return {r["team"]: float(r["elo"]) for r in csv.DictReader(f)}


def lambdas(elo_a: float, elo_b: float) -> tuple[float, float]:
    e = 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))
    gd = GD_SCALE * (e - 0.5)
    return (max(MIN_LAMBDA, (MU_TOTAL + gd) / 2),
            max(MIN_LAMBDA, (MU_TOTAL - gd) / 2))


def adj_elo(team: str, elo: dict[str, float]) -> float:
    return elo[team] + (HOME_BONUS if team in HOSTS else 0.0)


def play_group_match(a, b, elo, rng):
    la, lb = lambdas(adj_elo(a, elo), adj_elo(b, elo))
    return rng.poisson(la), rng.poisson(lb)


def play_knockout(a, b, elo, rng) -> str:
    ea, eb = adj_elo(a, elo), adj_elo(b, elo)
    la, lb = lambdas(ea, eb)
    ga, gb = rng.poisson(la), rng.poisson(lb)
    if ga != gb:
        return a if ga > gb else b
    ga, gb = rng.poisson(la * ET_FACTOR), rng.poisson(lb * ET_FACTOR)
    if ga != gb:
        return a if ga > gb else b
    p_a = 1.0 / (1.0 + 10 ** (-(ea - eb) / PENALTY_ELO_SCALE))
    return a if rng.random() < p_a else b

# ---------------------------------------------------------------- group stage

def rank_group(teams, results, rng):
    """results: dict[(a,b)] = (ga,gb). FIFA ordering with random last resort."""
    pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
    for (a, b), (ga, gb) in results.items():
        gf[a] += ga; gf[b] += gb
        gd[a] += ga - gb; gd[b] += gb - ga
        if ga > gb: pts[a] += 3
        elif gb > ga: pts[b] += 3
        else: pts[a] += 1; pts[b] += 1

    def sort_block(block):
        if len(block) == 1:
            return list(block)
        # head-to-head mini-table among the tied teams
        h2h_p = defaultdict(int); h2h_d = defaultdict(int); h2h_f = defaultdict(int)
        bs = set(block)
        for (a, b), (ga, gb) in results.items():
            if a in bs and b in bs:
                h2h_f[a] += ga; h2h_f[b] += gb
                h2h_d[a] += ga - gb; h2h_d[b] += gb - ga
                if ga > gb: h2h_p[a] += 3
                elif gb > ga: h2h_p[b] += 3
                else: h2h_p[a] += 1; h2h_p[b] += 1
        keyed = sorted(block, key=lambda t: (h2h_p[t], h2h_d[t], h2h_f[t],
                                             rng.random()), reverse=True)
        return keyed

    primary = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    out, i = [], 0
    while i < len(primary):
        j = i
        key = (pts[primary[i]], gd[primary[i]], gf[primary[i]])
        while j < len(primary) and (pts[primary[j]], gd[primary[j]],
                                    gf[primary[j]]) == key:
            j += 1
        out.extend(sort_block(primary[i:j]))
        i = j
    stats = {t: (pts[t], gd[t], gf[t]) for t in teams}
    return out, stats

# ------------------------------------------------- third-place slot matching

def assign_thirds(qualified_groups: list[str]) -> dict[str, str] | None:
    """Match 8 qualified third-place source groups to the 8 winner slots."""
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def bt(i: int) -> bool:
        if i == len(SLOT_ORDER):
            return True
        slot = SLOT_ORDER[i]
        for g in qualified_groups:
            if g not in used and g in THIRD_SLOTS[slot]:
                assignment[slot] = g
                used.add(g)
                if bt(i + 1):
                    return True
                used.discard(g)
                del assignment[slot]
        return False

    return assignment if bt(0) else None

# ---------------------------------------------------------------- tournament

def simulate_once(elo, rng, counters):
    winners, runners, thirds = {}, {}, {}
    third_stats = {}
    for g, teams in GROUPS.items():
        res = {}
        for i in range(4):
            for j in range(i + 1, 4):
                res[(teams[i], teams[j])] = play_group_match(
                    teams[i], teams[j], elo, rng)
        order, stats = rank_group(teams, res, rng)
        winners[g], runners[g], thirds[g] = order[0], order[1], order[2]
        third_stats[g] = stats[order[2]]
        counters["group_win"][order[0]] += 1

    # best 8 thirds: points, GD, GF, random (fair-play proxy)
    ranked = sorted(GROUPS, key=lambda g: (*third_stats[g], rng.random()),
                    reverse=True)
    qual = ranked[:8]
    slot_map = assign_thirds(qual)
    if slot_map is None:  # cannot happen with FIFA's sets, but be safe
        slot_map = dict(zip(SLOT_ORDER, qual))

    def resolve(ref):
        kind, key = ref
        if kind == "W": return winners[key]
        if kind == "R": return runners[key]
        return thirds[slot_map[key]]

    alive = {}
    for m, (ra, rb) in R32.items():
        a, b = resolve(ra), resolve(rb)
        counters["r32"][a] += 1; counters["r32"][b] += 1
        alive[m] = play_knockout(a, b, elo, rng)

    for stage, table, name in ((R16, "r16", "r16"), (QF, "qf", "qf"),
                               (SF, "sf", "sf")):
        nxt = {}
        for m, (ma, mb) in stage.items():
            a, b = alive[ma], alive[mb]
            counters[name][a] += 1; counters[name][b] += 1
            nxt[m] = play_knockout(a, b, elo, rng)
        alive = nxt

    a, b = alive[FINAL[0]], alive[FINAL[1]]
    counters["final"][a] += 1; counters["final"][b] += 1
    champ = play_knockout(a, b, elo, rng)
    counters["champion"][champ] += 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    elo = load_elo(Path(__file__).resolve().parents[1] / "data" / "raw" / "elo_manual_2026.csv")
    all_teams = [t for g in GROUPS.values() for t in g]
    missing = [t for t in all_teams if t not in elo]
    assert not missing, f"missing Elo: {missing}"

    rng = np.random.default_rng(20260611)
    stages = ["group_win", "r32", "r16", "qf", "sf", "final", "champion"]
    counters = {s: defaultdict(int) for s in stages}

    for _ in range(n):
        simulate_once(elo, rng, counters)

    rows = []
    for t in all_teams:
        rows.append([t, elo[t]] + [100 * counters[s][t] / n for s in stages])
    rows.sort(key=lambda r: r[-1], reverse=True)

    hdr = ["team", "elo", "win_group", "R32", "R16", "QF", "SF",
           "final", "champion"]
    out = Path(__file__).resolve().parents[1] / "outputs" / "baseline_sim_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr + ["n_sims"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.0f}"] + [f"{x:.2f}" for x in r[2:]]
                       + [n])

    print(f"{n:,} tournament simulations\n")
    print(f"{'team':<24}{'elo':>6}{'grp win':>9}{'R32':>7}{'R16':>7}"
          f"{'QF':>7}{'SF':>7}{'final':>7}{'champ':>7}")
    for r in rows[:20]:
        print(f"{r[0]:<24}{r[1]:>6.0f}" + "".join(f"{x:>7.1f}"
              for x in r[2:3]) + "".join(f"{x:>7.1f}" for x in r[3:]))
    print(f"\nfull table -> {out}")


if __name__ == "__main__":
    main()
