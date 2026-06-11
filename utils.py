import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson


def load_data(
    results_path: str, worldcup_path: str, name_map: dict[str, str] | None = None
) -> tuple[pl.DataFrame, dict]:
    name_map = name_map or {}
    res = pl.read_csv(results_path, null_values=["NA"]).with_columns(
        pl.col("home_team").replace(name_map),
        pl.col("away_team").replace(name_map),
    )
    with open(worldcup_path, "r") as f:
        wc = json.load(f)
    return res, wc


def build_model_df(res: pl.DataFrame, ref: date, half_life: int) -> pl.DataFrame:
    return (
        res
        .with_columns(pl.col("date").str.to_date())
        .drop_nulls(["home_score", "away_score"])
        .filter(pl.col("date") <= ref)
        .with_columns(
            weight=pl.lit(0.5).pow(
                (pl.lit(ref) - pl.col("date")).dt.total_days() / half_life
            )
        )
        # Vänskapsmatcher säger mindre om verklig styrka, halvera deras vikt
        .with_columns(
            pl.when(pl.col("tournament") == "Friendly")
            .then(pl.col("weight") * 0.5)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        .select(
            "date", "home_team", "away_team",
            "home_score", "away_score", "neutral", "weight",
        )
    )


def select_common_teams(
    df_model: pl.DataFrame,
    wc: dict,
    ref: date,
    window_years: int = 8,
    min_matches: int = 15,
) -> tuple[set[str], pl.DataFrame]:
    wc_teams = {
        m[k]
        for m in wc["matches"]
        for k in ("team1", "team2")
        if m.get("group") and m.get(k)
    }
    win = df_model.filter(pl.col("date") >= ref - timedelta(days=365 * window_years))
    cnt = Counter(win["home_team"].to_list()) + Counter(win["away_team"].to_list())
    # Lag med få matcher får egna parametrar bara om de spelar VM, annars "Other"
    common = {t for t, c in cnt.items() if c >= min_matches} | wc_teams
    return common, win


def apply_team_grouping(win: pl.DataFrame, common: set[str]) -> pl.DataFrame:
    return win.with_columns(
        pl.when(pl.col("home_team").is_in(common)).then(pl.col("home_team"))
          .otherwise(pl.lit("Other")).alias("home_team"),
        pl.when(pl.col("away_team").is_in(common)).then(pl.col("away_team"))
          .otherwise(pl.lit("Other")).alias("away_team"),
    )


def build_arrays(
    df_fit: pl.DataFrame,
) -> tuple[
    list[str], dict[str, int], int,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    teams = sorted(set(df_fit["home_team"]) | set(df_fit["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = np.array([idx[t] for t in df_fit["home_team"]])
    a = np.array([idx[t] for t in df_fit["away_team"]])
    x = df_fit["home_score"].to_numpy()
    y = df_fit["away_score"].to_numpy()
    w = df_fit["weight"].to_numpy()
    home_flag = np.where(df_fit["neutral"].to_numpy(), 0.0, 1.0)
    return teams, idx, n, h, a, x, y, w, home_flag


def neg_loglik(
    p: np.ndarray,
    n: int,
    h: np.ndarray,
    a: np.ndarray,
    home_flag: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> float:
    # Sista attackparametern är låst så att summan blir noll (identifierbarhet)
    attack = np.append(p[:n - 1], -p[:n - 1].sum())
    defense = p[n - 1:2 * n - 1]
    gamma = p[2 * n - 1]
    rho = p[2 * n]
    log_lam = attack[h] - defense[a] + gamma * home_flag
    log_mu = attack[a] - defense[h]
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    tau = np.ones_like(lam)
    m = (x == 0) & (y == 0)
    tau[m] = 1 - lam[m] * mu[m] * rho
    m = (x == 0) & (y == 1)
    tau[m] = 1 + lam[m] * rho
    m = (x == 1) & (y == 0)
    tau[m] = 1 + mu[m] * rho
    m = (x == 1) & (y == 1)
    tau[m] = 1 - rho
    tau = np.clip(tau, 1e-10, None)
    ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)

    return -ll.sum()


def fit_model(df_fit: pl.DataFrame) -> dict:
    teams, idx, n, h, a, x, y, w, home_flag = build_arrays(df_fit)
    p0 = np.zeros(2 * n + 1)
    p0[2 * n - 1] = 0.25
    p0[2 * n] = -0.05
    fit = minimize(
        neg_loglik, p0, args=(n, h, a, home_flag, x, y, w), method="L-BFGS-B"
    )

    attack = np.append(fit.x[:n - 1], -fit.x[:n - 1].sum())
    return {
        "teams": teams,
        "idx": idx,
        "attack": attack,
        "defense": fit.x[n - 1:2 * n - 1],
        "gamma": fit.x[2 * n - 1],
        "rho": fit.x[2 * n],
    }


def score_matrix(
    home: str, away: str, model: dict, neutral: bool = True
) -> np.ndarray:
    idx, attack, defense = model["idx"], model["attack"], model["defense"]
    gamma, rho = model["gamma"], model["rho"]
    hf = 0.0 if neutral else 1.0
    lam = np.exp(attack[idx[home]] - defense[idx[away]] + gamma * hf)
    mu = np.exp(attack[idx[away]] - defense[idx[home]])
    p = np.outer(
        poisson.pmf(np.arange(11), lam),
        poisson.pmf(np.arange(11), mu),
    )
    # Dixon-Coles-korrigering för låga resultat
    p[0, 0] *= 1 - lam * mu * rho
    p[0, 1] *= 1 + lam * rho
    p[1, 0] *= 1 + mu * rho
    p[1, 1] *= 1 - rho

    return p / p.sum()


def vmtipset_score(ah: int, aa: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = 2 * (x == ah).astype(float)
    pts += 2 * (y == aa).astype(float)
    pts += 3 * (np.sign(ah - aa) == np.sign(x - y)).astype(float)

    return pts


def best_tip(p: np.ndarray, score_fn) -> tuple[tuple[int, int], float]:
    xs, ys = np.indices(p.shape)
    best, best_ev = None, -1.0
    for a in range(p.shape[0]):
        for b in range(p.shape[1]):
            ev = (p * score_fn(a, b, xs, ys)).sum()
            if ev > best_ev:
                best_ev, best = ev, (a, b)

    return best, best_ev


def predict(
    home: str, away: str, model: dict, neutral: bool = True
) -> tuple[int, int, float, float, float, float]:
    p = score_matrix(home, away, model, neutral)
    (a, b), ev = best_tip(p, vmtipset_score)
    p1 = np.tril(p, -1).sum()
    px = np.trace(p)
    p2 = np.triu(p, 1).sum()

    return a, b, ev, p1, px, p2


def predict_all(wc: dict, model: dict) -> pl.DataFrame:
    rows = []
    for m in wc["matches"]:
        if m.get("group") and m.get("team1") and m.get("team2"):
            ht, at = m["team1"], m["team2"]
            a, b, ev, p1, px, p2 = predict(ht, at, model)
            rows.append({
                "group": m["group"],
                "home_team": ht,
                "away_team": at,
                "home_goals": a,
                "away_goals": b,
                "ev": ev,
                "p1": p1,
                "px": px,
                "p2": p2,
            })
    return pl.DataFrame(rows)


def predict_winner(
    home: str,
    away: str,
    model: dict,
    rng: np.random.Generator,
    neutral: bool = True,
) -> tuple[str, int, int]:
    a, b, _, _, _, _ = predict(home, away, model, neutral)
    if a > b:
        return home, a, b
    if b > a:
        return away, a, b
    # Oavgjort i slutspel avgörs med slantsingling
    return str(rng.choice([home, away])), a, b


# Slutspelsrundor mappade till hur långt ett lag tagit sig
STAGE = {"Round of 32": 1, "Round of 16": 2, "Quarter-final": 3, "Semi-final": 4, "Final": 5}


def precompute_matrices(wc: dict, model: dict) -> dict[tuple[str, str], np.ndarray]:
    teams = sorted(
        {m[k] for m in wc["matches"] for k in ("team1", "team2")
         if m.get("group") and m.get(k)}
    )
    return {
        (h, a): score_matrix(h, a, model)
        for h in teams for a in teams if h != a
    }


def sample_match(
    home: str, away: str, mats: dict[tuple[str, str], np.ndarray], rng: np.random.Generator
) -> tuple[int, int]:
    p = mats[(home, away)]
    k = int(rng.choice(p.size, p=p.ravel()))
    return divmod(k, p.shape[1])


def match_thirds(
    slots: list[tuple[str, set[str]]], qualified: set[str]
) -> dict[str, str]:
    # Tilldela varje slutspelsplats en av de åtta kvalificerade trean-grupperna
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def bt(i: int) -> bool:
        if i == len(slots):
            return True
        code, allowed = slots[i]
        for g in allowed & qualified:
            if g not in used:
                used.add(g)
                assignment[code] = g
                if bt(i + 1):
                    return True
                used.remove(g)
                del assignment[code]
        return False

    bt(0)
    return assignment


def simulate_tournament(
    wc: dict, mats: dict[tuple[str, str], np.ndarray], rng: np.random.Generator
) -> tuple[dict[str, int], tuple[str, str, int, int]]:
    matches = wc["matches"]

    groups: dict[str, set[str]] = {}
    for m in matches:
        if m.get("group"):
            g = m["group"].split()[1]
            groups.setdefault(g, set()).update([m["team1"], m["team2"]])

    pts: Counter = Counter()
    gf: Counter = Counter()
    ga: Counter = Counter()
    for m in matches:
        if m.get("group"):
            h, a = m["team1"], m["team2"]
            hg, ag = sample_match(h, a, mats, rng)
            gf[h] += hg
            ga[h] += ag
            gf[a] += ag
            ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif ag > hg:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1

    # Sortering: poäng, målskillnad, gjorda mål, sedan slumpmässigt
    def key(t):
        return (pts[t], gf[t] - ga[t], gf[t], rng.random())

    group_rank = {g: sorted(ts, key=key, reverse=True) for g, ts in groups.items()}

    # De åtta bästa treorna går vidare
    best_groups = sorted(groups, key=lambda g: key(group_rank[g][2]), reverse=True)[:8]
    qualified = set(best_groups)

    slots = [
        (c, set(c[1:].split("/")))
        for m in matches if m.get("round") == "Round of 32"
        for c in (m["team1"], m["team2"]) if c.startswith("3")
    ]
    third_assign = match_thirds(slots, qualified)

    stage = {t: 0 for ts in groups.values() for t in ts}
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    final: tuple[str, str, int, int] = ("", "", 0, 0)

    def resolve(code: str) -> str:
        c0 = code[0]
        if c0 == "W":
            return winners[int(code[1:])]
        if c0 == "L":
            return losers[int(code[1:])]
        if c0 in ("1", "2"):
            return group_rank[code[1]][int(c0) - 1]
        return group_rank[third_assign[code]][2]

    for m in matches:
        if m.get("group"):
            continue
        t1, t2 = resolve(m["team1"]), resolve(m["team2"])
        r = m["round"]
        if r in STAGE:
            stage[t1] = max(stage[t1], STAGE[r])
            stage[t2] = max(stage[t2], STAGE[r])
        hg, ag = sample_match(t1, t2, mats, rng)
        if hg > ag:
            w = t1
        elif ag > hg:
            w = t2
        else:
            # Oavgjort avgörs med slantsingling (motsvarar straffar)
            w = str(rng.choice([t1, t2]))
        l = t2 if w == t1 else t1
        num = m.get("num")
        if num is not None:
            winners[num] = w
            losers[num] = l
        if r == "Final":
            stage[w] = 6
            final = (t1, t2, hg, ag)

    return stage, final


def monte_carlo(
    wc: dict, model: dict, n_sims: int, seed: int = 0
) -> tuple[
    pl.DataFrame,
    list[tuple[tuple[str, str, int, int], int]],
    list[tuple[tuple[str, str], int]],
]:
    rng = np.random.default_rng(seed)
    mats = precompute_matrices(wc, model)

    team_group = {}
    for m in wc["matches"]:
        if m.get("group"):
            team_group[m["team1"]] = m["group"]
            team_group[m["team2"]] = m["group"]
    teams = sorted(team_group)

    counts = {t: np.zeros(7, dtype=int) for t in teams}
    finals: Counter = Counter()
    matchups: Counter = Counter()
    for _ in range(n_sims):
        stage, final = simulate_tournament(wc, mats, rng)
        for t, s in stage.items():
            counts[t][s] += 1
        finals[final] += 1
        matchups[tuple(sorted(final[:2]))] += 1

    rows = []
    for t in teams:
        c = counts[t]
        rows.append({
            "team": t,
            "group": team_group[t],
            "p_knockout": c[1:].sum() / n_sims,
            "p_r16": c[2:].sum() / n_sims,
            "p_qf": c[3:].sum() / n_sims,
            "p_sf": c[4:].sum() / n_sims,
            "p_final": c[5:].sum() / n_sims,
            "p_win": c[6] / n_sims,
        })
    return (
        pl.DataFrame(rows).sort("p_win", descending=True),
        finals.most_common(10),
        matchups.most_common(10),
    )
