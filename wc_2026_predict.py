import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from datetime import date, timedelta
    from collections import Counter
    import polars as pl
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import poisson
    import plotly.express as px
    import plotly.io as pio

    return Counter, date, json, minimize, np, pio, pl, poisson, timedelta


@app.cell
def _(pio):
    pio.templates.default = "plotly_dark"
    return


@app.cell
def _(np):
    rng = np.random.default_rng(0)
    return (rng,)


@app.cell
def _(json, pl):
    name_map = {"Bosnia and Herzegovina": "Bosnia & Herzegovina", "United States": "USA"}
    res = (
        pl.read_csv("data/results.csv", null_values=["NA"])
        .with_columns(
            pl.col("home_team").replace(name_map),
            pl.col("away_team").replace(name_map)
        )
    )
    with open("data/worldcup.json", "r") as _f:
        wc = json.load(_f)
    return res, wc


@app.cell
def _(date):
    ref = date(2026, 6, 11)
    half_life = 365 * 2
    return half_life, ref


@app.cell
def _(half_life, pl, ref, res):
    df_model = (
        res
        .with_columns(
            pl.col("date").str.to_date()
        )
        .drop_nulls(["home_score", "away_score"])
        .filter(pl.col("date") <= ref)
        .with_columns(
            weight = pl.lit(0.5).pow((pl.lit(ref) - pl.col("date")).dt.total_days() / half_life)
        )
        .with_columns(
            pl.when(pl.col("tournament") == "Friendly")
            .then(pl.col("weight") * 0.5)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        .select("date", "home_team", "away_team", "home_score", "away_score", "neutral", "weight")
    )
    df_model
    return (df_model,)


@app.cell
def _(Counter, df_model, pl, ref, timedelta, wc):
    wc_teams = {m[k] for m in wc["matches"] for k in ("team1", "team2") if m.get("group") and m.get(k)}
    win = df_model.filter(pl.col("date") >= ref - timedelta(days=365 * 8))
    cnt = Counter(win["home_team"].to_list()) + Counter(win["away_team"].to_list())
    common = {t for t, c in cnt.items() if c >= 15} | wc_teams
    return common, win


@app.cell
def _(common, pl, win):
    df_fit = win.with_columns(
        pl.when(pl.col("home_team").is_in(common)).then(pl.col("home_team"))
          .otherwise(pl.lit("Other")).alias("home_team"),
        pl.when(pl.col("away_team").is_in(common)).then(pl.col("away_team"))
          .otherwise(pl.lit("Other")).alias("away_team"),
    )
    return (df_fit,)


@app.cell
def _(df_fit, np):
    teams = sorted(set(df_fit["home_team"]) | set(df_fit["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = np.array([idx[t] for t in df_fit["home_team"]])
    a = np.array([idx[t] for t in df_fit["away_team"]])
    x = df_fit["home_score"].to_numpy()
    y = df_fit["away_score"].to_numpy()
    w = df_fit["weight"].to_numpy()
    home_flag = np.where(df_fit["neutral"].to_numpy(), 0.0, 1.0)
    return a, h, home_flag, idx, n, w, x, y


@app.cell
def _(a, h, home_flag, n, np, w, x, y):
    def neg_loglik(p):
        attack = np.append(p[:n-1], -p[:n-1].sum())
        defense = p[n-1:2*n-1]
        gamma = p[2*n-1]
        rho = p[2*n]
        log_lam = attack[h] - defense[a] + gamma * home_flag
        log_mu = attack[a] - defense[h]
        lam, mu = np.exp(log_lam), np.exp(log_mu)
        tau = np.ones_like(lam)
        m = (x==0) & (y==0)
        tau[m] = 1 - lam[m] * mu[m] * rho
        m = (x==0) & (y==1)
        tau[m] = 1 + lam[m] * rho
        m = (x==1) & (y==0)
        tau[m] = 1 + mu[m] * rho
        m = (x==1) & (y==1)
        tau[m] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)
    
        return -ll.sum()

    return (neg_loglik,)


@app.cell
def _(minimize, n, neg_loglik, np):
    p0 = np.zeros(2 * n + 1)
    p0[2*n-1] = 0.25
    p0[2*n] = -0.05
    fit = minimize(neg_loglik, p0, method="L-BFGS-B")

    attack = np.append(fit.x[:n-1], -fit.x[:n-1].sum())
    defense = fit.x[n-1:2*n-1]
    gamma, rho = fit.x[2*n-1], fit.x[2*n]
    return attack, defense, gamma, rho


@app.cell
def _(attack, defense, gamma, idx, np, poisson, rho):
    def score_matrix(home, away, neutral=True):
        hf = 0.0 if neutral else 1.0
        lam = np.exp(attack[idx[home]] - defense[idx[away]] + gamma * hf)
        mu = np.exp(attack[idx[away]] - defense[idx[home]])
        p = np.outer(
            poisson.pmf(np.arange(11), lam),
            poisson.pmf(np.arange(11), mu)
        )
        p[0, 0] *= 1 - lam * mu * rho
        p[0, 1] *= 1 + lam * rho
        p[1, 0] *= 1 + mu * rho
        p[1, 1] *= 1 - rho

        return p / p.sum()

    def vmtipset_score(ah, aa, x, y):
        pts = 2 * (x == ah).astype(float)
        pts += 2 * (y == aa).astype(float)
        pts += 3 * (np.sign(ah - aa) == np.sign(x - y)).astype(float)

        return pts

    def best_tip(p, score_fn):
        xs, ys = np.indices(p.shape)
        best, best_ev = None, -1.0
        for a in range(p.shape[0]):
            for b in range(p.shape[1]):
                ev = (p * score_fn(a, b, xs, ys)).sum()
                if ev > best_ev:
                    best_ev, best = ev, (a, b)

        return best, best_ev

    def predict(home, away, neutral=True):
        p = score_matrix(home, away, neutral)
        (a, b), ev = best_tip(p, vmtipset_score)
        p1 = np.tril(p, -1).sum()
        px = np.trace(p)
        p2 = np.triu(p, 1).sum()

        return a, b, ev, p1, px, p2

    return (predict,)


@app.cell
def _(predict, wc):
    for m in wc["matches"]:
        if m.get("group") and m.get("team1") and m.get("team2"):
            ht, at = m["team1"], m["team2"]
            _a, _b, _ev, _p1, _px, _p2 = predict(ht, at)
            print(f"{ht:20s} {_a}-{_b} {at:20s} EV={_ev:.2f} (1 {_p1:.0%} / X {_px:.0%} / 2 {_p2:.0%}")
    return


@app.cell
def _(predict):
    home = "Spain"
    away = "Argentina"
    _a, _b, _ev, _p1, _px, _p2 = predict(home, away)
    print(f"{home:20s} {_a}-{_b} {away:20s} EV={_ev:.2f} (1 {_p1:.0%} / X {_px:.0%} / 2 {_p2:.0%}")
    return away, home


@app.cell
def _(away, home, rng):
    rng.choice([home, away])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
