import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    from datetime import date
    import polars as pl

    from utils import (
        load_data,
        build_model_df,
        select_common_teams,
        apply_team_grouping,
        fit_model,
        predict,
        predict_all,
        predict_winner,
        monte_carlo,
    )

    return (
        apply_team_grouping,
        build_model_df,
        date,
        fit_model,
        load_data,
        mo,
        monte_carlo,
        np,
        pl,
        predict,
        predict_all,
        predict_winner,
        px,
        select_common_teams,
    )


@app.cell
def _(mo, px):
    import plotly.io as pio
    pio.templates.default = "plotly_dark" if  mo.app_meta().theme == "dark" else "ggplot2"
    pio.templates[pio.templates.default].layout.colorway = px.colors.qualitative.T10
    return


@app.cell
def _(np):
    rng = np.random.default_rng(0)
    return (rng,)


@app.cell
def _(load_data):
    name_map = {"Bosnia and Herzegovina": "Bosnia & Herzegovina", "United States": "USA"}
    res, wc = load_data("data/results.csv", "data/worldcup.json", name_map)
    return res, wc


@app.cell
def _(date):
    ref = date(2026, 6, 11)
    half_life = 365 * 2
    return half_life, ref


@app.cell
def _(build_model_df, half_life, ref, res):
    df_model = build_model_df(res, ref, half_life)
    return (df_model,)


@app.cell
def _(df_model, ref, select_common_teams, wc):
    common, win = select_common_teams(df_model, wc, ref)
    return common, win


@app.cell
def _(apply_team_grouping, common, win):
    df_fit = apply_team_grouping(win, common)
    return (df_fit,)


@app.cell
def _(df_fit, fit_model):
    model = fit_model(df_fit)
    return (model,)


@app.cell
def _(model, predict_all, wc):
    df_pred = predict_all(wc, model)
    df_pred
    return


@app.cell
def _(model, predict_winner, rng):
    home = "Spain"
    away = "Argentina"
    winner, home_goals, away_goals = predict_winner(home, away, model, rng)
    print(f"{home} {home_goals}-{away_goals} {away} → {winner}")
    return


@app.cell
def _(model, predict):
    _a, _b, _ev, _p1, _px, _p2 = predict("Brazil", "Germany", model)
    print(f"{_a}-{_b} EV={_ev:.2f} (1 {_p1:.0%} / X {_px:.0%} / 2 {_p2:.0%})")
    return


@app.cell
def _(model, monte_carlo, wc):
    n_sims = 100000
    df_sim, top_finals, top_matchups = monte_carlo(wc, model, n_sims=n_sims)
    df_sim
    return df_sim, n_sims, top_finals, top_matchups


@app.cell
def _(df_sim, pl, px):
    _fig = px.bar(df_sim.sort("p_win").tail(10).with_columns(pl.col("p_win") * 100), y="team", x="p_win", text_auto=".1f")
    _fig.update_layout(width=800, height=500, xaxis_title="Vinstchans (%)", yaxis_title="Land")
    return


@app.cell
def _(df_sim, pl, px):
    stage_map = {
        "p_knockout": "16-delsfinal",
        "p_r16": "Åttondelsfinal",
        "p_qf": "Kvartsfinal",
        "p_sf": "Semifinal",
        "p_final": "Final",
        "p_win": "Vinst"
    }
    _df_plot = (
        df_sim
        .filter(pl.col("team") == "Sweden")
        .unpivot(index=["team", "group"], variable_name="Steg", value_name="Sannolikhet (%)")
        .with_columns(
            pl.col("Steg").replace(stage_map),
            pl.col("Sannolikhet (%)") * 100
        )
    )
    px.bar(_df_plot.sort("Sannolikhet (%)"), x="Sannolikhet (%)", y="Steg", text_auto=".2f")
    return


@app.cell
def _(n_sims, pl, top_matchups):
    df_final_teams = pl.DataFrame([{"Finallag": " - ".join(tt), "Andel (%)": n/n_sims * 100} for tt, n in top_matchups])
    return (df_final_teams,)


@app.cell
def _(df_final_teams, px):
    px.bar(df_final_teams.sort("Andel (%)"), y="Finallag", x="Andel (%)", text_auto=True)
    return


@app.cell
def _(n_sims, pl, top_finals):
    df_final_result = pl.DataFrame(
        [
            {"Finalresultat": f"{tt[0]} {tt[2]} - {tt[3]} {tt[1]}", "Andel (%)": n / n_sims * 100}
            for tt, n in  top_finals
        ]
    )
    return (df_final_result,)


@app.cell
def _(df_final_result, px):
    px.bar(df_final_result.sort("Andel (%)"), y="Finalresultat", x="Andel (%)", text_auto=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
