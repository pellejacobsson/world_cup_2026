import json
from datetime import date

import polars as pl

from utils import (
    apply_team_grouping,
    build_model_df,
    fit_model,
    load_data,
    monte_carlo,
    most_likely_tournament,
    select_common_teams,
)

name_map = {"Bosnia and Herzegovina": "Bosnia & Herzegovina", "United States": "USA"}
res, wc = load_data("data/results.csv", "data/worldcup.json", name_map)

# Faktiska VM 2026-resultat per spelad match. Hålls utanför modellträningen så att
# lagstyrkorna är oförändrade — de används bara för att låsa redan spelade matcher.
wc2026 = (pl.col("tournament") == "FIFA World Cup") & (pl.col("date") >= "2026-06-11")
actual = {
    (r["home_team"], r["away_team"]): (r["home_score"], r["away_score"])
    for r in res.filter(wc2026).drop_nulls(["home_score", "away_score"]).iter_rows(named=True)
}

ref = date(2026, 6, 11)
half_life = 365 * 2

df_model = build_model_df(res.filter(~wc2026), ref, half_life)
common, win = select_common_teams(df_model, wc, ref)
df_fit = apply_team_grouping(win, common)
model = fit_model(df_fit)

# Samla VM-lag per grupp (sorterat alfabetiskt inom grupp, grupper i bokstavsordning)
teams_by_group: dict[str, list[str]] = {}
for m in wc["matches"]:
    if m.get("group"):
        teams_by_group.setdefault(m["group"], set()).update([m["team1"], m["team2"]])
teams_by_group = {g: sorted(ts) for g, ts in sorted(teams_by_group.items())}

# Lag med få historiska matcher klumpas som "Other" i modellen — använd dess parametrar
idx = model["idx"]
team_params: dict[str, dict] = {}
for g, ts in teams_by_group.items():
    for t in ts:
        key = t if t in idx else "Other"
        team_params[t] = {
            "group": g,
            "attack": float(model["attack"][idx[key]]),
            "defense": float(model["defense"][idx[key]]),
            "isOther": key == "Other",
        }

df_group, df_standings, df_knockout, champion = most_likely_tournament(wc, model, actual)

# Slutspelsträdet delas i två halvor: vänster matar finalens team1, höger team2.
# round_order ger matchnumren i en runda i vertikal ordning (så feeders radar upp sig).
ko_by_num = {m["num"]: m for m in wc["matches"] if not m.get("group") and m.get("num")}

def round_order(num: int, rnd: str) -> list[int]:
    m = ko_by_num[num]
    if m["round"] == rnd:
        return [num]
    feeders = [int(c[1:]) for c in (m["team1"], m["team2"]) if c.startswith("W")]
    return [n for f in feeders for n in round_order(f, rnd)]

final_m = next(m for m in wc["matches"] if m["round"] == "Final")
sf_left, sf_right = int(final_m["team1"][1:]), int(final_m["team2"][1:])
ko_rounds = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final"]
# feeders: för varje numrerad slutspelsmatch, vilka två matcher som matar in (för förbindelselinjer)
ko_feeders = {
    m["num"]: fs
    for m in wc["matches"] if not m.get("group") and m.get("num")
    if (fs := [int(c[1:]) for c in (m["team1"], m["team2"]) if c.startswith("W")])
}
bracket = {
    "left": [round_order(sf_left, r) for r in ko_rounds],
    "right": [round_order(sf_right, r) for r in ko_rounds],
    "feeders": ko_feeders,
    "finalFeeders": [sf_left, sf_right],
}

n_sims = 100_000
df_sim, _, _ = monte_carlo(wc, model, n_sims=n_sims, seed=0, actual=actual)

dashboard_data = {
    "teamsByGroup": teams_by_group,
    "teamParams": team_params,
    "gamma": float(model["gamma"]),
    "rho": float(model["rho"]),
    "tournament": {
        "groupMatches": df_group.to_dicts(),
        "standings": df_standings.to_dicts(),
        "knockout": df_knockout.to_dicts(),
        "champion": champion,
    },
    "bracket": bracket,
    "teamProbabilities": df_sim.to_dicts(),
    "nSims": n_sims,
}

HTML = r"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VM 2026 – prediktion</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
<style>
  :root { --primary: #0a2540; --primary-soft: #13294b; --accent: #b08d57; --accent-dark: #8b6f3f; --bg: #faf6ef; --card: #ffffff; --border: #e6dfd0; --muted: #6b6358; --text: #1a1f2e; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
  header { background: var(--primary); color: white; padding: 18px 28px; }
  header h1 { margin: 0; font-size: 22px; letter-spacing: 0.3px; }
  header .sub { font-size: 13px; opacity: 0.85; margin-top: 4px; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: white; padding: 0 28px; position: sticky; top: 0; z-index: 10; }
  .tab { padding: 14px 22px; cursor: pointer; border: none; background: none; font-size: 15px; color: var(--muted); border-bottom: 3px solid transparent; font-family: inherit; }
  .tab.active { color: var(--primary); border-bottom-color: var(--primary); font-weight: 600; }
  .panel { display: none; padding: 24px 28px; max-width: 1320px; margin: 0 auto; }
  .panel.active { display: block; }
  .controls { display: flex; gap: 20px; align-items: end; margin-bottom: 20px; flex-wrap: wrap; }
  .control label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.7px; font-weight: 600; }
  .control select { padding: 9px 12px; font-size: 15px; border: 1px solid #c9c0ad; border-radius: 6px; min-width: 240px; background: white; font-family: inherit; color: var(--text); }
  .control.toggle { display: flex; align-items: center; gap: 8px; padding-bottom: 9px; }
  .control.toggle label { margin: 0; text-transform: none; letter-spacing: 0; font-size: 14px; color: var(--text); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  .card h3 { margin: 0 0 12px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.7px; font-weight: 600; }
  .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; text-align: center; }
  .stat .value { font-size: 30px; font-weight: 700; color: var(--primary); line-height: 1.1; }
  .stat .label { font-size: 12px; color: var(--muted); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .match-grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 18px; }
  @media (max-width: 900px) { .match-grid { grid-template-columns: 1fr; } }
  @media (max-width: 640px) {
    header { padding: 14px 16px; }
    header h1 { font-size: 19px; }
    header .sub { font-size: 12px; line-height: 1.3; }
    .tabs { padding: 0 8px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .tab { padding: 12px 14px; font-size: 14px; white-space: nowrap; }
    .panel { padding: 16px 12px; }
    .controls { gap: 10px; }
    .control { width: 100%; }
    .control select { min-width: 0; width: 100%; }
    .control.toggle { width: auto; padding-bottom: 4px; }
    .summary { grid-template-columns: 1fr; gap: 10px; margin-bottom: 16px; }
    .stat { padding: 14px; }
    .stat .value { font-size: 26px; }
    .card { padding: 14px; }
    #heatmap { height: 360px !important; }
    .group-grid { grid-template-columns: 1fr; gap: 12px; }
    .ko-grid { grid-template-columns: 1fr; gap: 12px; }
    .ko-row { font-size: 13px; gap: 6px; }
    .ko-row .score { padding: 2px 6px; }
    .champion-banner { padding: 20px; }
    .champion-banner .name { font-size: 28px; }
    .champion-banner .label { font-size: 11px; letter-spacing: 1.5px; }
    table { font-size: 13px; }
    th, td { padding: 6px 5px; }
    .tournament-section h2 { font-size: 16px; }
  }
  .other-note { font-size: 12px; color: var(--accent); margin-top: 4px; }
  .tournament-section { margin-bottom: 32px; }
  .tournament-section h2 { font-size: 18px; border-bottom: 2px solid var(--primary); padding-bottom: 6px; margin-bottom: 16px; }
  .group-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 7px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; background: #f5efe0; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.center, th.center { text-align: center; }
  .qualifier { background: #f3ede1; }
  .qualifier td:first-child { border-left: 3px solid var(--accent); padding-left: 5px; }
  .ko-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .ko-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 14px; }
  .ko-row:last-child { border-bottom: none; }
  .ko-row .t1 { text-align: right; }
  .ko-row .score { font-weight: 700; padding: 2px 10px; background: #f3ede1; border-radius: 4px; font-variant-numeric: tabular-nums; color: var(--text); }
  .ko-row .winner { color: var(--primary); font-weight: 600; }
  .ko-row .loser { color: #a89e88; }
  .champion-banner { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-soft) 100%); padding: 28px; border-radius: 12px; text-align: center; color: #faf6ef; margin-top: 16px; box-shadow: 0 6px 18px rgba(10, 37, 64, 0.25); border: 1px solid var(--accent); position: relative; overflow: hidden; }
  .champion-banner::before { content: ""; position: absolute; inset: 0; background: linear-gradient(135deg, transparent 60%, rgba(176, 141, 87, 0.18) 100%); pointer-events: none; }
  .champion-banner .label { font-size: 13px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--accent); font-weight: 600; position: relative; }
  .champion-banner .name { font-size: 42px; font-weight: 800; margin-top: 8px; letter-spacing: 0.5px; position: relative; }
  .legend { font-size: 12px; color: var(--muted); margin-top: 8px; margin-bottom: 12px; }
  .legend .dot { display: inline-block; width: 10px; height: 10px; background: var(--accent); border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .ms { font-weight: 700; font-variant-numeric: tabular-nums; }
  .ms.played { color: var(--accent-dark); }
  .ms.predicted { color: var(--muted); font-weight: 600; }
  .played-mark { color: var(--accent-dark); font-size: 12px; font-weight: 700; }
  .prose { max-width: 860px; }
  .prose .card { margin-bottom: 18px; }
  .prose .card h2 { font-size: 18px; margin: 0 0 12px; color: var(--primary); border-bottom: 2px solid var(--primary); padding-bottom: 6px; }
  .prose p { line-height: 1.65; margin: 0 0 12px; }
  .prose ul { line-height: 1.65; margin: 0 0 12px; padding-left: 22px; }
  .prose li { margin-bottom: 7px; }
  .prose strong { color: var(--text); }
  .formula { background: #f5efe0; border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 14px; overflow-x: auto; margin: 4px 0 14px; color: var(--primary); }
  .param { display: inline-block; background: #f3ede1; border-radius: 4px; padding: 1px 7px; font-variant-numeric: tabular-nums; font-weight: 600; color: var(--primary); }
  /* Slutspelsträd */
  .bracket-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 8px; }
  .bracket { position: relative; display: flex; align-items: stretch; min-width: 1560px; min-height: 860px; }
  .bracket svg.bconn { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 0; overflow: visible; }
  .bside { display: flex; flex: 1; position: relative; z-index: 1; }
  .bside.right { flex-direction: row-reverse; }
  .bcol { display: flex; flex-direction: column; flex: 1; min-width: 170px; }
  .bcol-head { height: 24px; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; text-align: center; }
  .bcol-body { flex: 1; display: flex; flex-direction: column; justify-content: space-around; padding: 0 14px; }
  /* En match = två staplade lagrutor (samma storlek på alla nivåer) */
  .bmatch { position: relative; }
  .bteam { display: grid; grid-template-columns: 1fr auto auto; gap: 7px; align-items: center; height: 27px; padding: 0 9px; font-size: 13px; background: var(--card); border: 1px solid var(--border); }
  .bteam:first-child { border-radius: 6px 6px 0 0; }
  .bteam:last-child { border-radius: 0 0 6px 6px; border-top: none; }
  .bt-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
  .bt-prob { font-variant-numeric: tabular-nums; font-size: 11px; text-align: right; min-width: 30px; color: var(--muted); }
  .bt-goal { font-variant-numeric: tabular-nums; font-weight: 700; text-align: right; min-width: 11px; color: var(--text); }
  /* Vinnaren: fylld i primärfärg */
  .bteam-win { background: var(--primary); border-color: var(--primary); }
  .bteam-win .bt-name { color: #fff; font-weight: 700; }
  .bteam-win .bt-prob { color: var(--accent); }
  .bteam-win .bt-goal { color: #fff; }
  .bteam-lose .bt-name { color: var(--muted); }
  .bteam-lose .bt-prob, .bteam-lose .bt-goal { color: #b3aa97; }
  .bmatch-played .bteam-lose { background: #fbf8f2; }
  .bmatch-played .bt-mark { position: absolute; top: -7px; font-size: 11px; color: var(--accent-dark); background: var(--bg); padding: 0 3px; font-weight: 700; }
  .bside.left .bmatch-played .bt-mark { right: 2px; }
  .bside.right .bmatch-played .bt-mark { left: 2px; }
  /* Final + mitten */
  .bcenter { display: flex; flex-direction: column; justify-content: center; align-items: center; min-width: 196px; padding: 0 6px; position: relative; z-index: 1; }
  .bcenter .bcol-head { height: auto; margin-bottom: 8px; }
  .bfinal { width: 100%; }
  .bfinal .bteam-win { box-shadow: 0 0 0 2px var(--accent); }
  .bchamp { width: 100%; text-align: center; margin-top: 14px; padding: 12px; background: linear-gradient(135deg, var(--primary) 0%, var(--primary-soft) 100%); border: 1px solid var(--accent); border-radius: 10px; box-shadow: 0 4px 14px rgba(10,37,64,0.2); }
  .bchamp .label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); font-weight: 600; }
  .bchamp .name { font-size: 24px; font-weight: 800; color: #fff; margin-top: 4px; }
  .bthird { width: 100%; margin-top: 18px; }
  .bthird .bcol-head { color: var(--accent-dark); height: auto; margin-bottom: 6px; }
</style>
</head>
<body>
<header>
  <h1>VM 2026 – prediktion</h1>
  <div class="sub">Dixon-Coles Poisson-modell anpassad på landskamper t.o.m. 2026-06-11</div>
</header>

<div class="tabs">
  <button class="tab active" data-tab="match">Matchprediktor</button>
  <button class="tab" data-tab="tournament">Mest troliga turnering</button>
  <button class="tab" data-tab="bracket">Slutspelsträd</button>
  <button class="tab" data-tab="team">Lagets chanser</button>
  <button class="tab" data-tab="model">Om modellen</button>
</div>

<div class="panel active" id="panel-match">
  <div class="controls">
    <div class="control">
      <label>Lag A (hemma)</label>
      <select id="team-a"></select>
    </div>
    <div class="control">
      <label>Lag B (borta)</label>
      <select id="team-b"></select>
    </div>
    <div class="control toggle">
      <input type="checkbox" id="neutral" checked>
      <label for="neutral">Neutral plan</label>
    </div>
  </div>
  <div id="other-warning" class="other-note" style="display:none;"></div>

  <div class="summary">
    <div class="stat"><div class="value" id="p1">–</div><div class="label" id="lab-1">Vinst lag A</div></div>
    <div class="stat"><div class="value" id="px">–</div><div class="label">Oavgjort</div></div>
    <div class="stat"><div class="value" id="p2">–</div><div class="label" id="lab-2">Vinst lag B</div></div>
  </div>

  <div class="match-grid">
    <div class="card">
      <h3>Sannolikhet per resultat (%)</h3>
      <div id="heatmap" style="height: 460px;"></div>
    </div>
    <div class="card">
      <h3>Sammanfattning</h3>
      <table>
        <tr><th>Förväntade mål lag A</th><td class="num" id="ev-a"></td></tr>
        <tr><th>Förväntade mål lag B</th><td class="num" id="ev-b"></td></tr>
        <tr><th>Mest troliga resultat</th><td class="num" id="mode"></td></tr>
        <tr><th>Sannolikhet för det resultatet</th><td class="num" id="mode-p"></td></tr>
        <tr><th>Optimalt VM-tipset-tips</th><td class="num" id="tip"></td></tr>
        <tr><th>Förväntade poäng (VM-tipset)</th><td class="num" id="tip-ev"></td></tr>
      </table>
    </div>
  </div>
</div>

<div class="panel" id="panel-tournament">
  <div class="tournament-section">
    <h2>Gruppspel</h2>
    <div class="legend"><span class="dot"></span> Går vidare till slutspel (topp 2 + bästa 3:orna)</div>
    <div class="legend"><span class="played-mark">✓</span> Faktiskt resultat (spelad match) &nbsp;·&nbsp; matcher utan bock är modellens mest sannolika utfall</div>
    <div class="group-grid" id="groups"></div>
  </div>
  <div class="tournament-section">
    <h2>Slutspel</h2>
    <div class="ko-grid" id="knockout"></div>
  </div>
  <div class="champion-banner">
    <div class="label">Predikterad världsmästare 2026</div>
    <div class="name" id="champion"></div>
  </div>
</div>

<div class="panel" id="panel-bracket">
  <div class="legend"><span class="played-mark">✓</span> Spelad match (faktiskt resultat) &nbsp;·&nbsp; matcher utan bock visar modellens mest sannolika lag och resultat</div>
  <div class="legend">Procenttalen är sannolikheten att laget går vidare (vinst i ordinarie + halva oavgjort, givet att lagen möts). Siffran längst till höger är troligast antal mål.</div>
  <div class="bracket-scroll"><div class="bracket" id="bracket"></div></div>
</div>

<div class="panel" id="panel-team">
  <div class="controls">
    <div class="control">
      <label>Lag</label>
      <select id="team-pick"></select>
    </div>
  </div>
  <div id="team-other-warning" class="other-note" style="display:none;"></div>

  <div class="summary" id="team-stats"></div>

  <div class="card">
    <h3>Sannolikhet att nå varje steg (%)</h3>
    <div id="team-chart" style="height: 420px;"></div>
  </div>
  <div class="legend" id="team-sim-note" style="margin-top: 12px;"></div>
</div>

<div class="panel" id="panel-model">
  <div class="prose">
    <div class="card">
      <h2>Översikt</h2>
      <p>Prediktionerna bygger på en <strong>Dixon-Coles Poisson-modell</strong> som anpassas på historiska landskampsresultat, kombinerad med <strong>Monte Carlo-simulering</strong> av hela turneringen. Modellen lär sig en attack- och försvarsstyrka för varje lag, översätter det till en sannolikhetsfördelning för varje tänkbart matchresultat, och simulerar sedan VM tiotusentals gånger för att uppskatta hur långt varje lag når.</p>
    </div>

    <div class="card">
      <h2>1. Datat och viktningen</h2>
      <p>Träningsmängden är samtliga officiella landskamper sedan 1872. Eftersom gamla matcher säger mindre om dagens form viktas varje match ner ju äldre den är, med en <strong>halveringstid på 2 år</strong> &mdash; en match som är två år gammal räknas hälften så mycket som en helt färsk.</p>
      <ul>
        <li><strong>Vänskapsmatcher</strong> väger bara hälften så mycket som tävlingsmatcher, eftersom de säger mindre om verklig styrka.</li>
        <li>Lag med <strong>färre än 15 matcher de senaste 8 åren</strong> som inte spelar VM klumpas ihop till ett gemensamt lag <span class="param">Other</span>, så att modellen inte tvingas skatta instabila parametrar för lag den knappt sett.</li>
      </ul>
    </div>

    <div class="card">
      <h2>2. Dixon-Coles Poisson-modellen</h2>
      <p>Varje lag får två tal: en <strong>attackstyrka</strong> $\alpha$ (hur många mål det tenderar att göra) och en <strong>försvarsstyrka</strong> $\delta$ (hur svårt det är att göra mål mot det). Antalet mål ett lag gör i en match antas vara Poisson-fördelat, och det förväntade antalet mål för hemmalaget ($\lambda$) respektive bortalaget ($\mu$) ges av:</p>
      <div class="formula">$$\log \lambda_{\text{home}} = \alpha_{\text{home}} - \delta_{\text{away}} + \gamma \, \mathbb{1}_{\text{home}} \qquad\qquad \log \mu_{\text{away}} = \alpha_{\text{away}} - \delta_{\text{home}}$$</div>
      <p>Sannolikheten att ett lag gör exakt $k$ mål följer då Poisson-fördelningen $P(k) = \lambda^{k} e^{-\lambda} / k!$. Två globala parametrar finjusterar bilden:</p>
      <ul>
        <li><strong>Hemmaplansfördel</strong> <span class="param">$\gamma$ = <span id="m-gamma"></span></span> &mdash; spelar laget på hemmaplan får det ungefär <span id="m-gamma-pct"></span>&nbsp;% fler förväntade mål. I VM:s gruppspel räknas alla matcher som neutral plan ($\mathbb{1}_{\text{home}} = 0$), så fördelen används bara i matchprediktorn när du stänger av &quot;Neutral plan&quot;.</li>
        <li><strong>Dixon-Coles-korrigeringen</strong> <span class="param">$\rho$ = <span id="m-rho"></span></span> &mdash; en ren Poisson-modell underskattar hur ofta lågmålsresultat (0&ndash;0, 1&ndash;0, 0&ndash;1, 1&ndash;1) inträffar och hur korrelerade lagens mål är. $\rho$ justerar just de fyra resultaten så att modellen stämmer bättre med verkligheten.</li>
      </ul>
      <p>Sannolikheten för ett helt matchresultat $(x, y)$ blir då produkten av lagens Poisson-sannolikheter gånger korrigeringsfaktorn $\tau_{\rho}$:</p>
      <div class="formula">$$P(x, y) = \tau_{\rho}(x, y) \cdot \frac{\lambda^{x} e^{-\lambda}}{x!} \cdot \frac{\mu^{y} e^{-\mu}}{y!} \qquad\quad \tau_{\rho}(x, y) = \begin{cases} 1 - \lambda\mu\rho & (x,y) = (0,0) \\ 1 + \lambda\rho & (x,y) = (0,1) \\ 1 + \mu\rho & (x,y) = (1,0) \\ 1 - \rho & (x,y) = (1,1) \\ 1 & \text{annars} \end{cases}$$</div>
      <p>Alla parametrar anpassas samtidigt genom att <strong>maximera den (viktade) likelihooden</strong> för de observerade resultaten, med L-BFGS-B-optimering. För att modellen ska vara entydig låses summan av alla attackstyrkor till noll: $\sum_i \alpha_i = 0$.</p>
    </div>

    <div class="card">
      <h2>3. Från modell till matchprediktion</h2>
      <p>För en given match räknar modellen ut $\lambda$ och $\mu$ och bygger en <strong>11&times;11-matris</strong> $P(x, y)$ med sannolikheten för varje resultat från 0&ndash;0 upp till 10&ndash;10. Det är den matrisen du ser som värmekarta i &quot;Matchprediktor&quot;. Ur den fås direkt sannolikheten för vinst, oavgjort och förlust.</p>
      <p>För <strong>Svenska Spels VM-tipset</strong> ger ett tips poäng enligt: <span class="param">2 p</span> för rätt antal mål av hemmalaget, <span class="param">2 p</span> för rätt antal mål av bortalaget och <span class="param">3 p</span> för rätt 1X2-tecken. Dashboarden provar alla tänkbara tips $(a, b)$ och väljer det som <strong>maximerar den förväntade poängen</strong> &mdash; vilket inte alltid är det mest sannolika enskilda resultatet:</p>
      <div class="formula">$$(a, b)^{\star} = \underset{(a,\, b)}{\arg\max} \; \sum_{x,\, y} P(x, y) \cdot \text{points}(a, b, x, y)$$</div>
    </div>

    <div class="card">
      <h2>4. Turneringssimulering (Monte Carlo)</h2>
      <p>För att uppskatta hur långt ett lag når simuleras hela turneringen <strong><span id="m-nsims"></span> gånger</strong>. I varje simulering:</p>
      <ul>
        <li>Spelas alla gruppmatcher genom att <strong>slumpa ett resultat</strong> ur respektive matchmatris.</li>
        <li>Rankas lagen i gruppen på poäng &rarr; målskillnad &rarr; gjorda mål &rarr; slump, och de <strong>8 bästa grupptreorna</strong> väljs ut och placeras i slutspelsträdet.</li>
        <li>Spelas slutspelet match för match. Slutar en slutspelsmatch oavgjort avgörs den med <strong>slantsingling</strong> &mdash; det motsvarar att straffsparksläggning är ungefär ett myntkast.</li>
      </ul>
      <p>Genom att räkna hur ofta varje lag når varje steg fås sannolikheterna i &quot;Lagets chanser&quot;. Fliken &quot;Mest troliga turnering&quot; visar i stället ett enda utfall, där varje match alltid sätts till sitt mest sannolika resultat.</p>
    </div>

    <div class="card">
      <h2>5. Vad modellen inte vet</h2>
      <ul>
        <li>Den känner bara till <strong>resultat</strong> &mdash; inte skador, avstängningar, dagsform, taktik eller vilka spelare som faktiskt står på planen.</li>
        <li>Lag som klumpats som <span class="param">Other</span> delar parametrar och får därför grova skattningar.</li>
        <li>Straffsparksläggningar modelleras som rena myntkast.</li>
        <li>Sannolikheterna beskriver modellens världsbild, inte sanningen &mdash; fotboll är till sin natur oförutsägbart.</li>
      </ul>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'match' || btn.dataset.tab === 'team') window.dispatchEvent(new Event('resize'));
    if (btn.dataset.tab === 'bracket') requestAnimationFrame(drawBracketConnectors);
  });
});

window.addEventListener('resize', () => {
  if (document.getElementById('panel-bracket').classList.contains('active')) drawBracketConnectors();
});

function buildDropdowns() {
  const selA = document.getElementById('team-a');
  const selB = document.getElementById('team-b');
  for (const [g, teams] of Object.entries(DATA.teamsByGroup)) {
    const ogA = document.createElement('optgroup'); ogA.label = g;
    const ogB = document.createElement('optgroup'); ogB.label = g;
    teams.forEach(t => {
      const oA = document.createElement('option'); oA.value = t; oA.textContent = t; ogA.appendChild(oA);
      const oB = document.createElement('option'); oB.value = t; oB.textContent = t; ogB.appendChild(oB);
    });
    selA.appendChild(ogA);
    selB.appendChild(ogB);
  }
  selA.value = 'Sweden' in DATA.teamParams ? 'Sweden' : 'Argentina';
  selB.value = 'Brazil';
}

function poissonPmf(k, lam) {
  let p = Math.exp(-lam);
  for (let i = 1; i <= k; i++) p *= lam / i;
  return p;
}

function scoreMatrix(homeName, awayName, neutral) {
  const home = DATA.teamParams[homeName];
  const away = DATA.teamParams[awayName];
  const hf = neutral ? 0.0 : 1.0;
  const lam = Math.exp(home.attack - away.defense + DATA.gamma * hf);
  const mu = Math.exp(away.attack - home.defense);
  const N = 11;
  const ph = new Array(N), pa = new Array(N);
  for (let i = 0; i < N; i++) { ph[i] = poissonPmf(i, lam); pa[i] = poissonPmf(i, mu); }
  const p = Array.from({length: N}, (_, i) => pa.map(v => ph[i] * v));
  p[0][0] *= 1 - lam * mu * DATA.rho;
  p[0][1] *= 1 + lam * DATA.rho;
  p[1][0] *= 1 + mu * DATA.rho;
  p[1][1] *= 1 - DATA.rho;
  let s = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) s += p[i][j];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) p[i][j] /= s;
  return { matrix: p, lam, mu };
}

function bestTip(p) {
  const N = p.length;
  let bestEv = -1, bestA = 0, bestB = 0;
  for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
    let ev = 0;
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
      let s = 0;
      if (x === a) s += 2;
      if (y === b) s += 2;
      if (Math.sign(a - b) === Math.sign(x - y)) s += 3;
      ev += p[x][y] * s;
    }
    if (ev > bestEv) { bestEv = ev; bestA = a; bestB = b; }
  }
  return { a: bestA, b: bestB, ev: bestEv };
}

function updateMatch() {
  const a = document.getElementById('team-a').value;
  const b = document.getElementById('team-b').value;
  const neutral = document.getElementById('neutral').checked;
  const { matrix, lam, mu } = scoreMatrix(a, b, neutral);

  let p1 = 0, px = 0, p2 = 0;
  let mode = [0, 0], modeP = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      if (i > j) p1 += matrix[i][j];
      else if (i === j) px += matrix[i][j];
      else p2 += matrix[i][j];
      if (matrix[i][j] > modeP) { modeP = matrix[i][j]; mode = [i, j]; }
    }
  }
  const fmtPct = v => (v * 100).toFixed(1) + '%';
  document.getElementById('p1').textContent = fmtPct(p1);
  document.getElementById('px').textContent = fmtPct(px);
  document.getElementById('p2').textContent = fmtPct(p2);
  document.getElementById('lab-1').textContent = 'Vinst ' + a;
  document.getElementById('lab-2').textContent = 'Vinst ' + b;
  document.getElementById('ev-a').textContent = lam.toFixed(2);
  document.getElementById('ev-b').textContent = mu.toFixed(2);
  document.getElementById('mode').textContent = mode[0] + ' – ' + mode[1];
  document.getElementById('mode-p').textContent = fmtPct(modeP);
  const tip = bestTip(matrix);
  document.getElementById('tip').textContent = tip.a + ' – ' + tip.b;
  document.getElementById('tip-ev').textContent = tip.ev.toFixed(2);

  const warn = document.getElementById('other-warning');
  const others = [];
  if (DATA.teamParams[a].isOther) others.push(a);
  if (DATA.teamParams[b].isOther) others.push(b);
  if (others.length) {
    warn.textContent = 'Obs: ' + others.join(' och ') + ' saknar tillräcklig matchhistorik och använder gemensamma "Other"-parametrar.';
    warn.style.display = 'block';
  } else {
    warn.style.display = 'none';
  }

  const M = 7;
  const z = [], text = [];
  for (let i = 0; i < M; i++) {
    const row = [], tr = [];
    for (let j = 0; j < M; j++) { row.push(matrix[i][j] * 100); tr.push((matrix[i][j] * 100).toFixed(1)); }
    z.push(row); text.push(tr);
  }
  Plotly.react('heatmap', [{
    z, text, type: 'heatmap',
    x: Array.from({length: M}, (_, i) => i),
    y: Array.from({length: M}, (_, i) => i),
    colorscale: [[0, '#faf6ef'], [0.35, '#c9d3df'], [0.7, '#3e5d83'], [1, '#0a2540']],
    texttemplate: '%{text}',
    hovertemplate: a + ' %{y} – %{x} ' + b + '<br>%{z:.2f}%<extra></extra>',
    showscale: false
  }], {
    xaxis: { title: b + ' mål', dtick: 1, color: '#1a1f2e', fixedrange: true },
    yaxis: { title: a + ' mål', dtick: 1, color: '#1a1f2e', fixedrange: true },
    margin: { t: 10, l: 60, r: 20, b: 50 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#1a1f2e' }
  }, { displayModeBar: false, responsive: true });
}

function renderTournament() {
  const t = DATA.tournament;
  const byGroup = {};
  t.groupMatches.forEach(m => {
    byGroup[m.group] = byGroup[m.group] || { matches: [], standings: [] };
    byGroup[m.group].matches.push(m);
  });
  t.standings.forEach(s => {
    if (byGroup[s.group]) byGroup[s.group].standings.push(s);
  });

  const grid = document.getElementById('groups');
  for (const [g, data] of Object.entries(byGroup)) {
    const card = document.createElement('div');
    card.className = 'card';
    let html = '<h3>' + g + '</h3>';
    html += '<table><thead><tr><th>#</th><th>Lag</th><th class="num">P</th><th class="num">GM</th><th class="num">IM</th><th class="num">MS</th></tr></thead><tbody>';
    data.standings.forEach(s => {
      const cls = s.advanced ? 'qualifier' : '';
      html += '<tr class="' + cls + '"><td>' + s.position + '</td><td>' + s.team + '</td>';
      html += '<td class="num">' + s.points + '</td><td class="num">' + s.gf + '</td>';
      html += '<td class="num">' + s.ga + '</td><td class="num">' + (s.gd > 0 ? '+' : '') + s.gd + '</td></tr>';
    });
    html += '</tbody></table>';
    html += '<h3 style="margin-top:14px;">Matcher</h3><table>';
    data.matches.forEach(m => {
      const cls = m.played ? 'played' : 'predicted';
      const mark = m.played ? ' <span class="played-mark" title="Spelad match">✓</span>' : '';
      html += '<tr><td>' + m.home_team + '</td>';
      html += '<td class="center"><span class="ms ' + cls + '">' + m.home_goals + ' – ' + m.away_goals + '</span>' + mark + '</td>';
      html += '<td style="text-align:right;">' + m.away_team + '</td></tr>';
    });
    html += '</table>';
    card.innerHTML = html;
    grid.appendChild(card);
  }

  const rounds = {};
  t.knockout.forEach(m => { (rounds[m.round] = rounds[m.round] || []).push(m); });
  const ko = document.getElementById('knockout');
  const order = ['Round of 32', 'Round of 16', 'Quarter-final', 'Semi-final', 'Final'];
  order.forEach(r => {
    if (!rounds[r]) return;
    const card = document.createElement('div');
    card.className = 'card';
    let html = '<h3>' + r + '</h3>';
    rounds[r].forEach(m => {
      const t1Win = m.winner === m.team1;
      html += '<div class="ko-row">';
      html += '<span class="t1 ' + (t1Win ? 'winner' : 'loser') + '">' + m.team1 + '</span>';
      html += '<span class="score">' + m.goals1 + ' – ' + m.goals2 + '</span>';
      html += '<span class="' + (!t1Win ? 'winner' : 'loser') + '">' + m.team2 + '</span>';
      html += '</div>';
    });
    card.innerHTML = html;
    ko.appendChild(card);
  });

  document.getElementById('champion').textContent = t.champion;
}

const ROUND_SV = {
  'Round of 32': 'Sextondelsfinal',
  'Round of 16': 'Åttondelsfinal',
  'Quarter-final': 'Kvartsfinal',
  'Semi-final': 'Semifinal',
  'Final': 'Final',
  'Match for third place': 'Bronsmatch',
};

function bracketMatch(m, isFinal) {
  const w1 = m.winner === m.team1;
  const pct = v => Math.round(v * 100) + '%';
  const id = isFinal ? 'final' : m.match;
  const team = (name, prob, goal, win) =>
    '<div class="bteam ' + (win ? 'bteam-win' : 'bteam-lose') + '">' +
      '<span class="bt-name">' + name + '</span>' +
      '<span class="bt-prob">' + pct(prob) + '</span>' +
      '<span class="bt-goal">' + goal + '</span></div>';
  const mark = m.played ? '<span class="bt-mark" title="Spelad match">✓</span>' : '';
  return '<div class="bmatch' + (m.played ? ' bmatch-played' : '') + '" data-num="' + id + '">' +
    team(m.team1, m.p1, m.goals1, w1) +
    team(m.team2, m.p2, m.goals2, !w1) + mark + '</div>';
}

function renderBracket() {
  const t = DATA.tournament;
  const byNum = {};
  let finalRow = null, thirdRow = null;
  t.knockout.forEach(m => {
    if (m.match != null) byNum[m.match] = m;
    if (m.round === 'Final') finalRow = m;
    if (m.round === 'Match for third place') thirdRow = m;
  });
  const rounds = ['Round of 32', 'Round of 16', 'Quarter-final', 'Semi-final'];
  const col = (nums, roundName) =>
    '<div class="bcol"><div class="bcol-head">' + ROUND_SV[roundName] + '</div><div class="bcol-body">' +
    nums.map(n => bracketMatch(byNum[n], false)).join('') + '</div></div>';

  const leftHtml = DATA.bracket.left.map((nums, i) => col(nums, rounds[i])).join('');
  // Höger sida har samma kolumnordning (R32→SF); CSS row-reverse vänder den visuellt
  const rightHtml = DATA.bracket.right.map((nums, i) => col(nums, rounds[i])).join('');

  const center =
    '<div class="bcenter"><div class="bcol-head">Final</div>' +
    '<div class="bfinal">' + bracketMatch(finalRow, true) + '</div>' +
    '<div class="bchamp"><div class="label">Världsmästare</div><div class="name">' + finalRow.winner + '</div></div>' +
    (thirdRow ? '<div class="bthird"><div class="bcol-head">' + ROUND_SV['Match for third place'] +
      '</div>' + bracketMatch(thirdRow, false) + '</div>' : '') +
    '</div>';

  document.getElementById('bracket').innerHTML =
    '<div class="bside left">' + leftHtml + '</div>' + center +
    '<div class="bside right">' + rightHtml + '</div>';
}

// Rita förbindelselinjer från varje vinnarruta till nästa match (SVG-overlay).
// Körs när fliken visas/storlek ändras eftersom mått saknas medan panelen är dold.
function drawBracketConnectors() {
  const bracket = document.getElementById('bracket');
  if (!bracket || bracket.offsetParent === null) return;
  let svg = bracket.querySelector('svg.bconn');
  if (!svg) {
    svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'bconn');
    bracket.insertBefore(svg, bracket.firstChild);
  }
  const W = bracket.scrollWidth, H = bracket.scrollHeight;
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.innerHTML = '';
  const brect = bracket.getBoundingClientRect();
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#b08d57';

  const matchEl = num => bracket.querySelector('.bmatch[data-num="' + num + '"]');
  const isRight = num => !!matchEl(num).closest('.bside.right');
  // Vinnarrutans mittpunkt på ytterkanten (mot nästa match)
  const winPoint = (num, outerRight) => {
    const win = matchEl(num).querySelector('.bteam-win') || matchEl(num).querySelector('.bteam');
    const r = win.getBoundingClientRect();
    return { x: (outerRight ? r.right : r.left) - brect.left, y: r.top + r.height / 2 - brect.top };
  };
  // Förälderns inkommande punkt (kanten mot sina feeders), i höjd med hela matchen
  const edgePoint = (num, right) => {
    const r = matchEl(num).getBoundingClientRect();
    return { x: (right ? r.right : r.left) - brect.left, y: r.top + r.height / 2 - brect.top };
  };
  const elbow = (from, to) => {
    const midX = (from.x + to.x) / 2;
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', 'M ' + from.x + ' ' + from.y + ' H ' + midX + ' V ' + to.y + ' H ' + to.x);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', accent);
    p.setAttribute('stroke-width', '1.6');
    svg.appendChild(p);
  };

  // Numrerade föräldrar (R16/QF/SF): båda feeders på samma sida
  for (const [numStr, fs] of Object.entries(DATA.bracket.feeders)) {
    const num = +numStr, right = isRight(num);
    const pp = edgePoint(num, right);           // inkant = förälderns sida mot feeders
    fs.forEach(f => elbow(winPoint(f, !right), pp));
  }
  // Final: vänster och höger semifinal in mot mitten
  const [sfL, sfR] = DATA.bracket.finalFeeders;
  elbow(winPoint(sfL, true), edgePoint('final', false));
  elbow(winPoint(sfR, false), edgePoint('final', true));
}

const TEAM_PROB_BY_NAME = Object.fromEntries(DATA.teamProbabilities.map(r => [r.team, r]));
const STAGES = [
  { key: 'p_knockout', label: 'Slutspel' },
  { key: 'p_r16', label: 'Åttondelsfinal' },
  { key: 'p_qf', label: 'Kvartsfinal' },
  { key: 'p_sf', label: 'Semifinal' },
  { key: 'p_final', label: 'Final' },
  { key: 'p_win', label: 'Världsmästare' },
];

function buildTeamDropdown() {
  const sel = document.getElementById('team-pick');
  for (const [g, teams] of Object.entries(DATA.teamsByGroup)) {
    const og = document.createElement('optgroup'); og.label = g;
    teams.forEach(t => {
      const o = document.createElement('option'); o.value = t; o.textContent = t; og.appendChild(o);
    });
    sel.appendChild(og);
  }
  sel.value = 'Sweden' in DATA.teamParams ? 'Sweden' : 'Brazil';
}

function updateTeam() {
  const t = document.getElementById('team-pick').value;
  const probs = TEAM_PROB_BY_NAME[t];
  const fmtPct = v => (v * 100).toFixed(2) + '%';

  const stats = document.getElementById('team-stats');
  stats.innerHTML = '';
  stats.style.gridTemplateColumns = 'repeat(3, 1fr)';
  [
    { label: 'Slutspel', v: probs.p_knockout },
    { label: 'Semifinal', v: probs.p_sf },
    { label: 'Världsmästare', v: probs.p_win },
  ].forEach(s => {
    const div = document.createElement('div');
    div.className = 'stat';
    div.innerHTML = '<div class="value">' + fmtPct(s.v) + '</div><div class="label">' + s.label + '</div>';
    stats.appendChild(div);
  });

  const warn = document.getElementById('team-other-warning');
  if (DATA.teamParams[t].isOther) {
    warn.textContent = 'Obs: ' + t + ' saknar tillräcklig matchhistorik och använder gemensamma "Other"-parametrar.';
    warn.style.display = 'block';
  } else {
    warn.style.display = 'none';
  }

  const ys = STAGES.map(s => probs[s.key] * 100);
  const labels = STAGES.map(s => s.label);
  const text = ys.map(v => v.toFixed(2) + '%');
  Plotly.react('team-chart', [{
    x: labels, y: ys, text, type: 'bar',
    marker: { color: '#0a2540' },
    textposition: 'outside',
    hovertemplate: '%{x}: %{y:.2f}%<extra></extra>'
  }], {
    yaxis: { title: 'Sannolikhet (%)', range: [0, Math.max(100, Math.max(...ys) * 1.15)], color: '#1a1f2e' },
    xaxis: { color: '#1a1f2e' },
    margin: { t: 20, l: 60, r: 20, b: 60 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#1a1f2e' }
  }, { displayModeBar: false, responsive: true });

  const playedCount = DATA.tournament.groupMatches.filter(m => m.played).length;
  document.getElementById('team-sim-note').textContent =
    'Baserat på ' + DATA.nSims.toLocaleString('sv-SE') + ' Monte Carlo-simulerade turneringar, där de ' +
    playedCount + ' redan spelade gruppmatcherna är låsta till sina faktiska resultat.';
}

function renderModel() {
  document.getElementById('m-gamma').textContent = DATA.gamma.toFixed(3);
  document.getElementById('m-gamma-pct').textContent = ((Math.exp(DATA.gamma) - 1) * 100).toFixed(0);
  document.getElementById('m-rho').textContent = DATA.rho.toFixed(3);
  document.getElementById('m-nsims').textContent = DATA.nSims.toLocaleString('sv-SE');
}

buildDropdowns();
document.getElementById('team-a').addEventListener('change', updateMatch);
document.getElementById('team-b').addEventListener('change', updateMatch);
document.getElementById('neutral').addEventListener('change', updateMatch);
renderTournament();
renderBracket();
requestAnimationFrame(drawBracketConnectors);
updateMatch();
buildTeamDropdown();
document.getElementById('team-pick').addEventListener('change', updateTeam);
updateTeam();
renderModel();
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(dashboard_data))

with open("output/dashboard.html", "w") as f:
    f.write(html)

print(f"Dashboard skapad: output/dashboard.html")
print(f"  Antal lag i dropdowns: {len(team_params)}")
print(f"  Predikterad världsmästare (mest troliga path): {champion}")
