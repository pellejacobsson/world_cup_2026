# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projektets syfte

Förutsäga utfall i Fotbolls-VM 2026 (gruppspel + slutspel) genom att anpassa en Dixon-Coles Poisson-modell på historiska landskampsresultat och simulera turneringen via Monte Carlo. Modellen används bl.a. för Svenska Spels VM-tipset (`vmtipset_score` i `utils.py:171`).

## Arkitektur

Hela pipelinen lever som rena funktioner i `utils.py` och drivs interaktivt från marimo-notebooken `wc_2026_predict.py`. Det finns ingen CLI; `main.py` är en oanvänd platshållare. Notebooken är cell-uppbyggd och varje cell anropar en funktion i `utils.py` — håll den uppdelningen.

Flödet är:
1. **Ladda data** (`load_data`): `data/results.csv` (alla landskamper sedan 1872) + `data/worldcup.json` (VM-spelplanen med grupper, slutspelsträd och platshållarkoder som `W49`, `3A/B/F`).
2. **Bygg träningsmängd** (`build_model_df`): viktar matcher med exponentiellt avtagande halveringstid (`half_life`, default 2 år från `ref`), vänskapsmatcher får halverad vikt.
3. **Grupperingsval** (`select_common_teams` + `apply_team_grouping`): lag med <15 matcher senaste 8 åren som inte spelar VM klumpas ihop som `"Other"` för att inte få instabila parametrar.
4. **Modellanpassning** (`fit_model`): L-BFGS-B-optimering av Dixon-Coles negativ log-likelihood (`neg_loglik`). Parametrar: per-lag attack/defense, global hemmaplansfördel `gamma`, Dixon-Coles low-score-korrigering `rho`. Sista attackparametern är låst så att Σattack=0 för identifierbarhet.
5. **Prediktion** (`score_matrix`, `predict`, `predict_all`): bygger 11×11-resultatfördelning per match. `best_tip` hittar tipset som maximerar förväntad VM-tipsetpoäng (`vmtipset_score`: 2p per rätt mål, 3p per rätt 1X2).
6. **Monte Carlo** (`monte_carlo` → `simulate_tournament`): simulerar gruppspel, rankar lag (poäng → målskillnad → gjorda mål → slump), väljer de 8 bästa treorna och placerar dem i slutspelsslots via backtracking (`match_thirds`). Slutspelsmatcher som slutar oavgjort avgörs med slantsingling (motsvarar straffar). Returnerar sannolikheter för varje avancemangssteg per lag.

Slutspelsplatshållare i `worldcup.json` använder en egen kodning som `resolve()` i `simulate_tournament` tolkar: `W49`/`L49` = vinnare/förlorare av match 49, `1A`/`2B` = etta/tvåa i grupp A/B, `3A/B/F` = en av åtta bästa treorna från angivna grupper.

## Kommandon

Kör allt via `uv` (Python ≥3.13):

```bash
uv sync                                   # installera enligt lockfile
uv add <package>                          # lägg till beroende
uv run marimo edit wc_2026_predict.py     # öppna notebooken i webbläsaren
uv run marimo run wc_2026_predict.py      # kör notebooken som app (read-only)
uv run python -c "from utils import ..."  # testa enskilda funktioner
```

Det finns ingen testsvit, linter eller CI konfigurerad.

## Kodstil

**Simplicity is the #1 priority.** Skriv den kortaste, mest direkta koden som fungerar.

- **Inga thin wrappers, inga onödiga abstraktioner, ingen try/except.** Låt koden krascha — det avslöjar dataproblem direkt. Skapa inte hjälpfunktioner, utility-klasser, factory-mönster eller config-objekt om inte logiken återanvänds 3+ gånger redan nu.
- **Ingen over-engineering.** Inga basklasser, protocols, ABC, generiska typparametrar, plugin-system, registries eller callback-mönster.
- **Ingen generalitet.** Skriv kod för det exakta, konkreta användningsfallet. Hardkoda värden när det bara finns ett användningsfall. Konvertera inte typer tyst — låt det krascha.
- **Polars för dataframes, Plotly (express först) för grafer.** Anta senaste stabila versioner.
- **Type hints i `.py`-script, inga type hints i marimo-celler.**
- **Kommentarer på svenska, kod (variabler, funktioner, strängar) på engelska.** Skriv bara kommentarer som förklarar *varför*, aldrig *vad*. Inga docstrings på små eller uppenbara funktioner.
- **Dubbla citationstecken** för stränglitteraler.
- **Commit-meddelanden på svenska.**

### Anti-mönster — gör INTE så här:
```python
# DÅLIGT: thin wrapper som inte tillför något
def load_data(path):
    return pl.read_parquet(path)

# DÅLIGT: onödigt config-objekt
@dataclass
class TrainConfig:
    n_trials: int = 100
    ...

# DÅLIGT: onödigt abstraktionslager
class BaseTrainer:
    def train(self): ...
class LGBMTrainer(BaseTrainer): ...

# DÅLIGT: generisk funktion för ett användningsfall
def run_experiment(model_factory, vectorizer_factory, ...):
    ...
```

### Rätt stil:
```python
# BRA: direkt, konkret, inga wrappers
df = pl.read_parquet(path)

# BRA: enkel funktion, hardkodad för det enda användningsfallet
def train(df, run_name, n_trials=100):
    ...
```

## Beroendehantering med uv

- **Kör alltid Python via uv** — använd inte system-Python.
- `uv add <package>` för att lägga till beroenden.
- `uv sync` för att installera exakt det som lockfilen anger (kör efter pull som ändrar `pyproject.toml`/lockfile).
- Commita `pyproject.toml` och lockfilen tillsammans.
