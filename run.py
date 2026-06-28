import json
from datetime import date

import numpy as np

from utils import (
    apply_team_grouping,
    build_model_df,
    fit_model,
    load_data,
    monte_carlo,
    predict_all,
    select_common_teams,
)

name_map = {"Bosnia and Herzegovina": "Bosnia & Herzegovina", "United States": "USA"}
res, wc = load_data("data/results.csv", "data/worldcup.json", name_map)

ref = date(2026, 6, 11)
half_life = 365 * 2

df_model = build_model_df(res, ref, half_life)
common, win = select_common_teams(df_model, wc, ref)
df_fit = apply_team_grouping(win, common)
model = fit_model(df_fit)

df_pred = predict_all(wc, model)
df_pred.write_parquet("output/predictions.parquet")

n_sims = 100_000
df_sim, top_finals, top_matchups = monte_carlo(wc, model, n_sims=n_sims, seed=0)
df_sim.write_parquet("output/simulation.parquet")

with open("output/top_finals.json", "w") as f:
    json.dump(
        [
            {
                "team1": t[0],
                "team2": t[1],
                "goals1": int(t[2]),
                "goals2": int(t[3]),
                "count": int(n),
                "share": n / n_sims,
            }
            for t, n in top_finals
        ],
        f,
        indent=2,
    )

with open("output/top_matchups.json", "w") as f:
    json.dump(
        [
            {"teams": list(t), "count": int(n), "share": n / n_sims}
            for t, n in top_matchups
        ],
        f,
        indent=2,
    )

# Spara modellparametrarna så att de kan återanvändas utan ny anpassning
np.savez(
    "output/model.npz",
    teams=np.array(model["teams"]),
    attack=model["attack"],
    defense=model["defense"],
    gamma=model["gamma"],
    rho=model["rho"],
)

print(f"Antal simuleringar: {n_sims}")
print(f"Antal lag i modell: {len(model['teams'])}")
print(f"gamma (hemmaplan): {model['gamma']:.4f}")
print(f"rho (Dixon-Coles): {model['rho']:.4f}")
print()
print("Topp 10 vinstchans:")
print(df_sim.head(10))
print()
print("Topp 5 finallag:")
for t, n in top_matchups[:5]:
    print(f"  {t[0]} - {t[1]}: {n / n_sims:.2%}")
