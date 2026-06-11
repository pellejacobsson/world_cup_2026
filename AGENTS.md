# Coding standards for Python scripts and Python notebooks in this repository.

## Style — READ THIS CAREFULLY
- **Simplicity is the #1 priority.** Write the shortest, most direct code that works. Do NOT plan for future features or extensibility.
- **NO thin wrapper functions.** If a function only calls one other function, or adds no logic beyond forwarding arguments, do NOT create it. Call the underlying function directly.
- **NO unnecessary abstractions.** Do not create helper functions, utility classes, factory patterns, or configuration objects unless the logic is reused 3+ times in the codebase right now.
- **NO over-engineering.** Do not add: base classes, protocols, ABC, generic type parameters, plugin systems, registries, or callback patterns.
- **Conciseness:** Use type hinting in scripts, but not in notebooks.
- **Error Handling:** Do NOT use try/catch. Let the code crash. Crashes expose data issues early.
- **Strings:** Use double quotes for string literals.
- **NO generality.** Write code for the exact, concrete use case. Hardcode values when there is only one use case. Do NOT accept general inputs "just in case". Do NOT silently convert types — let it fail.

### Anti-patterns — do NOT do these:
```python
# BAD: thin wrapper that adds nothing
def load_data(path):
    return pl.read_parquet(path)

# BAD: unnecessary config object
@dataclass
class TrainConfig:
    n_trials: int = 100
    ...

# BAD: unnecessary abstraction layer
class BaseTrainer:
    def train(self): ...
class LGBMTrainer(BaseTrainer): ...

# BAD: generic function for one use case
def run_experiment(model_factory, vectorizer_factory, ...):
    ...
```

### Correct style:
```python
# GOOD: direct, concrete, no wrappers
df = pl.read_parquet(path)

# GOOD: plain function, hardcoded for the one use case
def train(df, run_name, n_trials=100):
    ...
```

## Libraries and versions
- **Dataframes:** Use Polars.
- **Visualization:** Use Plotly. First express, if not possible, use graph_objects.
- **Versions:** Assume the latest stable versions of all libraries.

## Documentation and Localization
- **Comments:** Minimal. Only explain "why", never "what". Do NOT add docstrings to small or obvious functions.
- **Language:** Write all comments and documentation in Swedish.
- **Code:** Keep all variable names, function names, and string literals in English.

## Running code with uv
- **Always run Python via uv:** Do not use the system Python directly.
- Examples:
  - Run a script: `uv run python path/to/script.py`

## Dependency & packaging management with uv
- **Add dependencies:** Use `uv add <package>`
  - Example: `uv add polars plotly`
- **Sync environment:** Use `uv sync` to install exactly what the lockfile specifies.
  - Use `uv sync` after pulling changes that modify `pyproject.toml` / lockfile.
- **Preferred workflow:**
  1. `uv add ...` to modify dependencies
  2. Commit `pyproject.toml` and the lockfile
  3. Teammates run `uv sync`
- **Assume latest stable versions** unless the project pins a version explicitly.

## Version control
- **Commit messages:** Use clear, descriptive commit messages in Swedish.
