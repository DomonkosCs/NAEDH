# Companion code for "An integration-free approach for particle flow filtering"

The implementation of the **N-step Analytic EDH (NAEDH)**
particle flow filter and the five baselines it is compared against. Reproduces
the main RMSE / runtime table and the three-panel figure of the manuscript.
The paper can be accessed at https://arxiv.org/abs/2605.14852.

## Layout

| File                 | Purpose                                                                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `problem.py`         | The example problem (dynamics, bearings-only measurement model, sine-y deterministic truth). One `Problem` dataclass is the single source of truth for every filter. |
| `naedh_d.py`         | **The contribution.** Pure math primitive `flow(...)` + uniform filter interface.                                                                                    |
| `edh.py`             | Daum-Huang Euler EDH, fixed and ΔL-adaptive.                                                                                                                         |
| `ekf.py`             | Extended Kalman Filter.                                                                                                                                              |
| `bootstrap_pf.py`    | Bootstrap PF with ESS-adaptive systematic resampling.                                                                                                                |
| `gromov.py`          | Stochastic Gromov flow, ΔL-adaptive.                                                                                                                                 |
| `results.py`         | Frozen dataclasses pickled into `results.pkl`.                                                                                                                       |
| `run_experiments.py` | MC sweep across σ_θ → `results.pkl`.                                                                                                                                 |
| `make_table.py`      | `results.pkl` → `table.tex`.                                                                                                                                         |
| `make_figure.py`     | `results.pkl` → `three_panel_figure.{png,pdf}`.                                                                                                                      |

Every filter file exposes the same three callables — `init(rng, problem, x0_hat, P0, **hp)`,
`predict(state, problem, rng)`, `update(state, problem, z, rng, **hp)`. The MC
runner in `run_experiments.py` consumes them uniformly; it has no knowledge
of what kind of filter is running. Adding a new filter is one file + one
tuple in the `FILTERS` list.

## Running

```bash
python NAEDH_final/run_experiments.py     # writes results.pkl (~minutes)
python NAEDH_final/make_table.py          # writes table.tex
python NAEDH_final/make_figure.py         # writes three_panel_figure.{png,pdf}
```

Dependencies: NumPy, SciPy, Matplotlib. The figure also requires a working
LaTeX installation on PATH (`text.usetex=True`).

## Reproducibility

Each filter runs against an **independent** seeded `default_rng`, keyed by
`(base_seed, mc_index, sigma_index, crc32(filter_name))`. Reordering or
adding/removing filters in `FILTERS` does not perturb any other filter's
numbers. The script is fully deterministic — same `base_seed` produces the
same `results.pkl`.

## LaTeX

`table.tex` requires the following macro in the preamble:

```latex
\newcommand{\pmstd}[1]{\!\pm\!#1}
```
