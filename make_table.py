"""Generate the LaTeX RMSE / runtime table from `results.pkl`.

Run as:  python NAEDH_final/make_table.py

Writes `NAEDH_final/table.tex` and echoes to stdout. The LaTeX requires
    \newcommand{\pmstd}[1]{\!\pm\!#1}
in the preamble — a reminder is emitted as a leading comment.
"""

import math
import pickle
import sys
from pathlib import Path

from results import Results  # registers the dataclasses for pickle.load


# Row layout. Filter names must match `FilterStats.name` strings produced by
# run_experiments.py. EDH-lin and any other filter not listed here is silently
# omitted from the table (EDH-lin is used only by the figure).
LAYOUT = [
    # (filter_name,        display_label,    N_lambda, N_particles, group_break_after)
    ('NAEDH-lin',          'NAEDH-lin.',     10,    500,    False),
    ('NAEDH-ccr',          'NAEDH-ccr.',     10,    500,    True),
    ('EDH-adaptive',       'EDH-adapt.',     10,    500,    False),
    ('Gromov',             'Gromov-adapt.',  10,    500,    True),
    ('PF (Np=10,000)',     'Bootstrap PF',   None,  10_000,  False),
    ('PF (Np=100,000)',    'Bootstrap PF',   None,  100_000, False),
    ('EKF',                'EKF',            None,  None,    False),
]

DIVERGED_FRAC = 0.99


def fmt(x, n=2):
    """Format `x` to n significant figures, dropping trailing zeros for ints."""
    if not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    y = round(x, d)
    if y == 0:
        return "0"
    if y == int(y) and abs(y) >= 10 ** (n - 1):
        return str(int(y))
    d = max(0, n - int(math.floor(math.log10(abs(y)))) - 1)
    return f"{y:.{d}f}"


def cell_rmse(mean, std):
    return rf"${fmt(mean)} \pmstd{{{fmt(std)}}}$"


def boldify(s):
    r"""Wrap the leading numeric part of `$x \pmstd{y}$` (or `$x$`) in \mathbf{}."""
    if not (s.startswith("$") and s.endswith("$")):
        return s
    inner = s[1:-1]
    if r"\pmstd" in inner:
        head, tail = inner.split(r"\pmstd", 1)
        return rf"$\mathbf{{{head.strip()}}} \pmstd{tail}$"
    return rf"$\mathbf{{{inner.strip()}}}$"


def np_str(N_p):
    if N_p is None:
        return "-"
    if N_p >= 1000 and math.log10(N_p).is_integer():
        return f"10^{int(math.log10(N_p))}"
    return f"{N_p:,}".replace(",", r"\,")


def np_field(N_lam, N_p):
    lam_s = "-" if N_lam is None else str(N_lam)
    return rf"$({lam_s},\,{np_str(N_p)})$"


def find_stats(filter_list, name):
    for fs in filter_list:
        if fs.name == name:
            return fs
    return None


def emit(results: Results) -> str:
    sigmas = [e.sigma_theta_deg for e in results.sweep]
    n_cols = len(sigmas)
    n_rows = len(LAYOUT)

    cells     = [[None] * n_cols for _ in range(n_rows)]
    means     = [[None] * n_cols for _ in range(n_rows)]
    time_strs = [None] * n_rows

    for r, (name, *_rest) in enumerate(LAYOUT):
        ms_vals = []
        for c, entry in enumerate(results.sweep):
            fs = find_stats(entry.filters, name)
            if fs is None:
                cells[r][c] = "n/a"
                continue
            if fs.frac_diverged >= DIVERGED_FRAC:
                cells[r][c] = "diverged"
            elif not math.isfinite(fs.mean_rmse):
                cells[r][c] = "n/a"
            else:
                cells[r][c] = cell_rmse(fs.mean_rmse, fs.std_rmse)
                means[r][c] = fs.mean_rmse
            if math.isfinite(fs.ms_per_update):
                ms_vals.append(fs.ms_per_update)
        time_strs[r] = (f"${fmt(sum(ms_vals) / len(ms_vals))}$"
                        if ms_vals else "n/a")

    # Per-column boldface: smallest formatted mean wins (2-sig-fig tie tolerance).
    for c in range(n_cols):
        vals = [(r, means[r][c]) for r in range(n_rows) if means[r][c] is not None]
        if not vals:
            continue
        mn_fmt = fmt(min(v for _, v in vals))
        for r, v in vals:
            if fmt(v) == mn_fmt:
                cells[r][c] = boldify(cells[r][c])

    lines = []
    lines.append(r"% Requires:  \newcommand{\pmstd}[1]{\!\pm\!#1}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RMSE (mean $\pm$ std) and per-update runtime at "
                 r"three measurement-noise levels $\sigma_\theta$.}")
    lines.append(r"\label{tab:rmse}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{@{}l c " + "c" * n_cols + r" c@{}}")
    lines.append(r"\toprule")
    lines.append(rf" & & \multicolumn{{{n_cols}}}{{c}}{{RMSE [m]}} & \\")
    lines.append(rf"\cmidrule(lr){{3-{2 + n_cols}}}")
    sigma_hdr = " & ".join(rf"$\sigma_\theta={sd:g}^\circ$" for sd in sigmas)
    lines.append(r"Method & $(N_\lambda, N_p)$")
    lines.append(rf"       & {sigma_hdr}")
    lines.append(r"       & ms/update \\")
    lines.append(r"\midrule")
    for r, (_name, label, N_lam, N_p, group_break) in enumerate(LAYOUT):
        rmse_row = " & ".join(cells[r])
        lines.append(f"{label:<14}& {np_field(N_lam, N_p):<22}"
                     f"& {rmse_row} & {time_strs[r]} \\\\")
        if group_break:
            lines.append(r"\midrule[0.3pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    here = Path(__file__).parent
    with open(here / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    tex = emit(results)
    print(tex)
    out_path = here / 'table.tex'
    out_path.write_text(tex + "\n")
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
