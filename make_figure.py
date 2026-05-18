"""Render the three-panel IEEE single-column figure from `results.pkl`.

Run as:  python NAEDH_final/make_figure.py

`text.usetex=True` requires a working LaTeX installation on PATH.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.ticker import LogLocator

from results import Results  # registers dataclasses for pickle.load

PANEL_C_FILTERS = ["NAEDH-ccr", "NAEDH-lin", "Gromov", "EDH-lin", "EDH-adaptive"]

STYLES_PANEL_C = {
    "NAEDH-ccr": ("C3", "D", "-"),
    "NAEDH-lin": ("C0", "o", "--"),
    "Gromov": ("C2", "s", "-"),
    "EDH-lin": ("0.5", "*", "-."),
    "EDH-adaptive": ("k", "v", ":"),
}

STYLES_PANEL_A = {
    # label_in_pickle:   (color, marker, linestyle, label_template)
    "naedh_ccr": ("C3", "D", "-", r"NAEDH-ccr\_10"),
    "edh_adapt_n10": ("k", "v", ":", r"EDH-adapt\_10"),
    "edh_adapt_avg": ("k", "^", "--", r"EDH-adapt\_{n}"),
}


def _set_rcparams():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.8,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def _panel_tag(ax, text, loc="tl"):
    """loc: 'tl' (top-left), 'tr' (top-right), 'br' (bottom-right)."""
    positions = {
        "tl": (0.03, 0.96, "left", "top"),
        "tr": (0.97, 0.96, "right", "top"),
        "br": (0.99, 0.04, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        rf"\textbf{{{text}}}",
        transform=ax.transAxes,
        fontsize=8,
        ha=ha,
        va=va,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
    )


def panel_a(ax, results: Results):
    """Pos err vs λ for the first measurement update — substep convergence."""
    truth_xy = results.s_truth[1, :2]
    for trace in results.first_step_substeps:
        if trace.label not in STYLES_PANEL_A:
            continue
        c, m, ls, label_template = STYLES_PANEL_A[trace.label]
        n = trace.lams.shape[0] - 1
        err = np.linalg.norm(trace.means[:, :2] - truth_xy, axis=1)
        positive = trace.lams > 0
        ax.plot(
            trace.lams[positive],
            err[positive],
            color=c,
            marker=m,
            linestyle=ls,
            markersize=2.0,
            label=label_template.format(n=n),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-10, 1.0)
    ax.set_xlabel(r"$\lambda$", labelpad=1)
    ax.set_ylabel("pos err [m]", labelpad=1)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4, subs=[1.0]))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4, subs=[1.0]))
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
    ax.legend(
        loc="lower left",
        framealpha=0.85,
        handlelength=1.6,
        borderpad=0.25,
        labelspacing=0.2,
        handletextpad=0.35,
        borderaxespad=0.3,
    )
    ax.tick_params(pad=1.5)
    _panel_tag(ax, "b)", "tr")


def panel_b(ax, results: Results):
    """Experimental setup — truth, sensors, prior mean, 1σ prior ellipse."""
    sensors = results.sensor_positions
    truth = results.s_truth
    s0_hat = results.s0_hat
    ax.plot(sensors[:, 0], sensors[:, 1], "k^", markersize=4, label="sensors")
    ax.plot(truth[:, 0], truth[:, 1], "k-", linewidth=1.0, label="truth")
    ax.plot(truth[1:, 0], truth[1:, 1], "ko", markersize=1.5)
    ax.plot(truth[0, 0], truth[0, 1], "k*", markersize=7, label=r"$x_0$")
    ax.plot(s0_hat[0], s0_hat[1], "rx", markersize=3.5, markeredgewidth=0.9)
    evals_xy, evecs_xy = np.linalg.eigh(results.P0[:2, :2])
    order = np.argsort(evals_xy)[::-1]
    evals_xy, evecs_xy = evals_xy[order], evecs_xy[:, order]
    angle_deg = np.degrees(np.arctan2(evecs_xy[1, 0], evecs_xy[0, 0]))
    width, height = 2.0 * np.sqrt(evals_xy)
    ax.add_patch(
        Ellipse(
            xy=(s0_hat[0], s0_hat[1]),
            width=width,
            height=height,
            angle=angle_deg,
            facecolor="red",
            edgecolor="red",
            alpha=0.22,
            linestyle="--",
            linewidth=0.8,
            label=r"$1\sigma$ prior",
        )
    )
    ax.set_xlabel("x [m]", labelpad=1)
    ax.set_ylabel("y [m]", labelpad=1)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(
        loc="lower center",
        ncol=2,
        framealpha=0.85,
        handlelength=1.5,
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.35,
        handleheight=1.2,
        borderaxespad=0.3,
        columnspacing=0.8,
    )
    ax.set_ylim(-50, 70)
    ax.tick_params(pad=1.5)
    _panel_tag(ax, "a)", "tl")


def panel_c(ax, results: Results):
    """Mean pos err vs time at the primary σ_θ for the five flow filters."""
    primary_sd = results.primary_sigma_theta_deg
    entry = next(e for e in results.sweep if e.sigma_theta_deg == primary_sd)
    T = entry.filters[0].per_time_pos_err_mean.shape[0]
    dt = results.config["dt"]
    t_axis = np.arange(1, T + 1) * dt
    for name in PANEL_C_FILTERS:
        fs = next((f for f in entry.filters if f.name == name), None)
        if fs is None:
            continue
        c, m, ls = STYLES_PANEL_C[name]
        ax.plot(
            t_axis,
            fs.per_time_pos_err_mean,
            color=c,
            marker=m,
            linestyle=ls,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("time [s]", labelpad=1)
    ax.set_ylabel("mean pos err [m]", labelpad=1)
    ax.set_xticks(np.arange(1, T + 1))
    ax.set_xlim(0.5, T + 0.5)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4, subs=[1.0]))
    ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
    ax.legend(
        loc="upper right",
        ncol=3,
        framealpha=0.85,
        handlelength=1.6,
        borderpad=0.25,
        labelspacing=0.2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.3,
    )
    ax.tick_params(pad=1.5)
    _panel_tag(ax, "c)", "br")


def main():
    here = Path(__file__).parent
    with open(here / "results.pkl", "rb") as f:
        results: Results = pickle.load(f)

    _set_rcparams()
    fig = plt.figure(figsize=(3.5, 3.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 0.9],
        width_ratios=[1.0, 1.0],
        hspace=0.32,
        wspace=0.45,
        left=0.12,
        right=0.995,
        top=0.97,
        bottom=0.13,
    )
    panel_b(fig.add_subplot(gs[0, 0]), results)
    panel_a(fig.add_subplot(gs[0, 1]), results)
    panel_c(fig.add_subplot(gs[1, :]), results)

    out_png = here / "three_panel_figure.png"
    out_pdf = here / "three_panel_figure.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
