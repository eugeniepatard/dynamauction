"""
Publication-quality benchmark plots (NeurIPS style).

All functions accept a ``save`` flag; figures are written to
``benchmark/plots/`` as PDF (vector, suitable for paper submission).

Open RL Benchmark integration
------------------------------
When ``rliable`` is installed (``pip install rliable``), the following
functions use stratified bootstrap confidence intervals and the official
Open RL Benchmark aggregation metrics:
  * ``plot_aggregate_metrics``   — IQM / Median / Mean bar chart
  * ``plot_performance_profiles`` — fraction of runs ≥ τ

Without ``rliable``, both fall back to plain bootstrap CIs and manual
computation, so figures are still correct — just without the IQM estimator.

Training-curve plots (``plot_training_curves``) parse TensorBoard event
files and require ``tensorboard`` (``pip install tensorboard``).

Quick start
-----------
    from benchmark.evaluate import run_full_benchmark, score_matrix
    from benchmark import plots

    df     = run_full_benchmark()
    scores = score_matrix(df)

    plots.plot_aggregate_metrics(scores)
    plots.plot_performance_profiles(scores)
    plots.plot_cumulative_reward(df)
    plots.plot_reward_distribution(df)
    plots.plot_training_curves()
    plots.plot_context_dynamics(df)
    plots.plot_full_benchmark(df, scores)    # 4-panel summary figure
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared colour palette (extend as needed) ───────────────────────────────────
PALETTE: Dict[str, str] = {
    "PPO":          "#4C72B0",
    "SAC":          "#DD8452",
    "TD3":          "#55A868",
    "FiPla":        "#C44E52",
    "Constant(0.5)":"#8172B2",
    "Random":       "#937860",
}

_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _style() -> None:
    plt.rcParams.update({
        "figure.dpi":       150,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        11,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })


def _color(name: str, idx: int = 0) -> str:
    return PALETTE.get(name, sns.color_palette("husl", 10)[idx % 10])


def _save(fig: plt.Figure, filename: str) -> None:
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")


def _try_rliable():
    """Return (rly, metrics, plot_utils) or (None, None, None) if not installed."""
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics
        from rliable import plot_utils as rly_plot
        return rly, rly_metrics, rly_plot
    except ImportError:
        return None, None, None


def _bootstrap_ci(
    arr: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 2_000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (point, lower, upper) with plain bootstrap."""
    boots = np.array([stat_fn(np.random.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 - (1 - ci) / 2) * 100)
    return float(stat_fn(arr)), float(lo), float(hi)


# ── 1. Aggregate metrics bar chart ─────────────────────────────────────────────

def plot_aggregate_metrics(
    score_dict:   Dict[str, np.ndarray],
    metric:       str            = "iqm",
    normalize_by: Optional[str]  = "FiPla",
    title:        str            = "Aggregate Performance",
    save:         bool           = True,
    filename:     str            = "aggregate_metrics.pdf",
) -> plt.Figure:
    """
    Horizontal bar chart of aggregate performance with 95 % bootstrap CIs.

    Uses ``rliable`` IQM / Median / Mean when available; falls back to plain
    bootstrap otherwise.

    Parameters
    ----------
    score_dict : {policy: (n_seeds,) array}
        Episode rewards per seed for each policy.
    metric : 'iqm' | 'median' | 'mean'
    normalize_by : str or None
        Normalise all scores by the mean score of this reference policy.
    """
    _style()
    rly, rly_metrics, rly_plot = _try_rliable()

    scores = {k: v.copy().astype(float) for k, v in score_dict.items()}
    if normalize_by and normalize_by in scores:
        ref = scores[normalize_by].mean()
        if ref != 0.0:
            scores = {k: v / ref for k, v in scores.items()}

    names  = list(scores.keys())
    colors = [_color(n, i) for i, n in enumerate(names)]
    xlabel = ("Normalised episode reward" if normalize_by
              else "Episode reward") + f"  [{metric.upper()}]"

    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(names) + 1)))

    if rly is not None:
        _metric_fn = {
            "iqm":    rly_metrics.aggregate_iqm,
            "median": rly_metrics.aggregate_median,
            "mean":   rly_metrics.aggregate_mean,
        }.get(metric, rly_metrics.aggregate_iqm)

        agg_fn = lambda x: np.array([_metric_fn(x)])
        # rliable expects shape (n_tasks=1, n_seeds)
        rl_scores = {k: v.reshape(1, -1) for k, v in scores.items()}
        agg, cis = rly.get_interval_estimates(rl_scores, agg_fn, reps=2_000)

        points = [float(agg[n])    for n in names]
        lows   = [float(cis[n][0]) for n in names]
        highs  = [float(cis[n][1]) for n in names]
    else:
        stat = {"iqm": lambda x: np.percentile(x, [25, 75]).mean(),
                "median": np.median, "mean": np.mean}.get(metric, np.mean)
        results = [_bootstrap_ci(scores[n], stat) for n in names]
        points = [r[0] for r in results]
        lows   = [r[1] for r in results]
        highs  = [r[2] for r in results]

    xerr = [[p - lo for p, lo in zip(points, lows)],
            [hi - p for p, hi in zip(points, highs)]]

    ax.barh(names, points, xerr=xerr, color=colors, capsize=4,
            alpha=0.85, height=0.55, error_kw={"elinewidth": 1.5})
    if normalize_by:
        ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.6, label="FiPla baseline")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save:
        _save(fig, filename)
    return fig


# ── 2. Performance profiles ────────────────────────────────────────────────────

def plot_performance_profiles(
    score_dict:   Dict[str, np.ndarray],
    tau_max:      Optional[float] = None,
    n_tau:        int             = 200,
    normalize_by: Optional[str]  = "FiPla",
    title:        str            = "Performance Profiles",
    save:         bool           = True,
    filename:     str            = "performance_profiles.pdf",
) -> plt.Figure:
    """
    Fraction of runs achieving episode reward ≥ τ, as a function of τ.
    (Open RL Benchmark Figure 2 equivalent.)

    Uses ``rliable`` stratified bootstrap CIs when available.
    """
    _style()
    rly, _, rly_plot = _try_rliable()

    scores = {k: v.copy().astype(float) for k, v in score_dict.items()}
    if normalize_by and normalize_by in scores:
        ref = scores[normalize_by].mean()
        if ref != 0.0:
            scores = {k: v / ref for k, v in scores.items()}

    all_vals = np.concatenate(list(scores.values()))
    tau_max  = tau_max if tau_max is not None else float(np.percentile(all_vals, 98))
    taus     = np.linspace(0.0, tau_max, n_tau)
    xlabel   = (r"Episode reward $\tau$"
                + (" (normalised)" if normalize_by else ""))

    fig, ax = plt.subplots(figsize=(7, 5))

    if rly is not None:
        rl_scores = {k: v.reshape(1, -1) for k, v in scores.items()}
        profiles, cis = rly.get_performance_profile(rl_scores, taus)
        colors = {k: _color(k, i) for i, k in enumerate(scores)}
        rly_plot.plot_performance_profiles(
            profiles, taus,
            performance_profile_cis=cis,
            colors=colors,
            xlabel=xlabel,
            ax=ax,
        )
    else:
        for i, (name, arr) in enumerate(scores.items()):
            profile = (arr[:, None] >= taus[None, :]).mean(axis=0)
            ax.plot(taus, profile, label=name,
                    color=_color(name, i), lw=2, marker=_MARKERS[i % len(_MARKERS)],
                    markevery=n_tau // 10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Fraction of runs $\geq\tau$")
        ax.legend(title="Policy")

    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 3. Cumulative reward trajectories ─────────────────────────────────────────

def plot_cumulative_reward(
    df:         pd.DataFrame,
    policies:   Optional[List[str]] = None,
    num_points: int                 = 300,
    title:      str                 = "Cumulative Net Reward",
    save:       bool                = True,
    filename:   str                 = "cumulative_reward.pdf",
) -> plt.Figure:
    """
    Mean ± std cumulative reward trajectories over time, interpolated onto a
    common time grid across seeds.
    """
    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()

    t_grid = np.linspace(plot_df["t"].min(), plot_df["t"].max(), num_points)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (policy, grp) in enumerate(plot_df.groupby("policy")):
        curves = [
            np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")["cum_reward"])
            for _, sg in grp.groupby("seed")
        ]
        mat  = np.array(curves)
        mean = mat.mean(0)
        std  = mat.std(0)
        c    = _color(policy, i)
        ax.plot(t_grid, mean, label=policy, color=c, lw=2)
        ax.fill_between(t_grid, mean - std, mean + std, alpha=0.15, color=c)

    ax.set_xlabel(r"Campaign time $t$")
    ax.set_ylabel("Cumulative net reward")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 4. Episode reward distributions ───────────────────────────────────────────

def plot_reward_distribution(
    df:       pd.DataFrame,
    policies: Optional[List[str]] = None,
    title:    str                 = "Episode Reward Distribution",
    save:     bool                = True,
    filename: str                 = "reward_distribution.pdf",
) -> plt.Figure:
    """
    Violin + strip plot of per-episode total rewards sorted by median.
    """
    from .evaluate import episode_summary

    _style()
    summary = episode_summary(df)
    if policies:
        summary = summary[summary["policy"].isin(policies)]

    order   = (summary.groupby("policy")["episode_reward"]
               .median().sort_values(ascending=False).index.tolist())
    palette = {p: _color(p, i) for i, p in enumerate(order)}

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.3), 5))
    sns.violinplot(data=summary, x="policy", y="episode_reward",
                   order=order, palette=palette, inner="box",
                   cut=0, linewidth=1.2, ax=ax)
    sns.stripplot(data=summary, x="policy", y="episode_reward",
                  order=order, color="black", alpha=0.35, size=3,
                  jitter=True, ax=ax)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Total episode reward")
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 5. Training curves from TensorBoard logs ──────────────────────────────────

def plot_training_curves(
    algos:    List[str]      = ("ppo", "sac", "td3"),
    logs_dir: Optional[Path] = None,
    smooth:   int            = 5,
    title:    str            = "Training Curves",
    save:     bool           = True,
    filename: str            = "training_curves.pdf",
) -> Optional[plt.Figure]:
    """
    Plot ``eval/mean_reward`` from TensorBoard event files.

    Requires ``pip install tensorboard``.  Returns ``None`` if tensorboard is
    not installed or no event files are found.
    """
    if logs_dir is None:
        logs_dir = Path(__file__).parent / "logs"

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("[WARNING] tensorboard not installed — skipping training curves.")
        return None

    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for i, algo in enumerate(algos):
        algo_dir = logs_dir / algo
        if not algo_dir.exists():
            print(f"  [WARNING] No log dir for {algo.upper()}")
            continue

        event_files = sorted(algo_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"  [WARNING] No event files for {algo.upper()}")
            continue

        ea = EventAccumulator(str(event_files[-1].parent))
        ea.Reload()
        tag = "eval/mean_reward"
        if tag not in ea.Tags().get("scalars", []):
            print(f"  [WARNING] Tag '{tag}' not found for {algo.upper()}")
            continue

        events  = ea.Scalars(tag)
        steps   = np.array([e.step  for e in events])
        rewards = np.array([e.value for e in events])

        if smooth > 1 and len(rewards) >= smooth:
            kernel  = np.ones(smooth) / smooth
            rewards = np.convolve(rewards, kernel, mode="valid")
            steps   = steps[smooth - 1:]

        ax.plot(steps, rewards, label=algo.upper(),
                color=_color(algo.upper(), i), lw=2)
        plotted = True

    if not plotted:
        print("[INFO] No training data to plot.")
        plt.close(fig)
        return None

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward (eval)")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1_000:.0f}K")
    )
    ax.legend(title="Algorithm")
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 6. Context dynamics & expected reward ─────────────────────────────────────

def plot_context_dynamics(
    df:         pd.DataFrame,
    policies:   Optional[List[str]] = None,
    num_points: int                 = 200,
    title:      str                 = "State Dynamics",
    save:       bool                = True,
    filename:   str                 = "context_dynamics.pdf",
) -> plt.Figure:
    """
    Two-panel figure:
      * Top   — context / saturation state over campaign time
      * Bottom — expected reward over campaign time
    Both with mean ± std bands across seeds.
    """
    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    t_grid  = np.linspace(plot_df["t"].min(), plot_df["t"].max(), num_points)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(title, fontweight="bold", fontsize=13)

    for i, (policy, grp) in enumerate(plot_df.groupby("policy")):
        c = _color(policy, i)
        for ax, col in ((ax1, "ctx"), (ax2, "expected_reward")):
            curves = [
                np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")[col])
                for _, sg in grp.groupby("seed")
            ]
            mat  = np.array(curves)
            mean = mat.mean(0)
            std  = mat.std(0)
            ax.plot(t_grid, mean, label=policy, color=c, lw=2)
            ax.fill_between(t_grid, mean - std, mean + std, alpha=0.12, color=c)

    ax1.set_ylabel("Context (kernel state)")
    ax1.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    ax2.set_xlabel(r"Campaign time $t$")
    ax2.set_ylabel(r"Expected reward $\hat{R}$")

    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 7. Win-rate bar chart ──────────────────────────────────────────────────────

def plot_win_rate(
    df:       pd.DataFrame,
    policies: Optional[List[str]] = None,
    title:    str                 = "Win Rate by Policy",
    save:     bool                = True,
    filename: str                 = "win_rate.pdf",
) -> plt.Figure:
    """Mean win rate (fraction of auctions won) with 95 % CI per policy."""
    from .evaluate import episode_summary

    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    summary = episode_summary(plot_df)

    order   = summary.groupby("policy")["win_rate"].mean().sort_values(ascending=False).index.tolist()
    palette = {p: _color(p, i) for i, p in enumerate(order)}

    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.2), 4))
    sns.barplot(data=summary, x="policy", y="win_rate",
                order=order, palette=palette,
                errorbar=("ci", 95), capsize=0.1, ax=ax)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Win rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


# ── 8. All-in-one 4-panel summary figure ──────────────────────────────────────

def plot_full_benchmark(
    df:         pd.DataFrame,
    score_dict: Dict[str, np.ndarray],
    policies:   Optional[List[str]] = None,
    save:       bool                = True,
    filename:   str                 = "full_benchmark.pdf",
) -> plt.Figure:
    """
    4-panel NeurIPS-style summary figure:
      [A] Training curves  |  [B] Performance profiles
      [C] Cumulative reward|  [D] Reward distributions

    Panel A is populated only when TensorBoard logs exist.
    """
    from .evaluate import episode_summary

    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    scores  = {k: v for k, v in score_dict.items()
               if (policies is None or k in policies)}

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.38)
    ax_train = fig.add_subplot(gs[0, 0])
    ax_prof  = fig.add_subplot(gs[0, 1])
    ax_cum   = fig.add_subplot(gs[1, 0])
    ax_dist  = fig.add_subplot(gs[1, 1])

    # ── Panel A: training curves ──────────────────────────────────────────────
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        logs_dir = Path(__file__).parent / "logs"
        algos_to_plot = [a for a in ("ppo", "sac", "td3")
                         if (policies is None or a.upper() in policies)]
        for i, algo in enumerate(algos_to_plot):
            algo_dir    = logs_dir / algo
            event_files = sorted(algo_dir.rglob("events.out.tfevents.*")) if algo_dir.exists() else []
            if not event_files:
                continue
            ea = EventAccumulator(str(event_files[-1].parent))
            ea.Reload()
            if "eval/mean_reward" not in ea.Tags().get("scalars", []):
                continue
            evts    = ea.Scalars("eval/mean_reward")
            steps   = np.array([e.step  for e in evts])
            rewards = np.array([e.value for e in evts])
            if len(rewards) >= 5:
                k = np.ones(5) / 5
                rewards = np.convolve(rewards, k, mode="valid")
                steps   = steps[4:]
            ax_train.plot(steps, rewards, label=algo.upper(),
                          color=_color(algo.upper(), i), lw=2)
        ax_train.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1_000:.0f}K")
        )
        ax_train.legend(title="Algo", fontsize=9)
    except Exception:
        ax_train.text(0.5, 0.5, "Run plot_training_curves()\nto populate",
                      ha="center", va="center", transform=ax_train.transAxes,
                      color="grey", fontsize=11)

    ax_train.set_xlabel("Steps")
    ax_train.set_ylabel("Eval mean reward")
    ax_train.set_title("(A) Training Curves", fontweight="bold")

    # ── Panel B: performance profiles ────────────────────────────────────────
    all_vals = np.concatenate(list(scores.values()))
    tau_max  = float(np.percentile(all_vals, 98))
    taus     = np.linspace(0.0, tau_max, 200)
    for i, (name, arr) in enumerate(scores.items()):
        profile = (arr[:, None] >= taus[None, :]).mean(axis=0)
        ax_prof.plot(taus, profile, label=name,
                     color=_color(name, i), lw=2,
                     marker=_MARKERS[i % len(_MARKERS)], markevery=20)
    ax_prof.set_xlabel(r"Episode reward $\tau$")
    ax_prof.set_ylabel(r"Fraction $\geq \tau$")
    ax_prof.set_title("(B) Performance Profiles", fontweight="bold")
    ax_prof.legend(fontsize=9)

    # ── Panel C: cumulative reward ─────────────────────────────────────────────
    t_grid = np.linspace(plot_df["t"].min(), plot_df["t"].max(), 200)
    for i, (policy, grp) in enumerate(plot_df.groupby("policy")):
        curves = [
            np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")["cum_reward"])
            for _, sg in grp.groupby("seed")
        ]
        mat  = np.array(curves)
        mean = mat.mean(0)
        std  = mat.std(0)
        c    = _color(policy, i)
        ax_cum.plot(t_grid, mean, label=policy, color=c, lw=2)
        ax_cum.fill_between(t_grid, mean - std, mean + std, alpha=0.12, color=c)
    ax_cum.set_xlabel(r"Campaign time $t$")
    ax_cum.set_ylabel("Cumulative net reward")
    ax_cum.set_title("(C) Cumulative Reward", fontweight="bold")
    ax_cum.legend(fontsize=9)

    # ── Panel D: violin distributions ─────────────────────────────────────────
    summary = episode_summary(plot_df)
    order   = (summary.groupby("policy")["episode_reward"]
               .median().sort_values(ascending=False).index.tolist())
    palette = {p: _color(p, i) for i, p in enumerate(order)}
    sns.violinplot(data=summary, x="policy", y="episode_reward",
                   order=order, palette=palette, inner="box",
                   cut=0, linewidth=1.2, ax=ax_dist)
    ax_dist.tick_params(axis="x", rotation=15)
    ax_dist.set_xlabel("Policy")
    ax_dist.set_ylabel("Total episode reward")
    ax_dist.set_title("(D) Reward Distributions", fontweight="bold")

    if save:
        _save(fig, filename)
    return fig
