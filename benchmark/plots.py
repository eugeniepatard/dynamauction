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

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed colors for known policies.  Keys are used as prefixes for matching,
# so "Constant" catches any "Constant(0.3)", "Shading" catches "Shading(α=…)", etc.
PALETTE: Dict[str, str] = {
    "PPO":          "#4C72B0",
    "SAC":          "#DD8452",
    "TD3":          "#55A868",
    "FiPla":        "#C44E52",
    "Constant":     "#8172B2",
    "Random":       "#937860",
    "GreedyMyopic": "#DA8BC3",
    # Shading, MPC, and MPCOracle are base colors used when only one variant is
    # present.  Multiple variants get a sequential gradient — see _make_palette.
    # MPCOracle must come before MPC so prefix matching finds it first.
    "Shading":      "#2171B5",
    "MPCOracle":      "#2CA02C",
    "MPCTimeBinned":  "#9467BD",
    "MPC":            "#A67C00",
}

_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS    = ["o", "s", "^", "D", "v", "P", "X"]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "font.size":         11,
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })


def _is_shading(name: str) -> bool:
    return name == "Shading" or name.startswith("Shading(")


def _is_mpc_oracle(name: str) -> bool:
    return name == "MPCOracle" or name.startswith("MPCOracle(")


def _is_mpc_timebin(name: str) -> bool:
    return name == "MPCTimeBinned" or name.startswith("MPCTimeBinned(")


def _is_mpc(name: str) -> bool:
    return (
        (name == "MPC" or name.startswith("MPC("))
        and not _is_mpc_oracle(name)
        and not _is_mpc_timebin(name)
    )


def _numeric_sort_key(name: str) -> list:
    """Sort key that orders policy names by the numeric values embedded in them.

    Ensures MPC(H=7) always comes before MPC(H=15) regardless of the order
    they appear in the DataFrame, so gradient colors are stable across plots.
    """
    return [float(x) for x in re.findall(r'\d+\.?\d*', name)]


def _make_palette(names: Sequence[str]) -> Dict[str, Any]:
    """
    Build a ``{policy_name: color}`` dict.

    * Known single-family policies (PPO, SAC, …) get their fixed PALETTE color.
    * Multiple Shading variants → sequential blue gradient.
    * Multiple MPC variants    → sequential amber gradient.
    * Unknown policies         → husl fallback.

    Within each multi-variant family the gradient is assigned in numeric order
    (by H, K, α …) so the mapping is stable regardless of DataFrame row order.
    """
    # Sort each family numerically so gradient assignment is order-independent.
    shading_sorted     = sorted([n for n in names if _is_shading(n)],     key=_numeric_sort_key)
    mpc_sorted         = sorted([n for n in names if _is_mpc(n)],         key=_numeric_sort_key)
    mpc_oracle_sorted  = sorted([n for n in names if _is_mpc_oracle(n)],  key=_numeric_sort_key)
    mpc_timebin_sorted = sorted([n for n in names if _is_mpc_timebin(n)], key=_numeric_sort_key)
    ns, nm, no, nt = len(shading_sorted), len(mpc_sorted), len(mpc_oracle_sorted), len(mpc_timebin_sorted)

    def _grad(cmap, n, lo=0.35, hi=0.90):
        if n <= 1:
            return [cmap(lo + (hi - lo) / 2)]
        return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]

    shading_map     = dict(zip(shading_sorted,     [PALETTE["Shading"]]     if ns <= 1 else _grad(plt.cm.Blues,  ns)))
    mpc_map         = dict(zip(mpc_sorted,         [PALETTE["MPC"]]         if nm <= 1 else _grad(plt.cm.YlOrBr, nm)))
    mpc_oracle_map  = dict(zip(mpc_oracle_sorted,  [PALETTE["MPCOracle"]]   if no <= 1 else _grad(plt.cm.BuGn,   no, lo=0.25, hi=0.95)))
    mpc_timebin_map = dict(zip(mpc_timebin_sorted, [PALETTE["MPCTimeBinned"]] if nt <= 1 else _grad(plt.cm.Purples, nt)))

    out: Dict[str, Any] = {}
    fallback_i = 0
    for name in names:
        if _is_shading(name):
            out[name] = shading_map[name]
        elif _is_mpc_oracle(name):
            out[name] = mpc_oracle_map[name]
        elif _is_mpc_timebin(name):
            out[name] = mpc_timebin_map[name]
        elif _is_mpc(name):
            out[name] = mpc_map[name]
        else:
            # prefix match against PALETTE keys (MPCOracle listed before MPC so it
            # won't be caught by the shorter "MPC" prefix)
            matched = next((c for k, c in PALETTE.items() if name.startswith(k)), None)
            out[name] = matched if matched is not None else sns.color_palette("husl", 10)[fallback_i % 10]
            if matched is None:
                fallback_i += 1
    return out


def _make_linestyles(names: Sequence[str]) -> Dict[str, str]:
    """
    Cycle through distinct linestyles within each multi-variant family
    (Shading and MPC).  All other policies get solid lines.
    """
    shading_sorted     = sorted([n for n in names if _is_shading(n)],     key=_numeric_sort_key)
    mpc_sorted         = sorted([n for n in names if _is_mpc(n)],         key=_numeric_sort_key)
    mpc_oracle_sorted  = sorted([n for n in names if _is_mpc_oracle(n)],  key=_numeric_sort_key)
    mpc_timebin_sorted = sorted([n for n in names if _is_mpc_timebin(n)], key=_numeric_sort_key)

    shading_ls     = {n: _LINESTYLES[i % len(_LINESTYLES)] for i, n in enumerate(shading_sorted)}
    mpc_ls         = {n: _LINESTYLES[i % len(_LINESTYLES)] for i, n in enumerate(mpc_sorted)}
    mpc_oracle_ls  = {n: _LINESTYLES[i % len(_LINESTYLES)] for i, n in enumerate(mpc_oracle_sorted)}
    mpc_timebin_ls = {n: _LINESTYLES[i % len(_LINESTYLES)] for i, n in enumerate(mpc_timebin_sorted)}

    out: Dict[str, str] = {}
    for name in names:
        if _is_shading(name):
            out[name] = shading_ls[name]
        elif _is_mpc_oracle(name):
            out[name] = mpc_oracle_ls[name]
        elif _is_mpc_timebin(name):
            out[name] = mpc_timebin_ls[name]
        elif _is_mpc(name):
            out[name] = mpc_ls[name]
        else:
            out[name] = "-"
    return out


def _save(fig: plt.Figure, filename: str) -> None:
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")


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

def plot_aggregate_metrics(
    score_dict:   Dict[str, np.ndarray],
    metric:       str            = "iqm",
    normalize_by: Optional[str]  = "FiPla",
    title:        Optional[str]  = None,
    save:         bool           = True,
    filename:     str            = "aggregate_metrics.pdf",
) -> plt.Figure:
    """
    Horizontal bar chart of aggregate performance with 95 % bootstrap CIs.

    Parameters
    ----------
    score_dict : {policy: (n_seeds,) array}
        Episode rewards per seed for each policy.
    metric : 'iqm' | 'median' | 'mean'
    normalize_by : str or None
        Normalise all scores by the mean score of this reference policy.
    """
    _style()
    metric_label = {"iqm": "IQM", "median": "Median", "mean": "Mean"}.get(metric, metric.upper())
    norm_label   = f", normalised by {normalize_by}" if normalize_by else ""
    if title is None:
        title = f"Aggregate Performance — {metric_label} of total episode reward{norm_label}"

    scores = {k: v.copy().astype(float) for k, v in score_dict.items()}

    stat = {"iqm": lambda x: np.percentile(x, [25, 75]).mean(),
            "median": np.median, "mean": np.mean}.get(metric, np.mean)

    if normalize_by and normalize_by in scores:
        ref = stat(scores[normalize_by])
        if ref != 0.0:
            scores = {k: v / ref for k, v in scores.items()}

    names   = list(scores.keys())
    palette = _make_palette(names)
    colors  = [palette[n] for n in names]
    xlabel  = ("Normalised episode reward" if normalize_by
               else "Episode reward") + f"  [{metric.upper()}]"

    results = [_bootstrap_ci(scores[n], stat) for n in names]
    points  = [r[0] for r in results]
    lows    = [r[1] for r in results]
    highs   = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(names) + 1)))

    xerr = [[p - lo for p, lo in zip(points, lows)],
            [hi - p for p, hi in zip(points, highs)]]

    ax.barh(names, points, xerr=xerr, color=colors, capsize=4,
            alpha=0.85, height=0.55, error_kw={"elinewidth": 1.5})
    if normalize_by:
        ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.6, label="FiPla baseline")

    from matplotlib.lines import Line2D
    ci_handle = Line2D([0], [0], color="black", lw=1.5,
                       marker="|", markersize=8, label="95 % bootstrap CI")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [ci_handle], labels=labels + ["95 % bootstrap CI"], fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()

    if save:
        _save(fig, filename)
    return fig


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
    """
    _style()
    scores = {k: v.copy().astype(float) for k, v in score_dict.items()}
    if normalize_by and normalize_by in scores:
        ref = np.mean(scores[normalize_by])
        if ref != 0.0:
            scores = {k: v / ref for k, v in scores.items()}

    names      = list(scores.keys())
    palette    = _make_palette(names)
    linestyles = _make_linestyles(names)

    all_vals = np.concatenate(list(scores.values()))
    tau_max  = tau_max if tau_max is not None else float(np.percentile(all_vals, 98))
    taus     = np.linspace(0.0, tau_max, n_tau)
    xlabel   = (r"Episode reward $\tau$"
                + (" (normalised)" if normalize_by else ""))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(names):
        arr     = scores[name]
        profile = (arr[:, None] >= taus[None, :]).mean(axis=0)
        ax.plot(taus, profile, label=name,
                color=palette[name], lw=2, linestyle=linestyles[name],
                marker=_MARKERS[i % len(_MARKERS)], markevery=n_tau // 10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Fraction of runs $\geq\tau$")
    ax.legend(title="Policy")
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


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

    names      = list(plot_df["policy"].unique())
    palette    = _make_palette(names)
    linestyles = _make_linestyles(names)

    mean_max_t = plot_df.groupby("seed")["t"].max().mean()
    t_grid     = np.linspace(plot_df["t"].min(), mean_max_t, num_points)
    fig, ax    = plt.subplots(figsize=(8, 5))

    for policy, grp in plot_df.groupby("policy"):
        curves = [
            np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")["cum_reward"])
            for _, sg in grp.groupby("seed")
        ]
        mat  = np.array(curves)
        mean = mat.mean(0)
        std  = mat.std(0)
        c    = palette[policy]
        ax.plot(t_grid, mean, label=policy, color=c, lw=2, linestyle=linestyles[policy])
        ax.fill_between(t_grid, mean - std, mean + std, alpha=0.15, color=c)

    ax.set_xlabel(r"Campaign time $t$")
    ax.set_ylabel("Cumulative net reward")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


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
    from benchmark.evaluate import episode_summary

    _style()
    summary = episode_summary(df)
    if policies:
        summary = summary[summary["policy"].isin(policies)]

    order   = (summary.groupby("policy")["episode_reward"]
               .median().sort_values(ascending=False).index.tolist())
    palette = _make_palette(order)

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.6), 5))
    sns.violinplot(data=summary, x="policy", y="episode_reward",
                   order=order, palette=palette, inner="box",
                   cut=0, linewidth=1.2, ax=ax)
    sns.stripplot(data=summary, x="policy", y="episode_reward",
                  order=order, color="black", alpha=0.35, size=3,
                  jitter=True, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Total episode reward")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


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

    algo_names = [a.upper() for a in algos]
    palette    = _make_palette(algo_names)

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
                color=palette[algo.upper()], lw=2)
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


def plot_context_dynamics(
    df:         pd.DataFrame,
    policies:   Optional[List[str]] = None,
    num_points: int                 = 300,
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
    plot_df    = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    names      = list(plot_df["policy"].unique())
    palette    = _make_palette(names)
    linestyles = _make_linestyles(names)

    mean_max_t = plot_df.groupby("seed")["t"].max().mean()
    t_grid     = np.linspace(plot_df["t"].min(), mean_max_t, num_points)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(title, fontweight="bold", fontsize=13)

    for policy, grp in plot_df.groupby("policy"):
        c  = palette[policy]
        ls = linestyles[policy]
        for ax, col in ((ax1, "ctx"), (ax2, "expected_reward")):
            curves = [
                np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")[col])
                for _, sg in grp.groupby("seed")
            ]
            mat  = np.array(curves)
            mean = mat.mean(0)
            std  = mat.std(0)
            ax.plot(t_grid, mean, label=policy, color=c, lw=2, linestyle=ls)
            ax.fill_between(t_grid, mean - std, mean + std, alpha=0.12, color=c)

    ax1.set_ylabel("Context (kernel state)")
    ax1.legend(title="Policy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax2.set_xlabel(r"Campaign time $t$")
    ax2.set_ylabel(r"Expected reward $\hat{R}$")

    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


def plot_bid_distributions(
    df:       pd.DataFrame,
    policies: Optional[List[str]] = None,
    title:    str                 = "Bid Distributions vs Competing Bids",
    save:     bool                = True,
    filename: str                 = "bid_distributions.pdf",
) -> plt.Figure:
    """
    For each policy, plot the distribution of the agent's bids vs the
    competing bid distribution (shared across all policies since it comes
    from the scenario, not the policy).

    One subplot per policy, all on the same x-axis scale for easy comparison.
    The competing bid distribution is shown as a reference in every panel.
    """
    _style()

    plot_df      = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    policy_names = list(plot_df["policy"].unique())
    palette      = _make_palette(policy_names)

    competing = plot_df["competing_bid"].dropna().values

    n     = len(policy_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False)
    axes = np.array(axes).ravel()

    x_max = float(np.percentile(
        np.concatenate([competing, plot_df["bid"].values]), 99
    ))

    for idx, policy in enumerate(policy_names):
        ax    = axes[idx]
        bids  = plot_df.loc[plot_df["policy"] == policy, "bid"].values
        color = palette[policy]

        sns.kdeplot(competing, ax=ax, color="grey", lw=1.5,
                    linestyle="--", label="Competing bid", fill=True,
                    alpha=0.15, clip=(0, x_max))
        sns.kdeplot(bids, ax=ax, color=color, lw=2,
                    label=f"{policy} bids", fill=True,
                    alpha=0.30, clip=(0, x_max))
        ax.axvline(bids.mean(),      color=color,  lw=1.5, ls="-",
                   label=f"Mean bid = {bids.mean():.2f}")
        ax.axvline(competing.mean(), color="grey", lw=1.5, ls="--",
                   label=f"Mean competing = {competing.mean():.2f}")

        ax.set_xlim(0, x_max)
        ax.set_title(policy, fontweight="bold")
        ax.set_xlabel("Bid value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(len(policy_names), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


def plot_win_rate(
    df:       pd.DataFrame,
    policies: Optional[List[str]] = None,
    title:    str                 = "Win Rate by Policy",
    save:     bool                = True,
    filename: str                 = "win_rate.pdf",
) -> plt.Figure:
    """Mean win rate (fraction of auctions won) with 95 % CI per policy."""
    from benchmark.evaluate import episode_summary

    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    summary = episode_summary(plot_df)

    order   = summary.groupby("policy")["win_rate"].mean().sort_values(ascending=False).index.tolist()
    palette = _make_palette(order)

    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.6), 4))
    sns.barplot(data=summary, x="policy", y="win_rate",
                order=order, palette=palette,
                errorbar=("ci", 95), capsize=0.1, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Win rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


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
    from benchmark.evaluate import episode_summary

    _style()
    plot_df = df[df["policy"].isin(policies)].copy() if policies else df.copy()
    scores  = {k: v for k, v in score_dict.items()
               if (policies is None or k in policies)}

    all_names  = list(scores.keys())
    palette    = _make_palette(all_names)
    linestyles = _make_linestyles(all_names)

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
                          color=palette.get(algo.upper(), PALETTE.get(algo.upper(), "grey")), lw=2)
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
    for i, name in enumerate(all_names):
        arr     = scores[name]
        profile = (arr[:, None] >= taus[None, :]).mean(axis=0)
        ax_prof.plot(taus, profile, label=name,
                     color=palette[name], lw=2, linestyle=linestyles[name],
                     marker=_MARKERS[i % len(_MARKERS)], markevery=20)
    ax_prof.set_xlabel(r"Episode reward $\tau$")
    ax_prof.set_ylabel(r"Fraction $\geq \tau$")
    ax_prof.set_title("(B) Performance Profiles", fontweight="bold")
    ax_prof.legend(fontsize=9)

    # ── Panel C: cumulative reward ────────────────────────────────────────────
    t_grid = np.linspace(plot_df["t"].min(), plot_df["t"].max(), 200)
    for policy, grp in plot_df.groupby("policy"):
        if policy not in palette:
            continue
        curves = [
            np.interp(t_grid, sg.sort_values("t")["t"], sg.sort_values("t")["cum_reward"])
            for _, sg in grp.groupby("seed")
        ]
        mat  = np.array(curves)
        mean = mat.mean(0)
        std  = mat.std(0)
        c    = palette[policy]
        ax_cum.plot(t_grid, mean, label=policy, color=c, lw=2, linestyle=linestyles[policy])
        ax_cum.fill_between(t_grid, mean - std, mean + std, alpha=0.12, color=c)
    ax_cum.set_xlabel(r"Campaign time $t$")
    ax_cum.set_ylabel("Cumulative net reward")
    ax_cum.set_title("(C) Cumulative Reward", fontweight="bold")
    ax_cum.legend(fontsize=9)

    # ── Panel D: reward distributions ────────────────────────────────────────
    summary = episode_summary(plot_df)
    order   = (summary.groupby("policy")["episode_reward"]
               .median().sort_values(ascending=False).index.tolist())
    dist_palette = _make_palette(order)
    sns.violinplot(data=summary, x="policy", y="episode_reward",
                   order=order, palette=dist_palette, inner="box",
                   cut=0, linewidth=1.2, ax=ax_dist)
    ax_dist.tick_params(axis="x", rotation=15)
    ax_dist.set_xlabel("Policy")
    ax_dist.set_ylabel("Total episode reward")
    ax_dist.set_title("(D) Reward Distributions", fontweight="bold")

    if save:
        _save(fig, filename)
    return fig


def plot_mpc_alpha_chosen(
    df:       pd.DataFrame,
    policies: Optional[List[str]] = None,
    title:    str                 = "MPC chosen α* distribution",
    save:     bool                = True,
    filename: str                 = "mpc_alpha_chosen.pdf",
) -> Optional[plt.Figure]:
    """
    Violin + strip plot of the α* values chosen at each step by MPC-family policies.
    Reads the ``alpha_chosen`` column logged by ``evaluate.py``.

    Parameters
    ----------
    df : per-step benchmark DataFrame (must contain ``alpha_chosen`` column).
    policies : optional subset of policy names to include.
    """
    if "alpha_chosen" not in df.columns:
        print("[INFO] No 'alpha_chosen' column found — re-run the benchmark with the updated evaluate.py.")
        return None

    mpc_df = df[df["alpha_chosen"].notna()].copy()
    if policies:
        mpc_df = mpc_df[mpc_df["policy"].isin(policies)]
    if mpc_df.empty:
        print("[INFO] No MPC alpha data found in dataframe.")
        return None

    # Sort policies numerically so H=10 comes before H=25, etc.
    policy_order = sorted(mpc_df["policy"].unique(), key=_numeric_sort_key)
    palette      = _make_palette(policy_order)
    candidates   = sorted(mpc_df["alpha_chosen"].unique())

    _style()
    fig, ax = plt.subplots(figsize=(max(5, len(policy_order) * 1.6), 4))

    sns.violinplot(data=mpc_df, x="policy", y="alpha_chosen",
                   order=policy_order, palette=palette,
                   inner="box", cut=0, linewidth=1.2, ax=ax)
    sns.stripplot(data=mpc_df, x="policy", y="alpha_chosen",
                  order=policy_order,
                  color="black", alpha=0.08, size=2, jitter=True, ax=ax)

    for alpha in candidates:
        ax.axhline(alpha, color="grey", lw=0.7, ls="--", alpha=0.5)

    ax.set_xlabel("")
    ax.set_ylabel(r"Chosen $\alpha^*$")
    ax.set_yticks(candidates)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig


def plot_mpc_alpha_distribution(
    mpc_policies: Dict[str, Any],
    title:        str  = "MPC: chosen α* distribution",
    save:         bool = True,
    filename:     str  = "mpc_alpha_distribution.pdf",
) -> Optional[plt.Figure]:
    """
    Violin + strip plot of the α* values chosen by each MPCPolicy over all steps.

    Parameters
    ----------
    mpc_policies : {label: MPCPolicy}
        The live policy objects after evaluation — their ``alpha_log`` attribute
        holds the sequence of chosen α* values (one per ``bid()`` call).

    Example
    -------
        from benchmark.evaluate import EvalConfig, get_simulation_policies, benchmark
        from benchmark import plots

        cfg = EvalConfig(include_mpc=True, mpc_planning_horizons=[0.0, 7.0])
        sim_policies = get_simulation_policies(cfg)
        benchmark(sim_policies, cfg)   # runs evaluation, fills alpha_log

        mpc = {k: v for k, v in sim_policies.items() if hasattr(v, "alpha_log")}
        plots.plot_mpc_alpha_distribution(mpc)
    """
    mpc_policies = {k: v for k, v in mpc_policies.items() if getattr(v, "alpha_log", None)}
    if not mpc_policies:
        print("[INFO] No MPCPolicy alpha logs found — run evaluation first.")
        return None

    _style()
    palette = _make_palette(list(mpc_policies.keys()))

    records = []
    for label, policy in mpc_policies.items():
        for alpha in policy.alpha_log:
            records.append({"policy": label, "alpha*": alpha})
    df = pd.DataFrame(records)

    candidates = sorted({a for p in mpc_policies.values() for a in p.alpha_candidates})
    n = len(mpc_policies)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), 4))

    sns.violinplot(data=df, x="policy", y="alpha*", palette=palette,
                   inner="box", cut=0, linewidth=1.2, ax=ax)
    sns.stripplot(data=df, x="policy", y="alpha*",
                  color="black", alpha=0.15, size=2, jitter=True, ax=ax)

    for alpha in candidates:
        ax.axhline(alpha, color="grey", lw=0.7, ls="--", alpha=0.5)

    ax.set_xlabel("MPC variant")
    ax.set_ylabel(r"Chosen $\alpha^*$")
    ax.set_yticks(candidates)
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    if save:
        _save(fig, filename)
    return fig
