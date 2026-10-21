"""
Train the MLPValuationHead used by MPCPolicy.

Usage
-----
    python -m benchmark.train_mlp
    python -m benchmark.train_mlp --n-episodes 500 --n-epochs 600 --lr 3e-4
    python -m benchmark.train_mlp --hidden-dims 128 128 --val-frac 0.2
    python -m benchmark.train_mlp --tensorboard          # launch TensorBoard live
    python -m benchmark.train_mlp --tensorboard --port 6007

Outputs (benchmark/models/mlp_valuation_head/)
----------------------------------------------
    <scenario_id>.pt        model weights
    <scenario_id>.json      training config + metrics
    <scenario_id>_loss.pdf  train / val loss curves
    <scenario_id>_pred.pdf  predicted vs actual (val set)
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

import dynamauction as dac
from dynamauction.baseline.estimators import MLPValuationHead
from dynamauction.baseline.utils import _obs_to_features
from dynamauction.env import AuctionEnv
from tqdm import tqdm

from benchmark.train_rl import SCENARIO_ID, MODELS_DIR, LOGS_DIR

MLP_DIR = MODELS_DIR / "mlp_valuation_head"
MLP_LOGS_DIR = LOGS_DIR / "mlp_valuation_head"


def collect_rollouts(scenario, n_episodes: int, seed: int = 0):
    """Run FiPla on scenario and return (X, y) feature/label arrays."""
    env = AuctionEnv(scenario=scenario)
    X_list, y_list = [], []
    for ep in tqdm(range(n_episodes), desc="Collecting rollouts", unit="ep"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            er = float(obs["expected_reward"].ravel()[0])
            X_list.append(_obs_to_features(obs))
            y_list.append(er)
            obs, _, done, _, _ = env.step(np.array([er], dtype=np.float32))
    env.close()
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def _plot_loss_curves(history: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="train")
    if history["val"]:
        ax.plot(epochs, history["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved loss curves    -> {out}")


def _plot_pred_vs_actual(mlp: MLPValuationHead, X_val: np.ndarray, y_val: np.ndarray, out: Path) -> None:
    feat = torch.tensor(X_val, dtype=torch.float32)
    mlp._net.eval()
    with torch.no_grad():
        y_pred = mlp._net(feat).numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_val, y_pred, alpha=0.3, s=5, rasterized=True)
    lo = min(float(y_val.min()), float(y_pred.min()))
    hi = max(float(y_val.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual  expected_reward")
    ax.set_ylabel("Predicted")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved pred vs actual -> {out}")


def train_mlp(
    scenario_id: str = SCENARIO_ID,
    scenario_overrides: Optional[dict] = None,
    n_episodes: int = 500,
    val_frac: float = 0.15,
    hidden_dims: Optional[List[int]] = None,
    activation: str = "tanh",
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    save_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> MLPValuationHead:
    """Train, save, and return a MLPValuationHead for the given scenario."""
    if hidden_dims is None:
        hidden_dims = [64, 64]
    out_dir = Path(save_dir) if save_dir else MLP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dims_str = "-".join(str(d) for d in hidden_dims)
    stem = f"{scenario_id}_h{dims_str}_act{activation}_ep{n_episodes}_e{n_epochs}_lr{lr:g}_s{seed}"
    tb_dir = Path(log_dir) if log_dir else MLP_LOGS_DIR / scenario_id / stem
    tb_dir.mkdir(parents=True, exist_ok=True)

    scenario = dac.make(scenario_id, **(scenario_overrides or {}))

    print("Collecting rollouts ...")
    X, y = collect_rollouts(scenario, n_episodes=n_episodes, seed=seed)

    n_val = max(1, int(len(X) * val_frac))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_train, y_train = X[idx[n_val:]], y[idx[n_val:]]
    X_val,   y_val   = X[idx[:n_val]], y[idx[:n_val]]
    print(f"  {len(X_train)} train samples, {len(X_val)} val samples")

    print("Training ...")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    mlp = MLPValuationHead(input_dim=X.shape[1], hidden_dims=hidden_dims, activation=activation)
    mlp.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
        log_dir=str(tb_dir),
    )

    mlp.save(out_dir / f"{stem}.pt")
    print(f"  Saved weights        -> {out_dir / f'{stem}.pt'}")

    meta = {
        "scenario_id":       scenario_id,
        "scenario_overrides": scenario_overrides or {},
        "n_episodes":        n_episodes,
        "val_frac":          val_frac,
        "hidden_dims":       hidden_dims,
        "activation":        activation,
        "n_epochs":          n_epochs,
        "lr":                lr,
        "batch_size":        batch_size,
        "seed":              seed,
        "n_train_samples":   len(X_train),
        "n_val_samples":     len(X_val),
        "final_train_loss":  mlp.history_["train"][-1],
        "final_val_loss":    mlp.history_["val"][-1] if mlp.history_["val"] else None,
        "trained_at":        datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved metadata       -> {out_dir / f'{stem}.json'}")

    _plot_loss_curves(mlp.history_, out_dir / f"{stem}_loss.pdf")
    _plot_pred_vs_actual(mlp, X_val, y_val, out_dir / f"{stem}_pred.pdf")

    return mlp


def load_mlp(scenario_id: str = SCENARIO_ID, path: Optional[str] = None) -> MLPValuationHead:
    """Load a saved MLPValuationHead.

    If *path* is given, loads from that file directly.  Otherwise scans
    ``MLP_DIR`` for ``.pt`` files matching ``<scenario_id>_*.pt``:
    - exactly one match -> loaded automatically
    - multiple matches  -> raises, listing candidates (use --mpc-model to pick one)
    - no match          -> raises, suggesting python -m benchmark.train_mlp
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No MLPValuationHead at {p}.")
    else:
        candidates = sorted(MLP_DIR.glob(f"{scenario_id}_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No MLPValuationHead found for scenario '{scenario_id}' in {MLP_DIR}.\n"
                "Train one first:  python -m benchmark.train_mlp"
            )
        if len(candidates) > 1:
            listing = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Multiple MLPValuationHead models found for '{scenario_id}':\n  {listing}\n"
                "Pass --mpc-models <path> to select one."
            )
        p = candidates[0]

    mlp = MLPValuationHead.load(p)
    print(f"  Loaded MLPValuationHead <- {p}  (hidden_dims={mlp.hidden_dims}, input_dim={mlp.input_dim})")
    return mlp


def _start_tensorboard(log_dir: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    print(f"\n  TensorBoard running -> http://localhost:{port}")
    print(  "  (open in your browser; updates every ~30 s during training)\n")
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLPValuationHead for MPCPolicy.")
    parser.add_argument("--n-episodes",  type=int,   default=500,         help="Rollout episodes for data collection (default: 500).")
    parser.add_argument("--val-frac",    type=float, default=0.15,        help="Fraction of samples held out for validation (default: 0.15).")
    parser.add_argument("--hidden-dims", type=int,   nargs="+", default=[64, 64], help="Hidden layer widths (default: 64 64).")
    parser.add_argument("--activation",  type=str,   default="tanh",
                        choices=["tanh", "relu", "leaky_relu"],
                        help="Hidden layer activation (default: tanh).")
    parser.add_argument("--n-epochs",    type=int,   default=500,         help="Training epochs (default: 500).")
    parser.add_argument("--lr",          type=float, default=1e-3,        help="Adam learning rate (default: 1e-3).")
    parser.add_argument("--batch-size",  type=int,   default=256,         help="Mini-batch size (default: 256).")
    parser.add_argument("--seed",        type=int,   default=42,           help="Base random seed (default: 42).")
    parser.add_argument("--save-dir",    type=str,   default=None,        help="Override output directory.")
    parser.add_argument("--log-dir",     type=str,   default=None,        help="Override TensorBoard log directory.")
    parser.add_argument("--tensorboard", action="store_true",             help="Launch TensorBoard in the background during training.")
    parser.add_argument("--port",        type=int,   default=6006,        help="TensorBoard port (default: 6006).")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else MLP_LOGS_DIR

    tb_proc: Optional[subprocess.Popen] = None
    if args.tensorboard:
        tb_proc = _start_tensorboard(log_dir, args.port)

    def _cleanup(sig=None, frame=None):
        if tb_proc is not None and tb_proc.poll() is None:
            print("\nStopping TensorBoard ...")
            tb_proc.terminate()
        sys.exit(0)

    if tb_proc is not None:
        signal.signal(signal.SIGINT,  _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

    try:
        train_mlp(
            n_episodes=args.n_episodes,
            val_frac=args.val_frac,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
        )
    finally:
        if tb_proc is not None and tb_proc.poll() is None:
            print(f"\nTraining finished.  TensorBoard still running at http://localhost:{args.port}")
            print("Press Ctrl-C to stop it.")
            tb_proc.wait()


if __name__ == "__main__":
    main()
