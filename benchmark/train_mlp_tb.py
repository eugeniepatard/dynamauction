"""
Train the MLPValuationHead and monitor training live in TensorBoard.

Launches TensorBoard in the background before training starts, so you
can open http://localhost:6006 in your browser and watch the loss curves
update in real time.  TensorBoard keeps running after training ends so
you can inspect the final curves; press Ctrl-C to stop it.

Usage (from repo root)
----------------------
    python benchmark/train_mlp_tb.py
    python benchmark/train_mlp_tb.py --n-episodes 500 --n-epochs 500 --hidden-dims 128 128
    python benchmark/train_mlp_tb.py --port 6007   # if 6006 is already in use
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.train_mlp import train_mlp, MLP_LOGS_DIR


def _start_tensorboard(log_dir: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Give TensorBoard a moment to start before printing the URL.
    time.sleep(2)
    print(f"\n  TensorBoard running → http://localhost:{port}")
    print(  "  (open in your browser; updates every ~30 s during training)\n")
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-episodes",  type=int,         default=500)
    parser.add_argument("--val-frac",    type=float,       default=0.2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--activation",  type=str, default="tanh",
                        choices=["tanh", "relu", "leaky_relu"],
                        help="Hidden layer activation (default: tanh).")
    parser.add_argument("--n-epochs",    type=int,         default=500)
    parser.add_argument("--lr",          type=float,       default=1e-3)
    parser.add_argument("--batch-size",  type=int,         default=256)
    parser.add_argument("--seed",        type=int,         default=42)
    parser.add_argument("--save-dir",    type=str,         default=None)
    parser.add_argument("--log-dir",     type=str,         default=None)
    parser.add_argument("--port",        type=int,         default=6006,
                        help="TensorBoard port (default: 6006).")
    parser.add_argument("--no-tb",       action="store_true",
                        help="Skip launching TensorBoard (training only).")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else MLP_LOGS_DIR

    tb_proc: Optional[subprocess.Popen] = None
    if not args.no_tb:
        tb_proc = _start_tensorboard(log_dir, args.port)

    def _cleanup(sig=None, frame=None):
        if tb_proc is not None and tb_proc.poll() is None:
            print("\nStopping TensorBoard …")
            tb_proc.terminate()
        sys.exit(0)

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
            print("\nTraining finished.  TensorBoard still running at "
                  f"http://localhost:{args.port}")
            print("Press Ctrl-C to stop it.")
            tb_proc.wait()


if __name__ == "__main__":
    main()
