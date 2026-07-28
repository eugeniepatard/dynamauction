"""
Print the best hyperparameters from a PPO hyperparameter search.

Usage:
    python benchmark/ppo_select_best.py --scenario immediate
    python benchmark/ppo_select_best.py --scenario recency
    python benchmark/ppo_select_best.py --scenario mixed

Output (bash-sourceable):
    BEST_TRIAL=042
    BEST_ENT_COEF=0.05
    BEST_LR=0.0003
    BEST_N_STEPS=1024
    BEST_GAE_LAMBDA=0.95
    BEST_REWARD=1.234
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SEARCH_DIRS = {
    "survival":  "/scratch/epatard/dynamicauc/ppo_hparam_search_survival",
    "immediate": "/scratch/epatard/dynamicauc/ppo_hparam_search_immediate",
    "recency":   "/scratch/epatard/dynamicauc/ppo_hparam_search_recency",
    "mixed":     "/scratch/epatard/dynamicauc/ppo_hparam_search_mixed",
}


def load_results(search_dir: Path) -> list[dict]:
    results = []
    for f in sorted(search_dir.glob("trial_*.json")):
        try:
            data = json.loads(f.read_text())
            results.append(data)
        except Exception as e:
            print(f"# Warning: could not load {f}: {e}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SEARCH_DIRS))
    parser.add_argument("--top", type=int, default=5, help="Print top-N trials")
    args = parser.parse_args()

    search_dir = Path(SEARCH_DIRS[args.scenario])
    if not search_dir.exists():
        print(f"# ERROR: search dir not found: {search_dir}")
        raise SystemExit(1)

    results = load_results(search_dir)
    if not results:
        print(f"# ERROR: no trial JSON files in {search_dir}")
        raise SystemExit(1)

    results.sort(key=lambda r: r["metrics"]["mean_reward"], reverse=True)

    print(f"# {len(results)} trials loaded from {search_dir}")
    print(f"# Top-{min(args.top, len(results))} trials:")
    for r in results[: args.top]:
        hp = r["hparams"]
        print(
            f"#   trial={r['trial_id']:03d}  reward={r['metrics']['mean_reward']:.4f}"
            f"  ent_coef={hp['ent_coef']}  lr={hp['learning_rate']:.0e}"
            f"  n_steps={hp['n_steps']}  gae={hp['gae_lambda']}"
        )

    best = results[0]
    hp = best["hparams"]
    print()
    print(f"BEST_TRIAL={best['trial_id']:03d}")
    print(f"BEST_ENT_COEF={hp['ent_coef']}")
    print(f"BEST_LR={hp['learning_rate']}")
    print(f"BEST_N_STEPS={hp['n_steps']}")
    print(f"BEST_GAE_LAMBDA={hp['gae_lambda']}")
    print(f"BEST_REWARD={best['metrics']['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
