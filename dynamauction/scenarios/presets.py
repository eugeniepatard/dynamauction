"""
Registry of preset scenarios.
"""

SCENARIO_PRESETS = {
    "survival-second-price-exponential": {
        "description": "Survival kernel with single conversion per display, second-price and lognormal competition",
        "kernel": {"name": "survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "alpha": 1.0}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "multiplicative", "params": {"c" : 0.9 , "initial_value": 1.0}},
        "max_history": 1,
        "probe": {"name": "survival"},
        "delay_sampler": {"name": "single"},
    },

    "saturation-second-price-uniform": {
        "description": "Saturation kernel with second-price auction and uniform competition",
        "kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "uniform", "params": {"low": 0.0, "high": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 50.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "gamma": {"name": "additive"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },
    "recency-second-price-lognormal": {
        "description": "Recency kernel with second-price auction and lognormal competition",
        "kernel": {"name": "recency", "params": {"alpha": 0.5, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}},
        "arrival": {"name": "pwpoisson", "params": {"breakpoints": [], "intensities": [1.0]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 30.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "gamma": {"name": "last-won-time"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },
    "mixed-second-price-exponential": {
        "description": "Mixed kernel (immediate + delayed) with second-price and exponential competition",
        "kernel": {
            "name": "mixed",
            "params": {
                "base_kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.1}},
                "kappa_base_rate": 0.2,
            },
        },
        "auction": {"name": "second-price"},
        "competition": {"name": "exponential", "params": {"scale": 1.0}},
        "arrival": {"name": "pwpoisson", "params": {"breakpoints": [], "intensities": [0.5]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 20.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 2}},
        "gamma": {"name": "additive"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },

    # ── Exponential-horizon variants (eta=50) for robustness comparison ────────
    # Same kernels/competition as above but with ExponentialHorizon(eta=50)
    # so results are directly comparable to the survival scenario.

    "saturation-second-price-uniform-eta50": {
        "description": "Saturation kernel, second-price, uniform competition, exponential horizon eta=50",
        "kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "uniform", "params": {"low": 0.0, "high": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "gamma": {"name": "additive"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },

    "recency-second-price-lognormal-eta50": {
        "description": "Recency kernel, second-price, lognormal competition, exponential horizon eta=50",
        "kernel": {"name": "recency", "params": {"alpha": 0.5, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}},
        "arrival": {"name": "pwpoisson", "params": {"breakpoints": [], "intensities": [1.0]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "gamma": {"name": "last-won-time"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },

    # ── Immediate-reward variant of the main survival scenario ───────────────────
    # Same conversion probability as survival-second-price-exponential but reward
    # is realized at bid time (kappa=f, lambda_=0).  Used to isolate whether
    # delayed credit assignment is the bottleneck for RL agents.

    "survival-second-price-exponential-immediate": {
        "description": "Immediate-reward survival kernel — same conversion prob but reward at bid time, no delay",
        "kernel": {"name": "immediate-survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "alpha": 1.0}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "multiplicative", "params": {"c": 0.9, "initial_value": 1.0}},
        "max_history": 1,
        "probe": {"name": "null"},
        "delay_sampler": {"name": "single"},
    },

    "mixed-second-price-exponential-eta50": {
        "description": "Mixed kernel, second-price, exponential competition, exponential horizon eta=50",
        "kernel": {
            "name": "mixed",
            "params": {
                "base_kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.1}},
                "kappa_base_rate": 0.2,
            },
        },
        "auction": {"name": "second-price"},
        "competition": {"name": "exponential", "params": {"scale": 1.0}},
        "arrival": {"name": "pwpoisson", "params": {"breakpoints": [], "intensities": [0.5]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 2}},
        "gamma": {"name": "additive"},
        "max_history": 10,
        "probe": {"name": "poisson"},
        "delay_sampler": {"name": "multi"},
    },
}