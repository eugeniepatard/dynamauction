"""
Registry of preset scenarios.
"""

SCENARIO_PRESETS = {
    "survival-second-price-exponential": {
        "description": "Survival kernel with single conversion per display, second-price and lognormal competition",
        "kernel": {"name": "survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "delay_rate": 1.0}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "multiplicative", "params": {"c" : 0.9 , "initial_value": 1.0, "beta": 1.0}},
        "max_history": 1,
        "probe": {"name": "bernoulli"},
        "delay_sampler": {"name": "single"},
        "kappa_distribution": "poisson",
    },

    "survival-second-price-exponential-2": {
            "description": "Survival kernel, second-price, lower lognormal competition (mu=-1, sigma=1), recovery rate beta=1",
            "kernel": {"name": "survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "delay_rate": 1.0}},
            "auction": {"name": "second-price"},
            "competition": {"name": "lognormal", "params": {"mu": -1.0, "sigma": 1.0}},
            "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
            "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
            "exogenous": {"name": "trivial"},
            "horizon": {"name": "exponential", "params": {"eta": 50.0}},
            "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
            "gamma": {"name": "multiplicative", "params": {"c" : 0.9 , "initial_value": 1.0, "beta": 1.0}},
            "max_history": 1,
            "probe": {"name": "bernoulli"},
            "delay_sampler": {"name": "single"},
            "kappa_distribution": "poisson",
        },

    "saturation-second-price-lognormal": {
        "description": "Saturation kernel with second-price auction and lognormal competition (mu=-1, sigma=1)",
        "kernel": {"name": "saturation", "params": {"base_rate": 1.0, "saturation_scale": 10.0, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": -1.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "additive"},
        "max_history": 10,
        "probe": {"name": "bernoulli"},
        "delay_sampler": {"name": "single"},
        "kappa_distribution": "poisson",
    },
    "recency-second-price-lognormal": {
        "description": "Recency kernel with second-price auction and lognormal competition (mu=-1, sigma=1)",
        "kernel": {"name": "recency", "params": {"recency_coef": 0.5, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": -1.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        # time of the last won auction; -1000 before any win (conversion prob 1 - exp(-0.5 (t + 1000)) = 1), kept on a
        # scale close to the real win times so that MLP inputs and VecNormalize statistics are not swamped by -1e6
        "gamma": {"name": "last-won-time", "params": {"initial_value": -1000.0}},
        "max_history": 10,
        "probe": {"name": "bernoulli"},
        "delay_sampler": {"name": "single"},
        "kappa_distribution": "poisson",
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
        "kappa_distribution": "poisson",
    },

    # ── Immediate-reward variant of the main survival scenario ───────────────────
    # Same conversion probability as survival-second-price-exponential but reward
    # is realized at bid time (kappa=f, lambda_=0).  Used to isolate whether
    # delayed credit assignment is the bottleneck for RL agents.

    # Scenario 2 with immediate rewards (diagnostic for RL: no delayed credit assignment; same fatigue dynamics,
    # competition LogNormal(-1, 1), beta = 1). The conversion happens at bid time with probability f, so V0 = f.
    "survival-second-price-exponential-2-immediate": {
        "description": "Immediate-reward survival kernel, scenario 2 (competition LogNormal(-1, 1), beta=1) — conversion at bid time, no delay",
        "kernel": {"name": "immediate-survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "delay_rate": 1.0}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": -1.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "multiplicative", "params": {"c": 0.9, "initial_value": 1.0, "beta": 1.0}},
        "max_history": 1,
        "probe": {"name": "null"},
        "delay_sampler": {"name": "single"},
        "kappa_distribution": "bernoulli",
    },

    "survival-second-price-exponential-immediate": {
        "description": "Immediate-reward survival kernel — same conversion prob but reward at bid time, no delay",
        "kernel": {"name": "immediate-survival", "params": {"w_u": [1.0, 1.0, 1.0], "w_x": [1.0, 1.0, 1.0], "delay_rate": 1.0}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 1.0}},
        "arrival": {"name": "cstpoisson", "params": {"rate": 1.0}},
        "display": {"name": "normal", "params": {"dim": 3, "mu": 0.0, "sigma": 1.0}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "exponential", "params": {"eta": 50.0}},
        "user": {"name": "normal", "params": {"full_dim": 3, "observable_dim": 3, "mu": 0.0, "sigma": 1.0}},
        "gamma": {"name": "multiplicative", "params": {"c": 0.9, "initial_value": 1.0, "beta": 1.0}},
        "max_history": 1,
        "probe": {"name": "null"},
        "delay_sampler": {"name": "single"},
        "kappa_distribution": "bernoulli",
    }
}