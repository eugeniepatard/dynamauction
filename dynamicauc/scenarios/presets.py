"""
Registry of preset scenarios.
"""

SCENARIO_PRESETS = {
    "saturation-second-price-uniform": {
        "description": "Saturation kernel with second-price auction and uniform competition",
        "kernel": {"name": "saturation", "params": {"base_rate": 0.5, "saturation_scale": 10.0, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "uniform", "params": {"low": 0.0, "high": 1.0}},
        "arrival": {"name": "poisson", "params": {"breakpoints": [5.0], "intensities": [0.1, 1.2]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 50.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "max_history": 10,
    },
    "recency-second-price-lognormal": {
        "description": "Recency kernel with second-price auction and lognormal competition",
        "kernel": {"name": "recency", "params": {"alpha": 0.5, "delay_rate": 0.5}},
        "auction": {"name": "second-price"},
        "competition": {"name": "lognormal", "params": {"mu": 0.0, "sigma": 0.5}},
        "arrival": {"name": "poisson", "params": {"breakpoints": [], "intensities": [1.0]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 30.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 3}},
        "max_history": 10,
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
        "arrival": {"name": "poisson", "params": {"breakpoints": [], "intensities": [0.5]}},
        "display": {"name": "iid", "params": {"dim": 3}},
        "exogenous": {"name": "trivial"},
        "horizon": {"name": "fixed", "params": {"value": 20.0}},
        "user": {"name": "uniform", "params": {"full_dim": 3, "observable_dim": 2}},
        "max_history": 10,
    },
}