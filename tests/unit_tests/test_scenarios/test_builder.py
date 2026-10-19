import dynamauction as dac

def test_make_preset():
    scenario = dac.make("saturation-second-price-lognormal")
    assert scenario is not None
    assert scenario.kernel is not None
    assert scenario.auction is not None
    assert scenario.max_history == 10

def test_scenario_builder():
    scenario = (
        dac.ScenarioBuilder()
        .with_auction("first-price")
        .with_horizon("fixed", value=20.0)
        .build()
    )
    assert scenario is not None
    assert scenario.auction.__class__.__name__ == "FirstPriceAuction"
    assert scenario.horizon_generator.__class__.__name__ == "FixedHorizon"

def test_builder_defaults_probe_and_delay_sampler():
    scenario = dac.ScenarioBuilder().build()
    assert scenario.probe.__class__.__name__ == "PoissonProbe"
    assert scenario.delay_sampler.__class__.__name__ == "SingleConversionSampler"

def test_builder_with_poisson_probe_and_single_sampler():
    scenario = (
        dac.ScenarioBuilder()
        .with_kernel("saturation", base_rate=0.5)
        .with_probe("poisson")
        .with_delay_sampler("single")
        .build()
    )
    assert scenario.probe.__class__.__name__ == "PoissonProbe"
    assert scenario.delay_sampler.__class__.__name__ == "SingleConversionSampler"

def test_builder_with_bernoulli_probe_and_multi_sampler():
    import numpy as np
    scenario = (
        dac.ScenarioBuilder()
        .with_kernel("survival", w_u=np.ones(3), w_x=np.ones(3), delay_rate=1.0)
        .with_probe("bernoulli")
        .with_delay_sampler("multi")
        .build()
    )
    assert scenario.probe.__class__.__name__ == "BernoulliProbe"
    assert scenario.delay_sampler.__class__.__name__ == "MultiConversionSampler"

def test_make_survival_second_price_exponential():
    scenario = dac.make("survival-second-price-exponential-1", horizon_eta=50)
    assert scenario is not None
    assert scenario.kernel.__class__.__name__ == "SurvivalKernel"
    assert scenario.max_history >= 1