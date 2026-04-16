import dynamicauc as dac

def test_make_preset():
    scenario = dac.make("saturation-second-price-uniform")
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