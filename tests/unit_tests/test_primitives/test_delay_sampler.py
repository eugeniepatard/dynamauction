"""Tests for dynamauction/primitives/delay_sampler.py: SingleConversionSampler, MultiConversionSampler."""
import numpy as np
from unittest.mock import MagicMock

from dynamauction.primitives.delay_sampler import MultiConversionSampler, SingleConversionSampler


def _make_probe(return_value):
    """Return a mock ConversionProbe with configurable sample_one / sample_all outputs."""
    from dynamauction.primitives.conversion_probe import ConversionProbe
    probe = MagicMock(spec=ConversionProbe)
    probe.sample_one.return_value = return_value
    probe.sample_all.return_value = return_value
    return probe


_KWARGS = dict(
    u=np.zeros(2),
    x=np.zeros(2),
    ctx=0.0,
    y_func=lambda t: 0.0,
    t=0.0,
    horizon=5.0,
    rng=np.random.default_rng(0),
)


class TestSingleConversionSampler:

    def test_delegates_to_probe_sample_one(self):
        probe = _make_probe([1.5])
        sampler = SingleConversionSampler()
        sampler.sample(probe, **_KWARGS)
        probe.sample_one.assert_called_once()
        probe.sample_all.assert_not_called()

    def test_returns_probe_output(self):
        probe = _make_probe([2.3])
        result = SingleConversionSampler().sample(probe, **_KWARGS)
        assert result == [2.3]

    def test_returns_empty_list_when_probe_returns_empty(self):
        probe = _make_probe([])
        result = SingleConversionSampler().sample(probe, **_KWARGS)
        assert result == []

    def test_result_is_list(self):
        probe = _make_probe([0.5])
        result = SingleConversionSampler().sample(probe, **_KWARGS)
        assert isinstance(result, list)


class TestMultiConversionSampler:

    def test_delegates_to_probe_sample_all(self):
        probe = _make_probe([0.5, 1.0, 2.0])
        MultiConversionSampler().sample(probe, **_KWARGS)
        probe.sample_all.assert_called_once()
        probe.sample_one.assert_not_called()

    def test_returns_probe_output(self):
        probe = _make_probe([0.5, 1.0, 2.0])
        result = MultiConversionSampler().sample(probe, **_KWARGS)
        assert result == [0.5, 1.0, 2.0]

    def test_returns_empty_list_when_probe_returns_empty(self):
        probe = _make_probe([])
        result = MultiConversionSampler().sample(probe, **_KWARGS)
        assert result == []

    def test_result_is_list(self):
        probe = _make_probe([1.0])
        result = MultiConversionSampler().sample(probe, **_KWARGS)
        assert isinstance(result, list)


class TestSamplerContrast:

    def test_single_uses_sample_one_multi_uses_sample_all(self):
        single_probe = _make_probe([0.7])
        multi_probe = _make_probe([0.7, 1.5])

        SingleConversionSampler().sample(single_probe, **_KWARGS)
        MultiConversionSampler().sample(multi_probe, **_KWARGS)

        single_probe.sample_one.assert_called_once()
        single_probe.sample_all.assert_not_called()
        multi_probe.sample_all.assert_called_once()
        multi_probe.sample_one.assert_not_called()
