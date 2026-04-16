import numpy as np
from dynamicauc.primitives.display_sampler import IIDUniformSampler, CatalogueSampler, FixedSampler

def test_iid_uniform_sampler():
    rng = np.random.default_rng(42)
    sampler = IIDUniformSampler(dim=3)
    samples = [sampler.sample(rng) for _ in range(1000)]
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    assert np.all((0.3 <= mean) & (mean <= 0.7))
    assert np.all((0.25 <= std) & (std <= 0.35))
    assert all(np.all((0.0 <= s) & (s <= 1.0)) for s in samples)
    assert sampler.dim == 3
    assert samples[0].shape == (3,)

def test_catalogue_sampler():
    rng = np.random.default_rng(42)
    creatives = [[1.0, 0.0], [0.0, 1.0]]
    sampler = CatalogueSampler(creatives=creatives)
    samples = [sampler.sample(rng) for _ in range(1000)]
    counts = np.zeros((2,))
    for s in samples:
        if np.array_equal(s, [1.0, 0.0]):
            counts[0] += 1
        elif np.array_equal(s, [0.0, 1.0]):
            counts[1] += 1
        else:
            assert False, "Sample not in catalogue"
    assert np.all((0.4 <= counts / 1000) & (counts / 1000 <= 0.6))
    assert sampler.dim == 2
    assert samples[0].shape == (2,)

def test_fixed_sampler():
    rng = np.random.default_rng(42)
    sampler = FixedSampler(x=[0.5, 0.2, 0.8])
    x = sampler.sample(rng)
    assert np.array_equal(x, np.array([0.5, 0.2, 0.8]))
