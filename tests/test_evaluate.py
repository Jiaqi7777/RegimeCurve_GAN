from __future__ import annotations

import numpy as np

from regimecurve.evaluate import (
    conditional_neighbours,
    curve_factors,
    nearest_neighbour_percentiles,
)

COLUMNS = [
    "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr",
    "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]


def test_curve_factors_have_expected_values():
    curves = np.zeros((2, 13))
    curves[:, 6], curves[:, 8], curves[:, 10] = 3.0, 4.0, 5.0
    np.testing.assert_allclose(curve_factors(curves, COLUMNS)[0], [5.0, 2.0, 0.0])


def test_conditional_neighbours_are_non_overlapping():
    rng = np.random.default_rng(7)
    hist = 4.0 + np.cumsum(rng.normal(0, 0.01, size=(200, 13)), axis=0)
    starts, distances, paths = conditional_neighbours(hist, COLUMNS, 10, neighbours=10)
    assert starts.shape == distances.shape == (10,)
    assert paths.shape == (10, 10, 3)
    assert min(abs(int(a) - int(b)) for i, a in enumerate(starts) for b in starts[i + 1:]) >= 10


def test_nearest_neighbour_percentiles_detect_outlier():
    rng = np.random.default_rng(3)
    historical = rng.normal(size=(100, 13))
    generated = np.vstack([historical[0], np.full(13, 50.0)])
    percentiles = nearest_neighbour_percentiles(historical, generated)
    assert percentiles.shape == (2,)
    assert percentiles[1] > 0.95
