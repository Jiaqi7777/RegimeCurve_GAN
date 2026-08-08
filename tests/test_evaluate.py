from __future__ import annotations

import numpy as np

from regimecurve.evaluate import conditional_neighbours, curve_factors

COLUMNS = [
    "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr",
    "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]


def test_curve_factors_have_expected_values():
    curves = np.zeros((2, 13))
    curves[:, 6] = 3.0
    curves[:, 8] = 4.0
    curves[:, 10] = 5.0
    factors = curve_factors(curves, COLUMNS)
    np.testing.assert_allclose(factors[0], [5.0, 2.0, 0.0])


def test_conditional_neighbours_return_complete_paths():
    rng = np.random.default_rng(7)
    hist = 4.0 + np.cumsum(rng.normal(0, 0.01, size=(100, 13)), axis=0)
    starts, distances, paths = conditional_neighbours(
        hist, COLUMNS, horizon=10, context_days=3, neighbours=20
    )
    assert starts.shape == distances.shape == (20,)
    assert paths.shape == (20, 10, 3)
    assert np.isfinite(paths).all()
