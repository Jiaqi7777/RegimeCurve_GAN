from __future__ import annotations

import numpy as np

from regimecurve.evaluate import conditional_neighbours, curve_factors

COLUMNS = [
    "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr",
    "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]


def test_curve_factors():
    curves = np.zeros((2, 13))
    curves[:, 6], curves[:, 8], curves[:, 10] = 3, 4, 5
    np.testing.assert_allclose(curve_factors(curves, COLUMNS)[0], [5, 2, 0])


def test_neighbours_do_not_overlap():
    rng = np.random.default_rng(4)
    hist = 4 + np.cumsum(rng.normal(0, .01, (300, 13)), axis=0)
    starts, distances, paths = conditional_neighbours(hist, COLUMNS, 10, neighbours=15)
    assert starts.shape == distances.shape == (15,)
    assert paths.shape == (15, 10, 3)
    assert min(abs(int(a) - int(b)) for i, a in enumerate(starts) for b in starts[i + 1:]) >= 10
