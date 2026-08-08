import numpy as np
import pandas as pd

from regimecurve.data import CurveTransform, load_curves, nelson_siegel_basis, valid_window_indices


def test_nelson_siegel_basis_is_finite():
    basis = nelson_siegel_basis(np.array([1 / 12, 1, 10, 30]), 0.7308)
    assert basis.shape == (4, 3)
    assert np.isfinite(basis).all()


def test_factor_round_trip_is_accurate():
    rng = np.random.default_rng(4)
    maturities = np.array([1 / 12, 0.25, 0.5, 1, 2, 5, 10, 20, 30])
    basis = nelson_siegel_basis(maturities, 0.7308)
    curves = rng.normal([4, -1, 0.5], 0.1, size=(100, 3)) @ basis.T
    curves += rng.normal(0, 0.005, curves.shape)
    transform = CurveTransform.fit(curves, maturities, 0.7308, 3)
    reconstructed = transform.decode(transform.encode(curves))
    assert np.mean(np.abs(curves - reconstructed)) < 0.01


def test_spline_residual_basis_is_orthogonal_to_nelson_siegel():
    rng = np.random.default_rng(7)
    maturities = np.array([1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    curves = rng.normal(size=(100, len(maturities)))
    transform = CurveTransform.fit(curves, maturities, 0.7308, 6)
    ns = nelson_siegel_basis(maturities, 0.7308)
    np.testing.assert_allclose(ns.T @ transform.residual_basis, 0.0, atol=1e-6)


def test_chronological_loading_and_interpolation(tmp_path):
    path = tmp_path / "curves.csv"
    pd.DataFrame({
        "Date": ["2024-01-03", "2024-01-02"],
        "1 Mo": [4.0, np.nan], "3 Mo": [4.1, 4.2], "6 Mo": [4.2, 4.3],
        "1 Yr": [4.3, 4.4], "2 Yr": [4.4, 4.5],
    }).to_csv(path, index=False)
    frame = load_curves(path)
    assert frame.index.is_monotonic_increasing
    assert not frame.isna().any().any()


def test_split_uses_target_end_date():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    indices = valid_window_indices(dates, 3, 10, None, "2020-01-20")
    assert all(dates[i + 9] <= pd.Timestamp("2020-01-20") for i in indices)
