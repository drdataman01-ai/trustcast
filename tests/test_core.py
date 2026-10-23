import numpy as np
import pytest

from trustcast.evaluation import metrics
from trustcast.evaluation.backtest import rolling_origin
from trustcast.models import LicenseError, get_model
from trustcast.models.classical import SeasonalNaive
from trustcast.uncertainty import calibration, conformal, ensemble

Q = [0.05, 0.1, 0.5, 0.9, 0.95]


def series(n=120):
    t = np.arange(n)
    rng = np.random.default_rng(0)
    return 100 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n)


def test_seasonal_naive_shapes():
    fc = SeasonalNaive().predict(series(), 6, Q, freq="M")
    assert fc.mean.shape == (6,)
    lo, hi = fc.interval(0.9)
    assert np.all(lo <= hi)


def test_backtest_runs():
    df = rolling_origin(SeasonalNaive(), series(), horizon=3, n_origins=10,
                        quantiles=Q, freq="M", m=12)
    assert len(df) == 10
    assert df["mase"].notna().all()


def test_metrics():
    assert metrics.coverage([1, 2, 3], [0, 0, 0], [2, 2, 2]) == pytest.approx(2 / 3)
    assert metrics.pinball([1.0], [1.0], 0.5) == 0


def test_conformal_widens_when_too_narrow():
    fc = SeasonalNaive().predict(series(), 3, Q, freq="M")
    adjusted = conformal.adjust(fc, np.array([2.0, 3.0, 4.0]), 0.9)
    lo0, hi0 = fc.interval(0.9)
    lo1, hi1 = adjusted.interval(0.9)
    assert np.all(lo1 < lo0) and np.all(hi1 > hi0)


def test_ensemble_and_calibration():
    fc = SeasonalNaive().predict(series(), 3, Q, freq="M")
    ens = ensemble.quantile_average([fc, fc])
    assert np.allclose(ens.mean, fc.mean)
    rep = calibration.coverage_report(np.array([1, 5]), np.array([0, 0]), np.array([2, 2]), 0.9)
    assert rep["status"] == "too narrow"


def test_license_guard():
    with pytest.raises(LicenseError):
        get_model("moirai-2.0", mode="commercial")


def test_data_store_roundtrip(tmp_path, monkeypatch):
    import pandas as pd

    from trustcast.data import store

    monkeypatch.setenv("TRUSTCAST_DATA", str(tmp_path))
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    store.save_series(s, "demo")
    store.save_series(s * 2, "client", private=True)
    assert list(store.load_series("demo").values) == [1.0, 2.0, 3.0]
    assert list(store.load_series("client").values) == [2.0, 4.0, 6.0]
    assert store.available() == ["demo", "private/client"]
    with pytest.raises(FileNotFoundError):
        store.load_series("missing")


def _yahoo_like():
    import pandas as pd

    idx = pd.bdate_range("2021-01-04", periods=400)
    rng = np.random.default_rng(1)
    cols = {}
    for t in ["AAA", "HALT", "SHORTGAP"]:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        if t == "HALT":
            px[100:120] = np.nan
        if t == "SHORTGAP":
            px[50:52] = np.nan
        for f in ["Close", "Volume"]:
            cols[(f, t)] = px
    raw = pd.DataFrame(cols, index=idx)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw


def test_etl_and_prepare_stocks():
    from trustcast.data import etl
    from trustcast.data.prepare import prepare
    from trustcast.data.sources import yahoo

    long = etl.standardize(yahoo.to_long(_yahoo_like(), "Close"))
    assert set(long.columns) == {"series_id", "date", "value"}
    prepared = prepare(long, {"align": True, "fill_limit": 5, "transform": "log_return", "min_obs": 100})
    status = prepared.report.set_index("series_id")["status"]
    assert status["HALT"] == "dropped"          # 20-day gap is too long
    assert status["SHORTGAP"] == "kept"         # 2-day gap is filled
    # inverting log returns gives back price levels
    r = prepared.series["AAA"].to_numpy()[-5:]
    levels = prepared.to_levels("AAA", r)
    assert np.allclose(levels, prepared.last_level["AAA"] * np.exp(np.cumsum(r)))


def test_prepare_keeps_mixed_frequencies():
    import pandas as pd

    from trustcast.data.prepare import prepare

    m = pd.DataFrame({"series_id": "cpi", "date": pd.date_range("2015-01-01", periods=60, freq="MS"),
                      "value": np.arange(60.0)})
    q = pd.DataFrame({"series_id": "gdp", "date": pd.date_range("2015-01-01", periods=40, freq="QS"),
                      "value": np.arange(40.0)})
    prepared = prepare(pd.concat([m, q]), {"fill": "none", "min_obs": 32})
    assert len(prepared.series["gdp"]) == 40 and len(prepared.series["cpi"]) == 60
