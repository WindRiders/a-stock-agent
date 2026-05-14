"""测试 ML 择时模型。"""

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ml_model import MLTimingModel, FeatureEngineer, XGBoostModel, MLPrediction


def _make_kline(n_days=400):
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 10.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.normal(0, 0.005, n_days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_days),
    })


class TestFeatureEngineer:
    def setup_method(self):
        self.kline = _make_kline(400)

    def test_build_features(self):
        X, y = FeatureEngineer.build_features(self.kline, horizon=5)
        assert isinstance(X, pd.DataFrame)
        assert X.shape[1] >= 10
        for val in y.unique():
            assert val in (0, 1, 2)

    def test_small_data(self):
        kline = _make_kline(80)
        X, _ = FeatureEngineer.build_features(kline, horizon=5)
        assert len(X) > 0

    def test_feature_names(self):
        assert len(FeatureEngineer.get_feature_names()) >= 20


class TestMLTimingModel:
    def setup_method(self):
        self.kline = _make_kline(400)

    def test_train_no_crash(self):
        model = MLTimingModel(model_type="xgboost")
        model.train(self.kline, horizon=5, verbose=False)

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="xgboost not installed")
    def test_predict_after_train(self):
        model = MLTimingModel(model_type="xgboost")
        model.train(self.kline, horizon=5, verbose=False)
        pred = model.predict(self.kline)
        assert pred.direction in ("UP", "DOWN", "FLAT")

    def test_predict_untrained(self):
        model = MLTimingModel(model_type="xgboost")
        pred = model.predict(self.kline)
        assert pred.direction is not None

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="xgboost not installed")
    def test_backtest(self):
        model = MLTimingModel(model_type="xgboost")
        result = model.backtest(self.kline, horizon=5)
        assert result.accuracy >= 0

    def test_backtest_small(self):
        model = MLTimingModel(model_type="xgboost")
        result = model.backtest(_make_kline(100), horizon=5)
        assert result is not None

    def test_report_empty(self):
        model = MLTimingModel()
        assert "训练" in model.generate_report()

    def test_different_horizons(self):
        model = MLTimingModel()
        model.train(self.kline, horizon=3, verbose=False)
        assert model._horizon == 3


class TestXGBoostModel:
    def setup_method(self):
        self.kline = _make_kline(400)
        X, y = FeatureEngineer.build_features(self.kline, horizon=5)
        valid = X.notna().all(axis=1)
        self.X = X[valid]
        self.y = y[valid]

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="xgboost not installed")
    def test_train_and_predict(self):
        xgb = XGBoostModel(n_estimators=50, max_depth=3)
        xgb.train(self.X, self.y, verbose=False)
        preds = xgb.predict(self.X)
        assert len(preds) == len(self.X)

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="xgboost not installed")
    def test_predict_proba(self):
        xgb = XGBoostModel(n_estimators=50, max_depth=3)
        xgb.train(self.X, self.y, verbose=False)
        proba = xgb.predict_proba(self.X)
        assert proba.shape == (len(self.X), 3)

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="xgboost not installed")
    def test_feature_importance(self):
        xgb = XGBoostModel(n_estimators=50, max_depth=3)
        xgb.train(self.X, self.y, verbose=False)
        assert len(xgb.feature_importance) > 0

    def test_predict_untrained_default(self):
        xgb = XGBoostModel()
        preds = xgb.predict(self.X)
        assert len(preds) == len(self.X)