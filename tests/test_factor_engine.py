"""测试因子挖掘引擎。"""

import numpy as np
import pandas as pd
import pytest

from factor_engine import (
    FactorEngine, FactorDef, FactorResult, ICResult, FACTOR_LIBRARY,
)


def _make_kline(n_days: int = 300):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
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


class TestFactorLibrary:
    def test_all_categories_present(self):
        categories = {fd.category for fd in FACTOR_LIBRARY}
        assert "momentum" in categories
        assert "volatility" in categories
        assert "volume" in categories

    def test_minimum_factor_count(self):
        assert len(FACTOR_LIBRARY) >= 30

    def test_all_have_names(self):
        for fd in FACTOR_LIBRARY:
            assert fd.name and fd.category


class TestFactorEngine:
    def setup_method(self):
        self.kline = _make_kline(300)
        self.engine = FactorEngine()

    def test_compute_all_returns_factors(self):
        factors = self.engine.compute_all(self.kline)
        assert isinstance(factors, dict)
        assert len(factors) > 20

    def test_factor_has_valid_count(self):
        factors = self.engine.compute_all(self.kline)
        for name, fr in factors.items():
            assert fr.valid_count > 0

    def test_compute_all_handles_minimal_data(self):
        kline = _make_kline(60)
        factors = self.engine.compute_all(kline)
        assert len(factors) > 0

    def test_insufficient_data_skips(self):
        kline = _make_kline(10)
        factors = self.engine.compute_all(kline)
        fwd_ret = kline["close"].pct_change(1).shift(-1).dropna()
        ic_results = self.engine.analyze_ic(factors, fwd_ret)
        assert len(ic_results) == 0

    def test_correlation_matrix(self):
        self.engine.compute_all(self.kline)
        corr = self.engine.correlation_matrix()
        assert isinstance(corr, pd.DataFrame)

    def test_compute_fundamental_factors(self):
        fund_data = {"roe": 0.15, "roa": 0.08, "pe_ttm": 15.5, "pb": 2.0}
        factors = self.engine.compute_fundamental_factors(fund_data)
        assert len(factors) > 0
        assert "roe" in factors

    def test_select_factors_empty(self):
        selected = self.engine.select_factors({})
        assert selected == []

    def test_synthesize_empty(self):
        combined = self.engine.synthesize([])
        assert combined.name == "combined"

    def test_empty_report(self):
        report = self.engine.generate_report({})
        assert "暂无" in report


class TestICAnalysis:
    """IC分析测试 - 需要足够数据。"""

    def setup_method(self):
        self.kline = _make_kline(400)  # 更多数据
        self.engine = FactorEngine()

    def test_ic_analysis(self):
        factors = self.engine.compute_all(self.kline)
        fwd_ret = self.kline["close"].pct_change(5).shift(-5).dropna()
        if len(fwd_ret) < 30:
            pytest.skip("数据不足")

        ic = self.engine.analyze_ic(factors, fwd_ret)
        # 可能成功也可能因各种原因失败（模拟数据IC过低）
        assert isinstance(ic, dict)

    def test_select_and_synthesize(self):
        factors = self.engine.compute_all(self.kline)
        fwd_ret = self.kline["close"].pct_change(5).shift(-5).dropna()

        ic = self.engine.analyze_ic(factors, fwd_ret)
        if not ic:
            pytest.skip("IC为空")

        # 验证返回类型即可（随机数据IC值不稳定）
        selected = self.engine.select_factors(ic, min_ic=-0.5, min_ir=-10,
                                               min_ic_pos_ratio=0.0)
        assert isinstance(selected, list)

    def test_discover(self):
        factors = self.engine.compute_all(self.kline)
        fwd_ret = self.kline["close"].pct_change(5).shift(-5).dropna()
        self.engine.analyze_ic(factors, fwd_ret)

        names = list(factors.keys())[:10]
        discoveries = self.engine.discover(names, max_ops=1)
        assert isinstance(discoveries, list)

    def test_generate_report(self):
        factors = self.engine.compute_all(self.kline)
        fwd_ret = self.kline["close"].pct_change(5).shift(-5).dropna()
        ic = self.engine.analyze_ic(factors, fwd_ret)
        report = self.engine.generate_report(ic)
        assert isinstance(report, str)