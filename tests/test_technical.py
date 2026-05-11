"""测试技术分析模块。"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from analysis.technical import TechnicalAnalyzer, TechResult


def make_kline_data(
    n_days: int = 200,
    trend: str = "up",
    start_price: float = 10.0,
) -> pd.DataFrame:
    """生成模拟K线数据。"""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

    if trend == "up":
        drift = 0.005
    elif trend == "down":
        drift = -0.005
    else:
        drift = 0.0

    returns = np.random.normal(drift, 0.02, n_days)
    prices = start_price * np.exp(np.cumsum(returns))

    # 生成 OHLC
    close = prices
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_vals = np.roll(close, 1)
    open_vals[0] = start_price
    volume = np.random.randint(1000000, 10000000, n_days)
    pct_change = np.array([0.0] + list((close[i] / close[i - 1] - 1) * 100 for i in range(1, n_days)))

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_vals,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "pct_change": pct_change,
        }
    )


class TestTechnicalAnalyzer:
    """技术分析测试。"""

    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_analyze_uptrend(self):
        """上升趋势中的技术分析。"""
        df = make_kline_data(n_days=200, trend="up")
        result = self.analyzer.analyze(df, "TEST01")

        assert isinstance(result, TechResult)
        assert result.latest_close > 0
        assert result.ma5 is not None
        assert result.ma20 is not None
        # 上升趋势中 MA 应该多头排列
        assert result.ma5 > result.ma20

    def test_analyze_downtrend(self):
        """下降趋势中的技术分析。"""
        df = make_kline_data(n_days=200, trend="down")
        result = self.analyzer.analyze(df, "TEST01")

        assert result.latest_close > 0
        # 下降趋势中均线趋势得分应该为负
        assert result.trend_score <= 0

    def test_rsi_calculation(self):
        """RSI 计算正确性。"""
        df = make_kline_data(n_days=200, trend="sideways")
        result = self.analyzer.analyze(df, "TEST01")

        assert 0 <= result.rsi_14 <= 100
        assert 0 <= result.rsi_6 <= 100

    def test_macd_calculation(self):
        """MACD 计算正确性。"""
        df = make_kline_data(n_days=200, trend="up")
        result = self.analyzer.analyze(df, "TEST01")

        assert result.macd is not None
        assert result.macd_signal is not None
        assert result.macd_hist is not None

    def test_bollinger_calculation(self):
        """布林带计算正确性。"""
        df = make_kline_data(n_days=200, trend="sideways")
        result = self.analyzer.analyze(df, "TEST01")

        assert result.bb_upper > result.bb_middle > result.bb_lower
        assert 0 <= result.bb_position <= 1

    def test_kdj_calculation(self):
        """KDJ 计算正确性。"""
        df = make_kline_data(n_days=200, trend="up")
        result = self.analyzer.analyze(df, "TEST01")

        assert result.k is not None
        assert result.d is not None
        assert result.j is not None

    def test_signal_generation(self):
        """信号生成逻辑。"""
        result = TechResult(symbol="TEST")
        result.total_score = 8
        signal = self.analyzer._generate_signal(result)
        assert signal == "STRONG_BUY"

        result.total_score = 4
        signal = self.analyzer._generate_signal(result)
        assert signal == "BUY"

        result.total_score = 0
        signal = self.analyzer._generate_signal(result)
        assert signal == "HOLD"

        result.total_score = -4
        signal = self.analyzer._generate_signal(result)
        assert signal == "SELL"

        result.total_score = -8
        signal = self.analyzer._generate_signal(result)
        assert signal == "STRONG_SELL"


class TestTechResult:
    """TechResult 数据类测试。"""

    def test_defaults(self):
        result = TechResult(symbol="TEST")
        assert result.symbol == "TEST"
        assert result.total_score == 0
        assert result.signal == "HOLD"
        assert result.trend_score == 0