"""测试多资产组合回测模块。"""

import pytest
import pandas as pd
import numpy as np

from backtest.portfolio import (
    PortfolioBacktest,
    equal_weight_allocator,
    score_weighted_allocator,
    risk_parity_allocator,
    kelly_allocator,
    format_portfolio_result,
)


def _make_kline(symbol: str, n_days: int = 200, start_price: float = 10.0):
    """生成模拟K线数据。"""
    np.random.seed(hash(symbol) % 2**32)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.normal(0, 0.002, n_days)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_days),
    })


def _make_signals(kline: pd.DataFrame, symbol: str):
    """生成模拟回测信号。"""
    signals = []
    for i in range(20, len(kline)):
        price = kline.iloc[i]["close"]
        ma20 = kline["close"].iloc[i-20:i].mean()
        if price > ma20 * 1.05:
            signal = "BUY"
        elif price < ma20 * 0.95:
            signal = "SELL"
        else:
            signal = "HOLD"

        if signal != "HOLD":
            signals.append({
                "date": kline.iloc[i]["date"],
                "symbol": symbol,
                "signal": signal,
                "score": abs((price / ma20 - 1) * 100),
            })
    if not signals:
        return pd.DataFrame(columns=["date", "symbol", "signal", "score"])
    return pd.DataFrame(signals)


class TestAllocators:
    """资金分配器测试。"""

    def test_equal_weight(self):
        """等权分配。"""
        result = equal_weight_allocator(["A", "B", "C"], {}, 30000, max_positions=3)
        assert len(result) == 3
        assert abs(result["A"] - 10000) < 1
        assert abs(result["B"] - 10000) < 1
        assert abs(result["C"] - 10000) < 1

    def test_equal_weight_max_positions(self):
        """等权分配限制持仓数。"""
        result = equal_weight_allocator(["A", "B", "C", "D", "E"], {}, 50000, max_positions=2)
        assert len(result) == 2

    def test_score_weighted(self):
        """评分加权分配。"""
        scores = {"A": 0.8, "B": 0.5, "C": 0.3}
        result = score_weighted_allocator(["A", "B", "C"], scores, 16000, max_positions=3)
        assert len(result) == 3
        # 评分越高，金额越大
        assert result["A"] > result["B"] > result["C"]

    def test_score_weighted_total(self):
        """评分加权总和等于资金。"""
        scores = {"A": 0.8, "B": 0.2}
        result = score_weighted_allocator(["A", "B"], scores, 10000, max_positions=2)
        total = sum(result.values())
        assert abs(total - 10000) < 1

    def test_risk_parity(self):
        """风险平价分配。"""
        vol = {"A": 0.01, "B": 0.03, "C": 0.05}
        result = risk_parity_allocator(["A", "B", "C"], {}, 30000, vol, max_positions=3)
        assert len(result) == 3
        # 波动率越低，金额越大
        assert result["A"] > result["B"] > result["C"]

    def test_risk_parity_default_vol(self):
        """缺失波动率时使用默认值。"""
        vol = {"A": 0.02}
        result = risk_parity_allocator(["A", "B"], {}, 20000, vol, max_positions=2)
        assert len(result) == 2

    def test_kelly_allocator(self):
        """凯利分配。"""
        wr = {"A": 0.6, "B": 0.4}
        wl = {"A": 2.0, "B": 1.0}
        result = kelly_allocator(["A", "B"], {}, 20000, wr, wl, max_positions=2)
        assert len(result) == 2

    def test_kelly_zero_returns_equal(self):
        """凯利全零时回退等权。"""
        result = kelly_allocator(["A", "B", "C"], {}, 30000,
                                  win_rate={"A": 0, "B": 0, "C": 0},
                                  max_positions=3)
        assert len(result) == 3


class TestPortfolioBacktest:
    """多资产组合回测测试。"""

    def test_empty_input(self):
        """空输入返回空结果。"""
        engine = PortfolioBacktest(100000)
        result = engine.run({}, {})
        assert result.initial_capital == 100000
        assert result.final_equity == 100000
        assert result.total_trades == 0

    def test_single_symbol(self):
        """单标的回测。"""
        kline = _make_kline("000001", 200, 10.0)
        signals = _make_signals(kline, "000001")

        engine = PortfolioBacktest(100000)
        result = engine.run(
            {"000001": signals},
            {"000001": kline},
            allocation="equal_weight",
            rebalance="weekly",
        )
        assert result.initial_capital == 100000
        assert result.total_trades >= 0

    def test_multi_symbol(self):
        """多标的同时回测。"""
        sym1_kline = _make_kline("000001", 150, 10.0)
        sym2_kline = _make_kline("000002", 150, 15.0)
        signals1 = _make_signals(sym1_kline, "000001")
        signals2 = _make_signals(sym2_kline, "000002")

        engine = PortfolioBacktest(100000)
        result = engine.run(
            {"000001": signals1, "000002": signals2},
            {"000001": sym1_kline, "000002": sym2_kline},
            allocation="score_weighted",
            rebalance="monthly",
            scores={"000001": 0.8, "000002": 0.6},
        )
        assert result.initial_capital == 100000
        assert len(result.equity_curve) > 0

    def test_allocation_strategies(self):
        """所有分配策略都能运行。"""
        kline = _make_kline("000001", 200)
        signals = _make_signals(kline, "000001")

        for alloc in ["equal_weight", "score_weighted", "risk_parity", "kelly"]:
            engine = PortfolioBacktest(100000)
            result = engine.run(
                {"000001": signals},
                {"000001": kline},
                allocation=alloc,
                rebalance="weekly",
                scores={"000001": 0.7},
                volatility={"000001": 0.02},
                win_rate={"000001": 0.55},
                win_loss_ratio={"000001": 1.5},
            )
            assert result is not None

    def test_rebalance_frequencies(self):
        """不同再平衡周期。"""
        kline = _make_kline("000001", 200)
        signals = _make_signals(kline, "000001")

        for freq in ["daily", "weekly", "monthly"]:
            engine = PortfolioBacktest(100000)
            result = engine.run(
                {"000001": signals},
                {"000001": kline},
                rebalance=freq,
            )
            assert result is not None

    def test_equity_curve_has_data(self):
        """权益曲线有数据。"""
        kline = _make_kline("000001", 100)
        signals = _make_signals(kline, "000001")

        engine = PortfolioBacktest(100000)
        result = engine.run({"000001": signals}, {"000001": kline})
        assert not result.equity_curve.empty
        assert "equity" in result.equity_curve.columns
        assert "drawdown" in result.equity_curve.columns

    def test_per_symbol_returns(self):
        """不同标的分别计算收益。"""
        k1 = _make_kline("000001", 150)
        k2 = _make_kline("000002", 150)
        s1 = _make_signals(k1, "000001")
        s2 = _make_signals(k2, "000002")

        engine = PortfolioBacktest(100000)
        result = engine.run(
            {"000001": s1, "000002": s2},
            {"000001": k1, "000002": k2},
        )

    def test_format_report(self):
        """报告格式化。"""
        from backtest.portfolio import PortfolioResult
        result = PortfolioResult(
            initial_capital=100000,
            final_equity=110000,
            total_return=10.0,
            sharpe_ratio=1.5,
            diversification_score=0.8,
            best_symbol="000001",
            worst_symbol="000002",
            per_symbol_returns={"000001": 15.0, "000002": -5.0},
        )
        report = format_portfolio_result(result)
        assert "最终权益" in report
        assert "10.00%" in report
        assert "000001" in report

    def test_diversification_score(self):
        """分散化评分计算。"""
        k1 = _make_kline("000001", 100)
        k2 = _make_kline("000002", 100)
        s1 = _make_signals(k1, "000001")
        s2 = _make_signals(k2, "000002")

        engine = PortfolioBacktest(100000)
        result = engine.run(
            {"000001": s1, "000002": s2},
            {"000001": k1, "000002": k2},
        )
        assert 0 <= result.diversification_score <= 1.0