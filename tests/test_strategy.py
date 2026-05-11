"""测试策略模块。"""

import pytest
from strategy.base import Signal, TradeSignal, BaseStrategy
from strategy.momentum import MomentumStrategy
from strategy.value import ValueStrategy
from strategy.trend import TrendFollowingStrategy
from strategy.factory import StrategyFactory
from analysis.scoring import StockScore


def make_score(**kwargs) -> StockScore:
    """创建模拟评分。"""
    defaults = {
        "symbol": "000001",
        "name": "测试股",
        "tech_score": 0,
        "fund_score": 0,
        "sentiment_score": 0,
        "capital_score": 0,
        "total_score": 0.5,
    }
    defaults.update(kwargs)
    return StockScore(**defaults)


class TestMomentumStrategy:
    """动量策略测试。"""

    def setup_method(self):
        self.strategy = MomentumStrategy()

    def test_name(self):
        assert self.strategy.name == "momentum"

    def test_strong_buy_signal(self):
        """强势股票应该生成强烈买入信号。"""
        scores = [make_score(tech_score=9, capital_score=2)]
        signals = self.strategy.generate_signals(scores)
        assert len(signals) == 1
        assert signals[0].signal == Signal.STRONG_BUY

    def test_strong_sell_signal(self):
        """技术面崩盘应该生成卖出信号。"""
        scores = [make_score(tech_score=-10, capital_score=-1)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.STRONG_SELL

    def test_hold_signal(self):
        """中性信号。"""
        scores = [make_score(tech_score=2, capital_score=0)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.HOLD

    def test_multiple_scores(self):
        """多只股票批量生成信号。"""
        scores = [
            make_score(symbol="000001", tech_score=9, capital_score=2),
            make_score(symbol="000002", tech_score=-8, capital_score=-1),
            make_score(symbol="000003", tech_score=2, capital_score=0),
        ]
        signals = self.strategy.generate_signals(scores)
        assert len(signals) == 3
        assert signals[0].signal == Signal.STRONG_BUY
        assert signals[1].signal == Signal.SELL
        assert signals[2].signal == Signal.HOLD


class TestValueStrategy:
    """价值策略测试。"""

    def setup_method(self):
        self.strategy = ValueStrategy()

    def test_name(self):
        assert self.strategy.name == "value"

    def test_strong_buy(self):
        """低估+高质量+技术共振。"""
        scores = [make_score(fund_score=5, tech_score=3)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.STRONG_BUY

    def test_sell_on_poor_fundamentals(self):
        """基本面极差。"""
        scores = [make_score(fund_score=-4)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.SELL


class TestTrendFollowingStrategy:
    """趋势跟踪策略测试。"""

    def setup_method(self):
        self.strategy = TrendFollowingStrategy()

    def test_name(self):
        assert self.strategy.name == "trend"

    def test_buy_on_high_score(self):
        """综合评分高。"""
        scores = [make_score(total_score=0.75)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.STRONG_BUY

    def test_sell_on_low_score_poor_fund(self):
        """综合评分低且基本面差。"""
        scores = [make_score(total_score=0.1, fund_score=-3)]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.SELL


class TestStrategyFactory:
    """策略工厂测试。"""

    def test_get_valid_strategies(self):
        for name in ["momentum", "value", "trend"]:
            s = StrategyFactory.get(name)
            assert isinstance(s, BaseStrategy)

    def test_get_invalid_strategy(self):
        with pytest.raises(ValueError):
            StrategyFactory.get("nonexistent")

    def test_list_strategies(self):
        strategies = StrategyFactory.list_strategies()
        assert "momentum" in strategies
        assert "value" in strategies
        assert "trend" in strategies
        assert len(strategies) >= 3


class TestSignal:
    """信号枚举测试。"""

    def test_signal_values(self):
        assert Signal.STRONG_BUY.value == "STRONG_BUY"
        assert Signal.HOLD.value == "HOLD"
        assert Signal.STRONG_SELL.value == "STRONG_SELL"