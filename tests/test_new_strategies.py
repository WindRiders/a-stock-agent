"""测试新增策略：网格交易 + 均值回归。"""

import pytest
from strategy.base import Signal, BaseStrategy
from strategy.grid import GridTradingStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.factory import StrategyFactory
from analysis.scoring import StockScore


def make_score(**kwargs) -> StockScore:
    defaults = {
        "symbol": "000001",
        "name": "测试股",
        "tech_score": 0,
        "fund_score": 0,
        "sentiment_score": 0,
        "capital_score": 0,
        "total_score": 0.5,
        "volume_ratio": 1.0,
    }
    defaults.update(kwargs)
    return StockScore(**defaults)


class TestGridTradingStrategy:
    """网格交易策略测试。"""

    def setup_method(self):
        self.strategy = GridTradingStrategy()

    def test_name(self):
        assert self.strategy.name == "grid"

    def test_buy_in_range(self):
        """震荡区间内买入信号。"""
        scores = [
            make_score(total_score=0.4, tech_score=2, volume_ratio=1.0)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.BUY

    def test_hold_on_breakout(self):
        """放量突破时不买入。"""
        scores = [
            make_score(total_score=0.3, tech_score=1, volume_ratio=3.0)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.HOLD

    def test_sell_on_breakdown(self):
        """跌破区间时卖出。"""
        scores = [
            make_score(total_score=0.1, tech_score=-5)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.SELL


class TestMeanReversionStrategy:
    """均值回归策略测试。"""

    def setup_method(self):
        self.strategy = MeanReversionStrategy()

    def test_name(self):
        assert self.strategy.name == "mean_reversion"

    def test_strong_buy_oversold(self):
        """严重超卖+基本面OK → 强烈买入。"""
        scores = [
            make_score(total_score=0.2, tech_score=-3, fund_score=0)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.STRONG_BUY

    def test_avoid_value_trap(self):
        """超卖+基本面差 → 不买入（价值陷阱）。"""
        scores = [
            make_score(total_score=0.1, tech_score=-3, fund_score=-3)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.HOLD

    def test_sell_overbought(self):
        """超买 → 卖出。"""
        scores = [
            make_score(total_score=0.8, tech_score=7)
        ]
        signals = self.strategy.generate_signals(scores)
        assert signals[0].signal == Signal.STRONG_SELL


class TestNewStrategiesInFactory:
    """验证新策略已注册到工厂。"""

    def test_grid_in_factory(self):
        s = StrategyFactory.get("grid")
        assert isinstance(s, GridTradingStrategy)

    def test_mean_reversion_in_factory(self):
        s = StrategyFactory.get("mean_reversion")
        assert isinstance(s, MeanReversionStrategy)

    def test_all_strategies_listed(self):
        strategies = StrategyFactory.list_strategies()
        assert "grid" in strategies
        assert "mean_reversion" in strategies
        assert len(strategies) == 5