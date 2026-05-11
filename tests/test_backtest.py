"""测试回测引擎。"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtest.engine import BacktestEngine, BacktestResult, Trade, Position


def make_price_data(
    symbol: str = "000001",
    n_days: int = 100,
    trend: str = "up",
    start_price: float = 10.0,
) -> pd.DataFrame:
    """生成模拟价格数据。"""
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        periods=n_days,
        freq="B",
    )

    if trend == "up":
        drift = 0.005
    elif trend == "down":
        drift = -0.005
    else:
        drift = 0.0

    returns = np.random.normal(drift, 0.02, n_days)
    prices = start_price * np.exp(np.cumsum(returns))

    high = prices * 1.02
    low = prices * 0.98
    open_vals = np.roll(prices, 1)
    open_vals[0] = start_price
    volume = np.random.randint(1000000, 5000000, n_days)
    pct_change = np.concatenate([[0], np.diff(prices) / prices[:-1] * 100])

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_vals,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
            "pct_change": pct_change,
        }
    )


class TestBacktestEngine:
    """回测引擎测试。"""

    def setup_method(self):
        self.engine = BacktestEngine(initial_capital=100000)

    def test_initial_state(self):
        """初始状态。"""
        assert self.engine.initial_capital == 100000
        assert self.engine.cash == 100000
        assert len(self.engine.positions) == 0
        assert len(self.engine.trades) == 0

    def test_buy_execution(self):
        """买入执行。"""
        price_data = {"000001": make_price_data("000001", n_days=100)}
        signals = pd.DataFrame(
            [{"date": price_data["000001"]["date"].iloc[50], "symbol": "000001", "signal": "BUY", "score": 0.8}]
        )
        result = self.engine.run(signals, price_data)
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 1

    def test_sell_execution(self):
        """卖出执行——先买后卖。"""
        price_data = {"000001": make_price_data("000001", n_days=100)}
        dates = price_data["000001"]["date"].values

        signals = pd.DataFrame(
            [
                {"date": dates[30], "symbol": "000001", "signal": "BUY", "score": 0.8},
                {"date": dates[60], "symbol": "000001", "signal": "SELL", "score": 0.2},
            ]
        )
        result = self.engine.run(signals, price_data)
        assert result.total_trades == 2

    def test_limit_up_block(self):
        """涨停不可买入。"""
        price_data = make_price_data("000001", n_days=100)
        # 修改某一天为涨停
        price_data.loc[50, "pct_change"] = 9.9
        price_data = {"000001": price_data}

        signals = pd.DataFrame(
            [{"date": price_data["000001"]["date"].iloc[50], "symbol": "000001", "signal": "BUY", "score": 0.8}]
        )
        result = self.engine.run(signals, price_data)
        # 涨停日不应有买入
        buys = [t for t in result.trades if t.action == "BUY"]
        assert len(buys) == 0

    def test_position_size_limit(self):
        """仓位限制：单只不超过20%。"""
        price_data = {"000001": make_price_data("000001", n_days=100)}
        dates = price_data["000001"]["date"].values

        signals = pd.DataFrame(
            [{"date": dates[30], "symbol": "000001", "signal": "STRONG_BUY", "score": 0.9}]
        )

        # 手动执行买入
        self.engine._execute_buy("000001", price_data, dates[30])

        # 检查持仓不超过 20%
        if "000001" in self.engine.positions:
            pos_value = self.engine.positions["000001"].market_value
            assert pos_value <= self.engine.initial_capital * 0.20 + 1000  # 允许微小浮动

    def test_result_metrics(self):
        """回测结果指标。"""
        price_data = {"000001": make_price_data("000001", n_days=100, trend="up")}
        dates = price_data["000001"]["date"].values

        signals = pd.DataFrame(
            [
                {"date": dates[20], "symbol": "000001", "signal": "BUY", "score": 0.9},
                {"date": dates[80], "symbol": "000001", "signal": "SELL", "score": 0.3},
            ]
        )
        result = self.engine.run(signals, price_data)

        assert result.initial_capital == 100000
        assert result.total_trades > 0
        assert result.max_drawdown <= 0  # 最大回撤是负数


class TestPosition:
    """持仓类测试。"""

    def test_position_creation(self):
        pos = Position(symbol="000001", shares=1000, avg_cost=10.0, current_price=11.0)
        assert pos.market_value == 11000.0
        assert pos.profit == 1000.0
        assert pos.profit_pct == pytest.approx(10.0)

    def test_empty_position(self):
        pos = Position(symbol="000001")
        assert pos.market_value == 0.0
        assert pos.profit == 0.0


class TestTrade:
    """交易记录测试。"""

    def test_buy_trade(self):
        trade = Trade(
            date="2024-01-15",
            symbol="000001",
            action="BUY",
            price=10.0,
            shares=1000,
            amount=10000.0,
            commission=5.0,
        )
        assert trade.action == "BUY"
        assert trade.stamp_tax == 0.0  # 买入不交印花税

    def test_sell_trade(self):
        trade = Trade(
            date="2024-01-15",
            symbol="000001",
            action="SELL",
            price=12.0,
            shares=1000,
            amount=12000.0,
            commission=5.0,
            stamp_tax=12.0,
        )
        assert trade.action == "SELL"
        assert trade.stamp_tax == 12.0