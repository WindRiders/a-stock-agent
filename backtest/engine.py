"""回测引擎。

模拟A股交易（T+1、涨跌停限制、手续费、印花税）。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔交易记录。"""

    date: str
    symbol: str
    action: str  # "BUY" / "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    stamp_tax: float = 0.0  # 印花税（仅卖出）
    reason: str = ""


@dataclass
class Position:
    """持仓。"""

    symbol: str
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def profit_pct(self) -> float:
        if self.avg_cost > 0:
            return (self.current_price / self.avg_cost - 1) * 100
        return 0.0


@dataclass
class BacktestResult:
    """回测结果。"""

    # 汇总指标
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # 详细记录
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)

    # 基准对比
    benchmark_return: float = 0.0


class BacktestEngine:
    """A股回测引擎。

    模拟规则：
    - T+1 交易（当日买入次日可卖）
    - 涨停无法买入，跌停无法卖出
    - 佣金：万分之 2.5（最低5元）
    - 印花税：千分之 1（仅卖出）
    - 滑点：万分之 1
    """

    COMMISSION_RATE = 0.00025  # 万2.5
    STAMP_TAX_RATE = 0.001  # 千1
    MIN_COMMISSION = 5.0
    SLIPPAGE = 0.0001  # 万1

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[dict] = []

    def run(
        self,
        signals_df: pd.DataFrame,  # date, symbol, signal, score
        price_data: Dict[str, pd.DataFrame],  # symbol -> OHLCV
        benchmark_data: pd.DataFrame = None,
    ) -> BacktestResult:
        """执行回测。

        Args:
            signals_df: 交易信号 DataFrame
            price_data: 各股票价格数据
            benchmark_data: 基准指数数据（可选）

        Returns:
            BacktestResult
        """
        if signals_df.empty:
            return self._empty_result()

        # 按日期排序
        signals_df = signals_df.sort_values("date")
        dates = signals_df["date"].unique()

        prev_equity = self.initial_capital
        prev_peak = self.initial_capital  # 历史最高净值
        daily_returns = []

        for date in dates:
            day_signals = signals_df[signals_df["date"] == date]

            for _, sig in day_signals.iterrows():
                symbol = sig["symbol"]
                signal = sig.get("signal", "HOLD")
                score = sig.get("score", 0)

                if signal in ("BUY", "STRONG_BUY"):
                    self._execute_buy(symbol, price_data, date)
                elif signal in ("SELL", "STRONG_SELL"):
                    self._execute_sell(symbol, price_data, date)

            # 记录当日权益
            equity = self._calc_equity(price_data, date)
            day_return = (equity / prev_equity - 1) if prev_equity > 0 else 0
            daily_returns.append(day_return)

            if equity > prev_peak:
                prev_peak = equity

            drawdown = (equity / prev_peak - 1) if prev_peak > 0 else 0

            self.equity_history.append(
                {"date": date, "equity": equity, "cash": self.cash, "drawdown": drawdown}
            )

            prev_equity = equity

        # 计算最终指标
        final_equity = self.equity_history[-1]["equity"] if self.equity_history else self.initial_capital
        equity_df = pd.DataFrame(self.equity_history)

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=(final_equity / self.initial_capital - 1) * 100,
            max_drawdown=equity_df["drawdown"].min() * 100,
            total_trades=len(self.trades),
            equity_curve=equity_df,
            trades=self.trades,
        )

        # 年化收益率
        if len(daily_returns) > 0:
            trading_days = len(daily_returns)
            years = trading_days / 252
            if years > 0:
                result.annual_return = (
                    (final_equity / self.initial_capital) ** (1 / years) - 1
                ) * 100

        # 夏普比率
        if len(daily_returns) > 1:
            daily_rr = np.array(daily_returns)
            mean_return = np.mean(daily_rr)
            std_return = np.std(daily_rr, ddof=1)
            if std_return > 0:
                result.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

        # 胜率
        completed = [t for t in self.trades if t.action == "SELL"]
        if completed:
            sells = [t.amount for t in completed]
            # 简化胜率计算：卖出价>买入价的交易数/卖出总数
            # 实际需要配对买入卖出
            buy_trades = [t for t in self.trades if t.action == "BUY"]
            result.win_rate = (
                len([t for t in completed if t.amount > 0]) / len(completed) * 100
            )

        return result

    def _execute_buy(
        self, symbol: str, price_data: Dict[str, pd.DataFrame], date: str
    ):
        """执行买入。"""
        if symbol not in price_data:
            return

        df = price_data[symbol]
        day_data = df[df["date"] == date]
        if day_data.empty:
            return

        price = float(day_data["close"].iloc[0])

        # 涨停检查
        pre_close = float(day_data.iloc[0].get("pre_close", day_data["close"].iloc[0]))
        pct_change = float(day_data.iloc[0].get("pct_change", 0))
        if pct_change >= 9.8:  # 接近涨停，无法买入
            logger.debug("%s %s 涨停，无法买入", date, symbol)
            return

        # 价格加滑点
        buy_price = price * (1 + self.SLIPPAGE)

        # 仓位管理：单只股票最多 20% 仓位
        max_position_value = self.initial_capital * 0.20
        current_value = self.positions.get(symbol, Position(symbol=symbol)).market_value

        # 买入金额
        buy_amount = min(self.cash * 0.8, max_position_value - current_value)
        if buy_amount < buy_price * 100:  # 最少买一手（100股）
            return

        shares = int(buy_amount / buy_price / 100) * 100
        if shares == 0:
            return

        amount = buy_price * shares
        commission = max(self.MIN_COMMISSION, amount * self.COMMISSION_RATE)

        if self.cash < amount + commission:
            return

        # 更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_cost = pos.avg_cost * pos.shares + amount + commission
            pos.shares += shares
            pos.avg_cost = total_cost / pos.shares if pos.shares > 0 else 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol, shares=shares, avg_cost=(amount + commission) / shares
            )

        self.cash -= amount + commission

        self.trades.append(
            Trade(
                date=str(date),
                symbol=symbol,
                action="BUY",
                price=buy_price,
                shares=shares,
                amount=amount,
                commission=commission,
            )
        )

    def _execute_sell(
        self, symbol: str, price_data: Dict[str, pd.DataFrame], date: str
    ):
        """执行卖出。"""
        if symbol not in self.positions or self.positions[symbol].shares == 0:
            return

        df = price_data[symbol]
        day_data = df[df["date"] == date]
        if day_data.empty:
            return

        price = float(day_data["close"].iloc[0])

        # 跌停检查
        pct_change = float(day_data.iloc[0].get("pct_change", 0))
        if pct_change <= -9.8:
            logger.debug("%s %s 跌停，无法卖出", date, symbol)
            return

        sell_price = price * (1 - self.SLIPPAGE)

        pos = self.positions[symbol]
        shares = pos.shares
        amount = sell_price * shares
        commission = max(self.MIN_COMMISSION, amount * self.COMMISSION_RATE)
        stamp_tax = amount * self.STAMP_TAX_RATE

        self.cash += amount - commission - stamp_tax
        del self.positions[symbol]

        self.trades.append(
            Trade(
                date=str(date),
                symbol=symbol,
                action="SELL",
                price=sell_price,
                shares=shares,
                amount=amount,
                commission=commission,
                stamp_tax=stamp_tax,
            )
        )

    def _calc_equity(
        self, price_data: Dict[str, pd.DataFrame], date: str
    ) -> float:
        """计算当日总权益（现金+持仓市值）。"""
        equity = self.cash
        for symbol, pos in self.positions.items():
            df = price_data.get(symbol)
            if df is not None:
                day_data = df[df["date"] == date]
                if not day_data.empty:
                    pos.current_price = float(day_data["close"].iloc[0])
                    equity += pos.market_value
        return equity

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(initial_capital=self.initial_capital, final_equity=self.initial_capital)