"""多资产组合回测引擎。

在单资产回测基础上，支持：
- 多标的动态调仓
- 资金池管理（等权 / 风险平价 / 凯利公式 / 评分加权）
- 再平衡周期（日/周/月）
- 组合层面指标（分散化评分、换手率、行业集中度）

用法:
    from backtest.portfolio import PortfolioBacktest

    engine = PortfolioBacktest(initial_capital=100000)
    result = engine.run(
        signals_by_symbol={"000001": signals_df1, "000002": signals_df2},
        price_data={"000001": kline_df1, "000002": kline_df2},
        allocation="equal_weight",
        rebalance="weekly",
    )
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, Trade, Position

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class PortfolioResult:
    """组合回测结果。"""

    # 汇总
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # 组合特有
    diversification_score: float = 0.0  # 分散化评分
    avg_turnover: float = 0.0  # 平均换手率
    max_sector_concentration: float = 0.0  # 最大行业集中度
    avg_positions: float = 0.0  # 平均持仓数
    best_symbol: str = ""  # 最佳标的
    worst_symbol: str = ""  # 最差标的
    per_symbol_returns: Dict[str, float] = field(default_factory=dict)

    # 详细
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: List[float] = field(default_factory=list)

    # 基准对比
    benchmark_return: float = 0.0


# ── 资金分配策略 ───────────────────────────────────────────

def equal_weight_allocator(
    candidates: List[str],
    scores: Dict[str, float],
    total_cash: float,
    max_positions: int = 5,
) -> Dict[str, float]:
    """等权分配。"""
    n = min(len(candidates), max_positions)
    if n == 0:
        return {}
    amount_per = total_cash / n
    return {c: amount_per for c in candidates[:n]}


def score_weighted_allocator(
    candidates: List[str],
    scores: Dict[str, float],
    total_cash: float,
    max_positions: int = 5,
) -> Dict[str, float]:
    """评分加权分配。"""
    # 只取有评分的标的前 N 个
    ranked = sorted(candidates, key=lambda c: scores.get(c, 0), reverse=True)
    ranked = ranked[:max_positions]

    total_score = sum(scores.get(c, 0) for c in ranked)
    if total_score <= 0:
        return equal_weight_allocator(ranked, scores, total_cash, max_positions)

    return {
        c: total_cash * (scores.get(c, 0) / total_score)
        for c in ranked
    }


def risk_parity_allocator(
    candidates: List[str],
    scores: Dict[str, float],
    total_cash: float,
    volatility: Dict[str, float],
    max_positions: int = 5,
) -> Dict[str, float]:
    """风险平价分配（波动率倒数加权）。

    每只标的分配金额 ∝ 1/波动率
    实现等风险贡献。
    """
    ranked = sorted(candidates, key=lambda c: scores.get(c, 0), reverse=True)
    ranked = ranked[:max_positions]

    inv_vol = {}
    for c in ranked:
        vol = volatility.get(c, 0.02)  # 默认 2% 日波动
        inv_vol[c] = 1.0 / max(vol, 0.001)

    total_inv = sum(inv_vol.values())
    if total_inv <= 0:
        return equal_weight_allocator(ranked, scores, total_cash, max_positions)

    return {c: total_cash * (inv_vol[c] / total_inv) for c in ranked}


def kelly_allocator(
    candidates: List[str],
    scores: Dict[str, float],
    total_cash: float,
    win_rate: Dict[str, float],
    avg_win_loss_ratio: Dict[str, float] = None,
    max_positions: int = 5,
    kelly_fraction: float = 0.5,  # 半凯利
) -> Dict[str, float]:
    """凯利公式分配。

    凯利比例 = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    使用半凯利降低风险。

    Args:
        win_rate: 各标的胜率
        avg_win_loss_ratio: 各标的平均盈亏比
        kelly_fraction: 凯利系数 (0.5 = 半凯利)
    """
    ranked = sorted(candidates, key=lambda c: scores.get(c, 0), reverse=True)
    ranked = ranked[:max_positions]

    allocation = {}
    total_kelly = 0

    for c in ranked:
        wr = win_rate.get(c, 0.5)
        wl = (avg_win_loss_ratio or {}).get(c, 1.5)
        # 凯利公式
        kelly = wr - (1 - wr) / max(wl, 0.1)
        kelly = max(0, min(kelly, 0.25))  # 单只上限 25%
        allocation[c] = kelly * kelly_fraction
        total_kelly += allocation[c]

    if total_kelly <= 0:
        return equal_weight_allocator(ranked, scores, total_cash, max_positions)

    if total_kelly > 1.0:
        # 归一化
        scale = 1.0 / total_kelly
        allocation = {c: v * scale for c, v in allocation.items()}

    return {c: total_cash * allocation[c] for c, v in allocation.items()}


# ── 分配器注册表 ──────────────────────────────────────────

ALLOCATORS = {
    "equal_weight": equal_weight_allocator,
    "score_weighted": score_weighted_allocator,
    "risk_parity": risk_parity_allocator,
    "kelly": kelly_allocator,
}


# ── 主引擎 ─────────────────────────────────────────────────

class PortfolioBacktest:
    """多资产组合回测引擎。

    模拟逻辑：
    - 按交易日遍历
    - 每个再平衡日 → 重新计算资金分配 → 调仓
    - 佣金、印花税、最小交易单位、涨跌停限制
    """

    COMMISSION_RATE = 0.00025
    STAMP_TAX_RATE = 0.001
    MIN_COMMISSION = 5.0
    SLIPPAGE = 0.0001

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[dict] = []
        self.positions_history: List[dict] = []

    def run(
        self,
        signals_by_symbol: Dict[str, pd.DataFrame],
        price_data: Dict[str, pd.DataFrame],
        allocation: str = "equal_weight",
        max_positions: int = 5,
        rebalance: str = "weekly",
        benchmark_data: pd.DataFrame = None,
        scores: Dict[str, float] = None,
        volatility: Dict[str, float] = None,
        win_rate: Dict[str, float] = None,
        win_loss_ratio: Dict[str, float] = None,
    ) -> PortfolioResult:
        """运行多资产组合回测。

        Args:
            signals_by_symbol: {symbol: signals_df (date, signal, score)}
            price_data: {symbol: OHLCV kline_df}
            allocation: 分配策略名称
            max_positions: 最大持仓数
            rebalance: 再平衡周期 (daily/weekly/monthly)
            benchmark_data: 基准指数
            scores: 各标的最新评分（评分加权用）
            volatility: 各标的波动率（风险平价用）
            win_rate: 胜率（凯利用）
            win_loss_ratio: 盈亏比（凯利用）

        Returns:
            PortfolioResult
        """
        if not signals_by_symbol or not price_data:
            return PortfolioResult(initial_capital=self.initial_capital,
                                   final_equity=self.initial_capital)

        # 获取全部交易日
        all_dates = self._get_all_dates(price_data)
        if not all_dates:
            return PortfolioResult(initial_capital=self.initial_capital,
                                   final_equity=self.initial_capital)

        scores = scores or {}
        volatility = volatility or {}
        win_rate = win_rate or {}
        win_loss_ratio = win_loss_ratio or {}

        allocator = ALLOCATORS.get(allocation, equal_weight_allocator)
        symbols = list(signals_by_symbol.keys())

        prev_equity = self.initial_capital
        prev_peak = self.initial_capital
        daily_returns = []
        last_rebalance_date = None

        for date in all_dates:
            date_str = str(date)[:10]

            # 是否需要再平衡
            should_rebalance = self._check_rebalance(date_str, last_rebalance_date, rebalance)

            if should_rebalance:
                last_rebalance_date = date_str
                # 计算当前权益
                equity = self._calc_equity(price_data, date_str)
                usable_cash = self.cash

                # 获取各标的信号和评分
                current_scores = {}
                for sym in symbols:
                    sig_df = signals_by_symbol.get(sym, pd.DataFrame())
                    if not sig_df.empty:
                        day_sigs = sig_df[sig_df["date"].astype(str).str[:10] == date_str]
                        if not day_sigs.empty:
                            current_scores[sym] = float(day_sigs["score"].iloc[0])

                # 候选标的：当前有买入信号的
                candidates = []
                for sym in symbols:
                    sig_df = signals_by_symbol.get(sym, pd.DataFrame())
                    if not sig_df.empty:
                        day_sigs = sig_df[sig_df["date"].astype(str).str[:10] == date_str]
                        if not day_sigs.empty:
                            sig = str(day_sigs["signal"].iloc[0])
                            if sig in ("BUY", "STRONG_BUY"):
                                candidates.append(sym)

                # 计算目标分配
                if allocation == "risk_parity":
                    target_allocation = risk_parity_allocator(
                        candidates, scores or current_scores, equity,
                        volatility=volatility, max_positions=max_positions,
                    )
                elif allocation == "kelly":
                    target_allocation = kelly_allocator(
                        candidates, scores or current_scores, equity,
                        win_rate=win_rate, avg_win_loss_ratio=win_loss_ratio,
                        max_positions=max_positions,
                    )
                else:
                    target_allocation = allocator(
                        candidates, scores or current_scores, equity,
                        max_positions=max_positions,
                    )

                if False:  # old branch, now unused
                    # 传入额外参数
                    if allocation == "risk_parity":
                        target_allocation = risk_parity_allocator(
                            candidates, scores or current_scores, equity,
                            volatility=volatility, max_positions=max_positions,
                        )
                    elif allocation == "kelly":
                        target_allocation = kelly_allocator(
                            candidates, scores or current_scores, equity,
                            win_rate=win_rate, avg_win_loss_ratio=win_loss_ratio,
                            max_positions=max_positions,
                        )

                # 执行调仓
                self._rebalance(target_allocation, price_data, date_str)

            # 记录权益
            equity = self._calc_equity(price_data, date_str)
            day_return = (equity / prev_equity - 1) if prev_equity > 0 else 0
            daily_returns.append(day_return)

            if equity > prev_peak:
                prev_peak = equity

            drawdown = (equity / prev_peak - 1) if prev_peak > 0 else 0

            self.equity_history.append({
                "date": date_str,
                "equity": equity,
                "cash": self.cash,
                "drawdown": drawdown,
                "positions": len(self.positions),
            })

            # 记录持仓明细
            positions_detail = {"date": date_str}
            for sym, pos in self.positions.items():
                if sym in price_data:
                    day_df = price_data[sym][
                        price_data[sym]["date"].astype(str).str[:10] == date_str
                    ]
                    if not day_df.empty:
                        pos.current_price = float(day_df["close"].iloc[0])
                positions_detail[sym] = {
                    "shares": pos.shares,
                    "price": pos.current_price,
                    "value": pos.market_value,
                }
            self.positions_history.append(positions_detail)

            prev_equity = equity

        # 计算最终指标
        final_equity = self.equity_history[-1]["equity"] if self.equity_history else self.initial_capital
        equity_df = pd.DataFrame(self.equity_history)

        result = PortfolioResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_return=(final_equity / self.initial_capital - 1) * 100,
            max_drawdown=equity_df["drawdown"].min() * 100 if not equity_df.empty else 0,
            total_trades=len(self.trades),
            equity_curve=equity_df,
            trades=self.trades,
            daily_returns=daily_returns,
            avg_positions=equity_df["positions"].mean() if not equity_df.empty else 0,
        )

        # 年化收益
        if len(daily_returns) > 0:
            years = len(daily_returns) / 252
            if years > 0:
                result.annual_return = (
                    (final_equity / self.initial_capital) ** (1 / years) - 1
                ) * 100

        # 夏普
        if len(daily_returns) > 1:
            dr = np.array(daily_returns)
            mean_dr = np.mean(dr)
            std_dr = np.std(dr, ddof=1)
            if std_dr > 0:
                result.sharpe_ratio = (mean_dr / std_dr) * np.sqrt(252)

        # 胜率
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        if sell_trades:
            # 配对买入卖出计算盈亏
            wins = 0
            total_completed = 0
            buy_map = {}  # symbol -> [(price, shares)]
            for t in self.trades:
                if t.action == "BUY":
                    buy_map.setdefault(t.symbol, []).append((t.price, t.shares))
                elif t.action == "SELL":
                    if t.symbol in buy_map and buy_map[t.symbol]:
                        buy_price, buy_shares = buy_map[t.symbol].pop(0)
                        if t.price > buy_price:
                            wins += 1
                        total_completed += 1
            if total_completed > 0:
                result.win_rate = wins / total_completed * 100

        # 分散化评分
        result.diversification_score = self._calc_diversification(price_data)

        # 换手率
        avg_turnover = 0.0
        if len(self.equity_history) > 1:
            turnover_events = 0
            for i in range(1, len(self.equity_history)):
                prev_positions = self.equity_history[i - 1].get("positions", 0)
                curr_positions = self.equity_history[i].get("positions", 0)
                if prev_positions != curr_positions:
                    turnover_events += 1
            avg_turnover = turnover_events / (len(self.equity_history) - 1)
        result.avg_turnover = avg_turnover

        # 各标的收益
        per_sym = {}
        for sym in price_data:
            if sym not in self.trades:
                continue
            sym_trades = [t for t in self.trades if t.symbol == sym]
            buys = [t for t in sym_trades if t.action == "BUY"]
            sells = [t for t in sym_trades if t.action == "SELL"]
            if buys and sells:
                total_buy = sum(t.amount for t in buys)
                total_sell = sum(t.amount for t in sells)
                if total_buy > 0:
                    per_sym[sym] = (total_sell / total_buy - 1) * 100
        result.per_symbol_returns = per_sym

        if per_sym:
            result.best_symbol = max(per_sym, key=per_sym.get)
            result.worst_symbol = min(per_sym, key=per_sym.get)

        return result

    # ── 再平衡 ─────────────────────────────────────────────

    def _rebalance(
        self,
        target_allocation: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        date: str,
    ):
        """根据目标分配调仓。

        1. 卖出不在目标中的持仓
        2. 调整目标中持仓的股数
        3. 买入新标
        """
        # 1. 清仓不在目标中的标的
        to_sell = [s for s in self.positions if s not in target_allocation]
        for sym in to_sell:
            self._sell_all(sym, price_data, date)

        # 2. 调整现有持仓
        total_equity = self._calc_equity(price_data, date)
        total_cash = self.cash

        for sym, target_amount in target_allocation.items():
            if sym not in price_data:
                continue

            day_data = price_data[sym][
                price_data[sym]["date"].astype(str).str[:10] == date
            ]
            if day_data.empty:
                continue

            price = float(day_data["close"].iloc[0])

            # 涨跌停检查
            pct_change = float(day_data.iloc[0].get("pct_change", 0))
            if pct_change >= 9.8 or pct_change <= -9.8:
                continue

            current_value = self.positions[sym].market_value if sym in self.positions else 0

            if abs(current_value - target_amount) < price * 100:
                continue  # 差异太小，不调

            if current_value > target_amount:
                # 减仓
                sell_amount = current_value - target_amount
                sell_shares = int(sell_amount / price / 100) * 100
                if sell_shares >= 100 and sym in self.positions:
                    actual_shares = min(sell_shares, self.positions[sym].shares)
                    self._sell(sym, price, actual_shares, date)
            else:
                # 加仓
                buy_amount = target_amount - current_value
                buy_shares = int(buy_amount / price / 100) * 100
                if buy_shares >= 100:
                    self._buy(sym, price, buy_shares, date)

        # 3. 买入新标的
        for sym, target_amount in target_allocation.items():
            if sym in self.positions or sym not in price_data:
                continue

            day_data = price_data[sym][
                price_data[sym]["date"].astype(str).str[:10] == date
            ]
            if day_data.empty:
                continue

            price = float(day_data["close"].iloc[0])
            pct_change = float(day_data.iloc[0].get("pct_change", 0))
            if pct_change >= 9.8:
                continue

            buy_shares = int(target_amount / price / 100) * 100
            if buy_shares >= 100:
                self._buy(sym, price, buy_shares, date)

    def _buy(self, symbol: str, price: float, shares: int, date: str):
        total = price * shares
        commission = max(self.MIN_COMMISSION, total * self.COMMISSION_RATE)
        buy_price = price * (1 + self.SLIPPAGE)

        if self.cash < total + commission:
            return

        if symbol in self.positions:
            pos = self.positions[symbol]
            new_cost = pos.avg_cost * pos.shares + total + commission
            pos.shares += shares
            pos.avg_cost = new_cost / pos.shares if pos.shares > 0 else 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol, shares=shares,
                avg_cost=(total + commission) / shares,
                current_price=price,
            )

        self.cash -= total + commission
        self.trades.append(Trade(
            date=date, symbol=symbol, action="BUY", price=price,
            shares=shares, amount=total, commission=commission,
        ))

    def _sell(self, symbol: str, price: float, shares: int, date: str):
        sell_price = price * (1 - self.SLIPPAGE)
        total = sell_price * shares
        commission = max(self.MIN_COMMISSION, total * self.COMMISSION_RATE)
        stamp_tax = total * self.STAMP_TAX_RATE

        self.cash += total - commission - stamp_tax

        pos = self.positions[symbol]
        pos.shares -= shares
        if pos.shares <= 0:
            del self.positions[symbol]

        self.trades.append(Trade(
            date=date, symbol=symbol, action="SELL", price=sell_price,
            shares=shares, amount=total, commission=commission,
            stamp_tax=stamp_tax,
        ))

    def _sell_all(self, symbol: str, price_data: Dict[str, pd.DataFrame], date: str):
        if symbol not in self.positions or symbol not in price_data:
            return
        day_data = price_data[symbol][
            price_data[symbol]["date"].astype(str).str[:10] == date
        ]
        if day_data.empty:
            return
        price = float(day_data["close"].iloc[0])
        pct_change = float(day_data.iloc[0].get("pct_change", 0))
        if pct_change <= -9.8:
            return
        self._sell(symbol, price, self.positions[symbol].shares, date)

    def _calc_equity(self, price_data: Dict[str, pd.DataFrame], date: str) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            df = price_data.get(sym)
            if df is not None:
                day = df[df["date"].astype(str).str[:10] == date]
                if not day.empty:
                    pos.current_price = float(day["close"].iloc[0])
                    equity += pos.market_value
        return equity

    def _calc_diversification(self, price_data: Dict[str, pd.DataFrame]) -> float:
        """计算组合分散化评分。"""
        # 简化：基于持仓标的之间的平均相关性
        if len(self.positions) < 2:
            return 1.0

        returns = {}
        for sym, pos in self.positions.items():
            if sym in price_data:
                df = price_data[sym]
                if len(df) > 20:
                    returns[sym] = df["close"].pct_change().dropna().tail(60)

        if len(returns) < 2:
            return 0.5

        ret_df = pd.DataFrame(returns)
        corr = ret_df.corr()

        n = corr.shape[0]
        off_diag = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag += abs(corr.iloc[i, j])
                count += 1

        avg_corr = off_diag / count if count > 0 else 1.0
        return round(max(0, 1.0 - avg_corr * 0.8), 4)

    def _get_all_dates(self, price_data: Dict[str, pd.DataFrame]) -> list:
        all_dates = set()
        for df in price_data.values():
            if "date" in df.columns:
                all_dates.update(df["date"].astype(str).str[:10].tolist())
        return sorted(all_dates)

    def _check_rebalance(self, date: str, last_date: str, frequency: str) -> bool:
        if last_date is None:
            return True

        if frequency == "daily":
            return True

        d = datetime.strptime(date, "%Y-%m-%d")
        ld = datetime.strptime(last_date, "%Y-%m-%d")
        diff = (d - ld).days

        if frequency == "weekly":
            return diff >= 5  # 大约一周
        elif frequency == "monthly":
            return diff >= 20  # 大约一月
        return False


# ── 报告 ───────────────────────────────────────────────────

def format_portfolio_result(result: PortfolioResult) -> str:
    """格式化组合回测结果。"""
    lines = [
        "╔══════════════════════════════════════╗",
        "║    多 资 产 组 合 回 测 报 告      ║",
        "╠══════════════════════════════════════╣",
        f"║  初始资金: ¥{result.initial_capital:,.0f}",
        f"║  最终权益: ¥{result.final_equity:,.0f}",
        f"║  总收益率: {result.total_return:+.2f}%",
        f"║  年化收益: {result.annual_return:+.2f}%",
        "╠══════════════════════════════════════╣",
        f"║  夏普比率: {result.sharpe_ratio:.2f}",
        f"║  最大回撤: {result.max_drawdown:.2f}%",
        f"║  胜率:     {result.win_rate:.1f}%",
        f"║  交易次数: {result.total_trades}",
        f"║  平均持仓: {result.avg_positions:.1f}",
        "╠══════════════════════════════════════╣",
        f"║  分散化评分: {result.diversification_score:.2f}",
        f"║  平均换手率: {result.avg_turnover:.3f}",
    ]

    if result.per_symbol_returns:
        lines.append("╠══════════════════════════════════════╣")
        lines.append("║  各标的收益:")
        for sym, ret in sorted(result.per_symbol_returns.items(),
                                key=lambda x: -x[1]):
            lines.append(f"║    {sym}: {ret:+.2f}%")

    if result.best_symbol:
        lines.append("║")
        lines.append(f"║  最佳: {result.best_symbol}  ({result.per_symbol_returns.get(result.best_symbol, 0):+.2f}%)")
    if result.worst_symbol:
        lines.append(f"║  最差: {result.worst_symbol}  ({result.per_symbol_returns.get(result.worst_symbol, 0):+.2f}%)")

    lines.append("╚══════════════════════════════════════╝")
    return "\n".join(lines)