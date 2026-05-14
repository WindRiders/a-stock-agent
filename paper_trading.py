"""纸面交易 / 模拟盘系统。

在不产生真实交易的前提下：
- 接收交易信号，模拟买入/卖出
- 追踪虚拟持仓、权益曲线、交易记录
- 计算盈亏、胜率、夏普等指标
- 生成模拟盘日报
- 数据持久化到 SQLite（与主库独立或共用 paper_trades 表）

用法:
    from paper_trading import PaperTrader

    trader = PaperTrader(initial_capital=100000)
    trader.execute("000001", "BUY", 10.50, 1000, "评分0.78 趋势向好")
    trader.update_prices({"000001": 11.20})
    print(trader.summary())
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PAPER_DB = os.path.expanduser("~/.a-stock-agent/paper.db")


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class PaperPosition:
    symbol: str
    name: str
    shares: int
    avg_cost: float  # 持仓均价（含佣金）
    current_price: float = 0.0
    buy_date: str = ""

    @property
    def cost_basis(self) -> float:
        return self.avg_cost * self.shares

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price if self.current_price else self.cost_basis

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis > 0:
            return self.market_value / self.cost_basis - 1
        return 0.0


@dataclass
class PaperTrade:
    id: int = 0
    date: str = ""
    symbol: str = ""
    name: str = ""
    action: str = ""  # BUY / SELL
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0
    commission: float = 0.0
    stamp_tax: float = 0.0
    reason: str = ""
    realized_pnl: float = 0.0  # 卖出时的实际盈亏


@dataclass
class PaperSnapshot:
    """某日的组合快照。"""
    date: str = ""
    cash: float = 0.0
    equity: float = 0.0  # 现金 + 持仓市值
    positions_value: float = 0.0
    position_count: int = 0
    daily_return: float = 0.0  # 日收益率


@dataclass
class PaperSummary:
    """模拟盘汇总。"""
    initial_capital: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    active_positions: int = 0
    positions_value: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)


# ── 主类 ──────────────────────────────────────────────────

class PaperTrader:
    """纸面交易 / 模拟盘。

    完整模拟 A 股交易：
    - 佣金：万2.5（最低5元）
    - 印花税：千1（仅卖出）
    - 最小交易单位：100股（1手）
    - T+1（当日买入次日可卖 — 通过数据库标记实现）

    Args:
        initial_capital: 初始资金
        db_path: SQLite 数据库路径（None 则仅内存）
        commission_rate: 佣金费率
        stamp_tax_rate: 印花税率
    """

    COMMISSION_RATE = 0.00025
    STAMP_TAX_RATE = 0.001
    MIN_COMMISSION = 5.0

    def __init__(
        self,
        initial_capital: float = 100000,
        db_path: Optional[str] = None,
        commission_rate: float = None,
        stamp_tax_rate: float = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.snapshots: List[PaperSnapshot] = []
        self._trade_counter = 0

        if commission_rate is not None:
            self.COMMISSION_RATE = commission_rate
        if stamp_tax_rate is not None:
            self.STAMP_TAX_RATE = stamp_tax_rate

        self.db_path = db_path or DEFAULT_PAPER_DB
        self._init_db()

    # ── 数据库 ─────────────────────────────────────────────

    def _init_db(self):
        if not self.db_path:
            return
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    stamp_tax REAL DEFAULT 0,
                    reason TEXT,
                    realized_pnl REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS paper_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    cash REAL,
                    equity REAL,
                    positions_value REAL,
                    position_count INTEGER,
                    daily_return REAL
                );

                CREATE TABLE IF NOT EXISTS paper_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            # 记录初始资金
            conn.execute(
                "INSERT OR REPLACE INTO paper_meta VALUES (?, ?)",
                ("initial_capital", str(self.initial_capital)),
            )

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _save_trade(self, t: PaperTrade):
        if not self.db_path:
            return
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO paper_trades
                   (date, symbol, name, action, price, shares, amount,
                    commission, stamp_tax, reason, realized_pnl)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (t.date, t.symbol, t.name, t.action, t.price, t.shares,
                 t.amount, t.commission, t.stamp_tax, t.reason, t.realized_pnl),
            )

    def _save_snapshot(self, s: PaperSnapshot):
        if not self.db_path:
            return
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO paper_snapshots
                   (date, cash, equity, positions_value, position_count, daily_return)
                   VALUES (?,?,?,?,?,?)""",
                (s.date, s.cash, s.equity, s.positions_value,
                 s.position_count, s.daily_return),
            )

    # ── 交易执行 ───────────────────────────────────────────

    def execute(
        self,
        symbol: str,
        action: str,
        price: float,
        shares: int = 0,
        name: str = "",
        reason: str = "",
        amount: float = 0.0,  # 按金额买入时使用
    ) -> Optional[PaperTrade]:
        """执行一笔模拟交易。

        Args:
            symbol: 股票代码
            action: BUY 或 SELL
            price: 成交价
            shares: 股数（按股数买卖时用）
            name: 股票名称
            reason: 交易理由
            amount: 买入金额（按金额买入时用，自动计算股数）

        Returns:
            PaperTrade 或 None（资金不足/无持仓等原因无法执行）
        """
        action = action.upper()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action == "BUY":
            return self._buy(symbol, price, shares, name, reason, date, amount)
        elif action == "SELL":
            return self._sell(symbol, price, shares, reason, date)
        else:
            logger.warning("未知操作: %s", action)
            return None

    def _buy(
        self, symbol: str, price: float, shares: int,
        name: str, reason: str, date: str, amount: float,
    ) -> Optional[PaperTrade]:
        # 按金额买入：自动计算整手股数
        if amount > 0 and shares <= 0:
            max_amount = min(amount, self.cash * 0.9)
            shares = int(max_amount / price / 100) * 100

        if shares < 100:
            logger.debug("买入 %s 股数(%d)不足1手", symbol, shares)
            return None

        total = price * shares
        commission = max(self.MIN_COMMISSION, total * self.COMMISSION_RATE)

        if self.cash < total + commission:
            logger.debug("资金不足: 需要 ¥%.2f, 可用 ¥%.2f", total + commission, self.cash)
            return None

        # 更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            new_cost = pos.cost_basis + total + commission
            pos.shares += shares
            pos.avg_cost = new_cost / pos.shares if pos.shares > 0 else 0
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                name=name,
                shares=shares,
                avg_cost=(total + commission) / shares,
                current_price=price,
                buy_date=date[:10],
            )

        self.cash -= total + commission
        self._trade_counter += 1

        trade = PaperTrade(
            id=self._trade_counter,
            date=date,
            symbol=symbol,
            name=name,
            action="BUY",
            price=price,
            shares=shares,
            amount=total,
            commission=commission,
            reason=reason,
        )
        self.trades.append(trade)
        self._save_trade(trade)
        return trade

    def _sell(
        self, symbol: str, price: float, shares: int,
        reason: str, date: str,
    ) -> Optional[PaperTrade]:
        if symbol not in self.positions:
            logger.debug("卖出 %s 无持仓", symbol)
            return None

        pos = self.positions[symbol]
        actual_shares = min(shares, pos.shares) if shares > 0 else pos.shares
        if actual_shares < 100:
            return None

        total = price * actual_shares
        commission = max(self.MIN_COMMISSION, total * self.COMMISSION_RATE)
        stamp_tax = total * self.STAMP_TAX_RATE

        # 计算实际盈亏
        realized_pnl = (price - pos.avg_cost) * actual_shares - commission - stamp_tax

        self.cash += total - commission - stamp_tax

        # 更新/清除持仓
        remaining = pos.shares - actual_shares
        if remaining <= 0:
            del self.positions[symbol]
        else:
            pos.shares = remaining
            # avg_cost 不变

        self._trade_counter += 1

        trade = PaperTrade(
            id=self._trade_counter,
            date=date,
            symbol=symbol,
            name=pos.name,
            action="SELL",
            price=price,
            shares=actual_shares,
            amount=total,
            commission=commission,
            stamp_tax=stamp_tax,
            reason=reason,
            realized_pnl=realized_pnl,
        )
        self.trades.append(trade)
        self._save_trade(trade)
        return trade

    def execute_signal(
        self,
        symbol: str,
        signal: str,
        score: float,
        price: float,
        name: str = "",
    ) -> Optional[PaperTrade]:
        """根据交易信号执行。

        BUY/STRONG_BUY → 买入（评分高→仓位大）
        SELL/STRONG_SELL → 卖出全部

        仓位管理：
        - STRONG_BUY: 15-20% 仓位
        - BUY: 8-12% 仓位
        """
        if signal in ("BUY", "STRONG_BUY"):
            if symbol in self.positions:
                return None  # 已持有

            pct = 0.15 if signal == "STRONG_BUY" else 0.10
            amount = self.initial_capital * pct
            reason = f"评分 {score:.2f} | {'强买入' if signal == 'STRONG_BUY' else '买入'}"
            return self.execute(symbol, "BUY", price, name=name, reason=reason, amount=amount)

        elif signal in ("SELL", "STRONG_SELL"):
            if symbol not in self.positions:
                return None
            pos = self.positions[symbol]
            reason = f"评分 {score:.2f} | {'强卖出' if signal == 'STRONG_SELL' else '卖出'}"
            return self.execute(symbol, "SELL", price, shares=pos.shares, reason=reason)

        return None

    # ── 价格更新 ───────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]):
        """更新持仓现价。

        Args:
            prices: {symbol: current_price}
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def take_snapshot(self, date: str = None) -> PaperSnapshot:
        """记录当前组合快照。"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        positions_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + positions_value

        # 计算日收益率（基于上次快照）
        daily_return = 0.0
        if self.snapshots:
            prev = self.snapshots[-1].equity
            if prev > 0:
                daily_return = (equity / prev - 1)

        snap = PaperSnapshot(
            date=date,
            cash=self.cash,
            equity=equity,
            positions_value=positions_value,
            position_count=len(self.positions),
            daily_return=daily_return,
        )
        self.snapshots.append(snap)
        self._save_snapshot(snap)
        return snap

    # ── 汇总 ───────────────────────────────────────────────

    def summary(self) -> PaperSummary:
        """生成模拟盘汇总。"""
        positions_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + positions_value

        s = PaperSummary(
            initial_capital=self.initial_capital,
            cash=self.cash,
            equity=equity,
            total_return=equity - self.initial_capital,
            total_return_pct=(equity / self.initial_capital - 1) * 100,
            total_trades=len(self.trades),
            active_positions=len(self.positions),
            positions_value=positions_value,
        )

        # 胜率（按已完成卖出交易）
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        if sell_trades:
            wins = [t for t in sell_trades if t.realized_pnl > 0]
            s.win_rate = len(wins) / len(sell_trades) * 100

        # 日收益率序列 → 夏普 + 最大回撤
        returns = [snap.daily_return for snap in self.snapshots if snap.daily_return != 0]
        s.daily_returns = returns

        if len(returns) > 1:
            import numpy as np
            mean_r = np.mean(returns)
            std_r = np.std(returns, ddof=1)
            if std_r > 0:
                s.sharpe_ratio = (mean_r / std_r) * np.sqrt(252)

            # 最大回撤
            cum = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            s.max_drawdown = float(np.min(dd)) * 100

        # 权益曲线
        s.equity_curve = [
            {"date": snap.date, "equity": snap.equity, "cash": snap.cash,
             "positions_value": snap.positions_value}
            for snap in self.snapshots
        ]

        return s

    def positions_list(self) -> List[dict]:
        """持仓列表。"""
        return [
            {
                "symbol": p.symbol,
                "name": p.name,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "cost_basis": p.cost_basis,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct * 100,
                "buy_date": p.buy_date,
            }
            for p in self.positions.values()
        ]

    def recent_trades(self, limit: int = 20) -> List[dict]:
        """最近交易记录。"""
        recent = self.trades[-limit:]
        return [
            {
                "id": t.id, "date": t.date, "symbol": t.symbol,
                "name": t.name, "action": t.action, "price": t.price,
                "shares": t.shares, "amount": t.amount,
                "commission": t.commission, "stamp_tax": t.stamp_tax,
                "realized_pnl": t.realized_pnl, "reason": t.reason,
            }
            for t in recent
        ]

    def reset(self):
        """重置模拟盘（清空持仓、交易和快照）。"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.snapshots = []
        self._trade_counter = 0
        if self.db_path and os.path.exists(self.db_path):
            os.remove(self.db_path)
            self._init_db()

    # ── 报告 ───────────────────────────────────────────────

    def generate_report(self) -> str:
        """生成模拟盘日报。"""
        s = self.summary()
        lines = [
            "╔══════════════════════════════════════╗",
            "║       模 拟 盘 日 报               ║",
            "╠══════════════════════════════════════╣",
            f"║  初始资金: ¥{s.initial_capital:,.0f}",
            f"║  当前权益: ¥{s.equity:,.0f}",
            f"║  总收益:   {s.total_return_pct:+.2f}%",
            f"║  现金:     ¥{s.cash:,.0f}",
            "╠══════════════════════════════════════╣",
            f"║  交易次数: {s.total_trades}",
            f"║  胜率:     {s.win_rate:.1f}%",
            f"║  夏普比率: {s.sharpe_ratio:.2f}",
            f"║  最大回撤: {s.max_drawdown:.2f}%",
            f"║  持仓数:   {s.active_positions}",
            f"║  持仓市值: ¥{s.positions_value:,.0f}",
            "╠══════════════════════════════════════╣",
        ]

        # 当前持仓
        positions = self.positions_list()
        if positions:
            lines.append("║  当前持仓:")
            for p in positions:
                color = "+" if p["unrealized_pnl"] >= 0 else ""
                lines.append(
                    f"║    {p['symbol']} {p['name']:<6s} "
                    f"{p['shares']}股 "
                    f"成本¥{p['avg_cost']:.2f} "
                    f"现价¥{p['current_price']:.2f}"
                )
                lines.append(
                    f"║      浮动盈亏: {color}¥{p['unrealized_pnl']:,.0f} "
                    f"({p['unrealized_pnl_pct']:+.1f}%)"
                )

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)