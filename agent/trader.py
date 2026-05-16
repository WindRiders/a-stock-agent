"""
自主交易调度引擎 (Autonomous Trading Engine)。

将分析→信号→执行→监控→通知串联成完整的交易闭环。

两种运行模式：
  - paper（模拟盘）: 使用 PaperTrader，零风险，用于策略验证和训练
  - live（实盘）:     使用 Broker 接口，真金白银（需配置券商凭证）

运行模式：
  - once（单次）:   扫描→执行→报告→退出，适合 cron 定时任务
  - monitor（监控）: 持续运行，交易时段内轮询行情和持仓

用法:
    from agent.trader import AutoTrader

    trader = AutoTrader(mode="paper", capital=100000)
    trader.run_once()       # 单次运行
    # trader.run_monitor()  # 持续监控

CLI:
    python cli.py trade run        # 单次自主交易
    python cli.py trade monitor    # 启动持续监控
    python cli.py trade status     # 查看状态
    python cli.py trade dashboard  # 交易仪表盘
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.config import AgentConfig
from agent.config_file import ConfigManager
from agent.core import TradingAgent
from agent.market_state import MarketStateDetector
from agent.notifier import Notifier
from data.market import MarketData
from data.store import DataStore
from paper_trading import PaperTrader, PaperSummary
from risk import RiskLimits, RiskManager
from strategy.factory import StrategyFactory

logger = logging.getLogger(__name__)


# ── 枚举 ──────────────────────────────────────────────────

class TraderMode(Enum):
    PAPER = "paper"          # 模拟盘
    LIVE = "live"            # 实盘


class RunMode(Enum):
    ONCE = "once"            # 单次运行
    MONITOR = "monitor"      # 持续监控


class TradePhase(Enum):
    IDLE = "IDLE"
    PRE_MARKET = "PRE_MARKET"         # 盘前准备
    MARKET_OPEN = "MARKET_OPEN"       # 盘中交易
    LUNCH_BREAK = "LUNCH_BREAK"       # 午休
    POST_MARKET = "POST_MARKET"       # 盘后总结
    STOPPED = "STOPPED"


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class TradeSession:
    """一次交易会话的状态记录。"""
    session_id: str = ""
    date: str = ""
    mode: TraderMode = TraderMode.PAPER
    phase: TradePhase = TradePhase.IDLE
    market_state: str = ""
    recommended_strategy: str = ""
    total_capital: float = 0
    cash: float = 0
    equity: float = 0
    positions_count: int = 0
    positions_value: float = 0
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    signals_generated: int = 0
    orders_executed: int = 0
    orders_rejected: int = 0
    alerts_triggered: int = 0
    started_at: str = ""
    updated_at: str = ""


@dataclass
class TradeDecision:
    """单笔交易决策。"""
    symbol: str
    name: str
    action: str          # BUY / SELL / HOLD
    signal: str           # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    score: float
    rating: str
    price: float
    amount: float = 0     # 建议金额
    shares: int = 0       # 建议股数
    pct_of_capital: float = 0  # 占总资金比例
    reason: str = ""
    strategy: str = ""
    stop_loss: float = 0
    take_profit: float = 0


# ── 交易引擎 ──────────────────────────────────────────────

class AutoTrader:
    """自主交易引擎。

    核心闭环：
    1. 市场状态检测 → 自适应策略选择
    2. 全市场扫描 → 多因子评分
    3. 信号生成 → 仓位计算
    4. 风控检查 → 订单执行
    5. 持仓监控 → 止损止盈
    6. 日报生成 → 通知推送

    Args:
        mode: 'paper' 模拟盘 或 'live' 实盘
        capital: 初始资金（模拟盘）
        run_mode: 'once' 单次 或 'monitor' 持续
        strategy: 强制指定策略（None=自适应选择）
        risk_limits: 风控参数
        config: AgentConfig
    """

    # A股交易时间（北京时间）
    TRADING_START = dtime(9, 30)
    TRADING_END = dtime(15, 0)
    LUNCH_START = dtime(11, 30)
    LUNCH_END = dtime(13, 0)
    PRE_MARKET_SCAN_TIME = dtime(9, 0)   # 盘前扫描时间
    POST_MARKET_REPORT_TIME = dtime(15, 0)  # 盘后报告时间

    def __init__(
        self,
        mode: str = "paper",
        capital: float = 100000,
        run_mode: str = "once",
        strategy: str = None,
        risk_limits: RiskLimits = None,
        config: AgentConfig = None,
    ):
        self.mode = TraderMode(mode)
        self.run_mode = RunMode(run_mode)
        self.initial_capital = capital
        self.config = config or AgentConfig()

        # 风控
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_manager = RiskManager(self.risk_limits)

        # 核心组件
        self.agent = TradingAgent(self.config)
        self.market_detector = MarketStateDetector()
        self.market_data = MarketData()
        self.notifier = Notifier()
        self.store = DataStore()
        self.config_manager = ConfigManager()

        # 策略
        self._strategy_override = strategy
        self.current_strategy = strategy or self.config.strategy

        # 模拟盘
        self.paper_trader = PaperTrader(initial_capital=capital)

        # 实盘券商（延迟初始化）
        self._live_broker = None

        # 状态
        self.phase = TradePhase.IDLE
        self.session: Optional[TradeSession] = None
        self._decisions: List[TradeDecision] = []
        self._daily_orders = 0
        self._daily_loss = 0.0
        self._running = False
        self._prices_cache: Dict[str, float] = {}
        self._position_cache: Dict[str, dict] = {}

        # 持久化路径
        self._log_dir = os.path.expanduser("~/.a-stock-agent/logs")
        self._state_file = os.path.expanduser("~/.a-stock-agent/trader_state.json")

    # ── 运行入口 ────────────────────────────────────────────

    def run_once(self, top_n: int = 50, dry_run: bool = False) -> TradeSession:
        """单次完整交易循环。

        盘前扫描 → 信号生成 → 执行 → 盘后报告。

        Args:
            top_n: 扫描股票数量
            dry_run: 仅分析不执行（用于盘前预演）

        Returns:
            TradeSession 会话摘要
        """
        self._begin_session()
        logger.info("=" * 60)
        logger.info("自主交易引擎启动 | 模式: %s | 资金: ¥%s",
                     self.mode.value, f"{self.initial_capital:,.0f}")
        logger.info("=" * 60)

        # ── Phase 1: 市场状态检测 ──
        self._phase_pre_market()

        # ── Phase 2: 全市场扫描 ──
        self._phase_scan(top_n)

        # ── Phase 3: 信号生成 & 仓位计算 ──
        self._phase_signals()

        # ── Phase 4: 执行交易 ──
        if not dry_run:
            self._phase_execute()
        else:
            logger.info("[DRY RUN] 跳过实际执行，仅生成决策报告")

        # ── Phase 5: 盘后总结 ──
        self._phase_post_market()

        # ── 通知 ──
        self._send_daily_report()

        return self.session

    def run_monitor(self, interval: int = 60):
        """持续监控模式。

        交易时段内轮询：行情 → 持仓检查 → 止损/止盈 → 新信号。

        Args:
            interval: 轮询间隔（秒）
        """
        self._begin_session()
        self._running = True
        logger.info("🔄 持续监控模式启动，轮询间隔 %d 秒", interval)

        try:
            while self._running:
                now = datetime.now().time()
                phase = self._get_current_phase(now)

                if phase == TradePhase.MARKET_OPEN:
                    self._monitor_tick()
                elif phase == TradePhase.LUNCH_BREAK:
                    # 午休：暂停操作
                    time.sleep(60)
                    continue
                elif phase == TradePhase.POST_MARKET:
                    # 收盘：生成报告后停止
                    self._phase_post_market()
                    self._send_daily_report()
                    break
                else:
                    # 非交易时间，等待
                    wait = self._seconds_until_market_open()
                    logger.info("⏰ 非交易时间，等待 %d 分钟后市场开盘", wait // 60)
                    time.sleep(min(wait, 600))  # 最多等10分钟
                    continue

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("监控已手动停止")
        finally:
            self._running = False
            self._save_state()

    def stop(self):
        """停止监控。"""
        self._running = False
        self.phase = TradePhase.STOPPED

    # ── 各阶段实现 ──────────────────────────────────────────

    def _begin_session(self):
        """初始化会话。"""
        date = datetime.now().strftime("%Y-%m-%d")
        self.session = TradeSession(
            session_id=f"{date}-{int(time.time())}",
            date=date,
            mode=self.mode,
            phase=TradePhase.IDLE,
            total_capital=self.initial_capital,
            cash=self.paper_trader.cash if self.mode == TraderMode.PAPER else 0,
            started_at=datetime.now().isoformat(),
        )

        # 重置当日计数器
        self._daily_orders = 0
        self._daily_loss = 0.0
        self._decisions = []

    def _phase_pre_market(self):
        """盘前准备：市场状态检测、策略选择。"""
        logger.info("")
        logger.info("── Phase 1: 盘前准备 ──")
        self.phase = TradePhase.PRE_MARKET

        # 检测市场状态
        try:
            # 获取指数K线数据作为市场状态检测输入
            index_df = self.market_data.get_index_kline("000001")  # 上证指数
            state = self.market_detector.detect(index_df)
            if state:
                self.session.market_state = state.regime_cn
                self.session.recommended_strategy = state.recommended_strategy

                logger.info("  市场状态: %s", state.regime_cn)
                logger.info("  趋势方向: %s | 波动: %s | 风险: %s",
                            state.trend_direction, state.volatility_regime, state.risk_level)
                logger.info("  推荐策略: %s (置信度 %.0f%%)",
                            state.recommended_strategy, state.strategy_confidence * 100)

                # 自适应策略切换
                if not self._strategy_override and state.strategy_confidence >= 0.6:
                    self.current_strategy = state.recommended_strategy
                    self.agent.strategy = StrategyFactory.get(self.current_strategy)
                    logger.info("  ✓ 已自动切换策略: %s", self.current_strategy)
        except Exception as e:
            logger.warning("  市场状态检测失败: %s", e)

    def _phase_scan(self, top_n: int):
        """全市场扫描并评分。"""
        logger.info("")
        logger.info("── Phase 2: 全市场扫描 (Top %d) ──", top_n)

        self.phase = TradePhase.MARKET_OPEN

        try:
            scores = self.agent.scan_market(top_n=top_n, verbose=False)
            self.agent._scores = scores
            logger.info("  ✓ 扫描完成，有效评分: %d 只", len(scores))

            if scores:
                top3 = scores[:3]
                for i, s in enumerate(top3, 1):
                    logger.info("    #%d %s %s 评分:%.2f 信号:%s PE:%s 价格:¥%s",
                                i, s.symbol, s.name, s.total_score,
                                s.signal, s.pe or "-", s.latest_price or "-")
        except Exception as e:
            logger.error("  扫描失败: %s", e)

    def _phase_signals(self):
        """生成交易信号和仓位建议。"""
        logger.info("")
        logger.info("── Phase 3: 信号生成 ──")

        scores = self.agent._scores
        if not scores:
            logger.info("  无有效评分，跳过信号生成")
            return

        # 生成信号
        signals = self.agent.generate_signals()
        buy_signals = [s for s in signals if s.signal in ("BUY", "STRONG_BUY")]
        sell_signals = [s for s in signals if s.signal in ("SELL", "STRONG_SELL")]

        logger.info("  买入信号: %d 条 | 卖出信号: %d 条",
                     len(buy_signals), len(sell_signals))
        self.session.signals_generated = len(buy_signals) + len(sell_signals)

        # 仓位计算（买入信号）
        if buy_signals:
            positions = self.risk_manager.calculate_position_sizes(
                scores=[s for s in scores if s.symbol in [x.symbol for x in buy_signals]],
                total_capital=(self.paper_trader.cash +
                              sum(p.market_value for p in self.paper_trader.positions.values())
                              if self.mode == TraderMode.PAPER else self.initial_capital),
            )

            # 转换为 TradeDecision
            for ps in positions[:self.risk_limits.max_total_positions]:
                # 找对应的评分
                match = next((s for s in scores if s.symbol == ps.symbol), None)
                if not match:
                    continue

                amount = self.initial_capital * ps.suggested_pct
                shares = int(amount / (match.latest_price or 10) / 100) * 100

                decision = TradeDecision(
                    symbol=ps.symbol,
                    name=ps.name,
                    action="BUY",
                    signal="BUY",
                    score=ps.score,
                    rating=match.rating,
                    price=match.latest_price or 0,
                    amount=amount,
                    shares=shares,
                    pct_of_capital=ps.suggested_pct,
                    reason=ps.reason,
                    strategy=self.current_strategy,
                    stop_loss=ps.stop_loss_price or 0,
                    take_profit=ps.take_profit_price or 0,
                )
                self._decisions.append(decision)

                logger.info("  📈 %s %s | 评分:%.2f | 仓位:%.1f%% | ¥%s | 止损:%.2f 止盈:%.2f",
                            ps.symbol, ps.name, ps.score,
                            ps.suggested_pct * 100, f"{amount:,.0f}",
                            ps.stop_loss_price or 0, ps.take_profit_price or 0)

        # 卖出信号（针对已有持仓）
        if sell_signals:
            for sig in sell_signals:
                if sig.symbol in self.paper_trader.positions:
                    pos = self.paper_trader.positions[sig.symbol]
                    profit_pct = (pos.current_price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0

                    decision = TradeDecision(
                        symbol=sig.symbol,
                        name=sig.name,
                        action="SELL",
                        signal=sig.signal,
                        score=sig.score,
                        rating=sig.rating,
                        price=sig.latest_price or 0,
                        shares=pos.shares,
                        reason=f"卖出信号 | 持仓盈亏 {profit_pct:+.1f}%",
                        strategy=self.current_strategy,
                    )
                    self._decisions.append(decision)

                    logger.info("  📉 %s %s | 卖出信号 | 持仓 %d 股 | 盈亏 %+.1f%%",
                                sig.symbol, sig.name, pos.shares, profit_pct)

    def _phase_execute(self):
        """执行交易决策。"""
        logger.info("")
        logger.info("── Phase 4: 执行交易 ──")

        if not self._decisions:
            logger.info("  无交易决策，跳过执行")
            return

        # 先执行卖出（释放资金）
        sell_decisions = [d for d in self._decisions if d.action == "SELL"]
        buy_decisions = [d for d in self._decisions if d.action == "BUY"]

        # 卖出
        for d in sell_decisions:
            if self._daily_orders >= self.risk_limits.max_total_positions * 3:
                logger.warning("  ⚠ 当日下单次数已达上限，跳过后续")
                break

            trade = self.paper_trader.execute(
                symbol=d.symbol,
                action="SELL",
                price=d.price,
                shares=d.shares,
                reason=d.reason,
            )

            if trade:
                self.session.orders_executed += 1
                self._daily_orders += 1
                logger.info("  ✓ 卖出 %s %s | %d股 @ ¥%.2f | 盈亏 ¥%+.0f",
                            d.symbol, d.name, d.shares, d.price, trade.realized_pnl)
            else:
                self.session.orders_rejected += 1

        # 更新可用资金
        available_cash = self.paper_trader.cash

        # 买入
        for d in buy_decisions:
            if self._daily_orders >= self.risk_limits.max_total_positions * 3:
                logger.warning("  ⚠ 当日下单次数已达上限")
                break

            # 检查持仓数量
            if len(self.paper_trader.positions) >= self.risk_limits.max_total_positions:
                logger.info("  ⚠ 持仓已达上限 %d 只，停止买入",
                            self.risk_limits.max_total_positions)
                break

            # 风控检查
            if d.pct_of_capital > self.risk_limits.max_single_position:
                logger.info("  ⚠ %s 仓位 %.1f%% 超限，调整至 %.0f%%",
                            d.symbol, d.pct_of_capital * 100,
                            self.risk_limits.max_single_position * 100)
                d.pct_of_capital = self.risk_limits.max_single_position
                d.amount = self.initial_capital * d.pct_of_capital
                d.shares = int(d.amount / d.price / 100) * 100

            if d.shares < 100:
                logger.info("  ⚠ %s 股数 %d 不足1手，跳过", d.symbol, d.shares)
                continue

            if d.amount > available_cash * 0.9:
                logger.info("  ⚠ %s 需 ¥%s > 可用 ¥%s，跳过",
                            d.symbol, f"{d.amount:,.0f}", f"{available_cash:,.0f}")
                continue

            # 执行买入
            trade = self.paper_trader.execute(
                symbol=d.symbol,
                action="BUY",
                price=d.price,
                shares=d.shares if d.shares >= 100 else 0,
                name=d.name,
                amount=d.amount,
                reason=f"{d.reason} | {d.signal} | 策略:{d.strategy}",
            )

            if trade:
                self.session.orders_executed += 1
                self._daily_orders += 1
                available_cash -= trade.amount + trade.commission
                logger.info("  ✓ 买入 %s %s | %d股 @ ¥%.2f | ¥%s | 止损:%.2f",
                            d.symbol, d.name, d.shares, d.price,
                            f"{trade.amount:,.0f}", d.stop_loss)
            else:
                self.session.orders_rejected += 1

    def _phase_post_market(self):
        """盘后：更新持仓价格、记录快照、生成总结。"""
        logger.info("")
        logger.info("── Phase 5: 盘后总结 ──")
        self.phase = TradePhase.POST_MARKET

        # 更新持仓现价
        if self.paper_trader.positions:
            try:
                quotes = self.market_data.get_realtime_quotes()
                if not quotes.empty:
                    price_map = {}
                    for _, row in quotes.iterrows():
                        price_map[row["symbol"]] = float(row.get("price", 0))
                    self.paper_trader.update_prices(price_map)
                    logger.info("  ✓ 更新 %d 只持仓现价", len(self.paper_trader.positions))
            except Exception as e:
                logger.warning("  更新现价失败: %s", e)

        # 记录快照
        self.paper_trader.take_snapshot()

        # 生成汇总
        summary = self.paper_trader.summary()

        # 更新会话状态
        self.session.cash = summary.cash
        self.session.equity = summary.equity
        self.session.positions_count = summary.active_positions
        self.session.positions_value = summary.positions_value
        self.session.daily_pnl = summary.total_return
        self.session.daily_pnl_pct = summary.total_return_pct
        self.session.updated_at = datetime.now().isoformat()

        # 打印总结
        logger.info("  ┌─────────────────────────────────┐")
        logger.info("  │  📊 交易总结                    │")
        logger.info("  ├─────────────────────────────────┤")
        logger.info("  │  总资产:   ¥%10s     │", f"{summary.equity:,.0f}")
        logger.info("  │  现金:     ¥%10s     │", f"{summary.cash:,.0f}")
        logger.info("  │  持仓市值: ¥%10s     │", f"{summary.positions_value:,.0f}")
        logger.info("  │  持仓数:   %10d              │", summary.active_positions)
        logger.info("  │  累计收益: ¥%+8s (%+5.1f%%) │",
                     f"{summary.total_return:,.0f}", summary.total_return_pct)
        logger.info("  │  累计交易: %8d 笔           │", summary.total_trades)
        logger.info("  │  胜率:     %8.1f%%           │", summary.win_rate)
        logger.info("  │  夏普:     %8.2f             │", summary.sharpe_ratio)
        logger.info("  │  最大回撤: %7.2f%%           │", summary.max_drawdown)
        logger.info("  └─────────────────────────────────┘")

        # 打印持仓明细
        if self.paper_trader.positions:
            logger.info("")
            logger.info("  当前持仓:")
            for symbol, pos in self.paper_trader.positions.items():
                pnl = pos.unrealized_pnl
                pnl_pct = pos.unrealized_pnl_pct * 100
                logger.info("    %s %-8s | %4d股 | 成本¥%.2f | 现价¥%.2f | 盈亏 %+.1f%%",
                            symbol, pos.name, pos.shares,
                            pos.avg_cost, pos.current_price, pnl_pct)

        # 止损检查
        if self.paper_trader.positions:
            self._check_stops()

    def _monitor_tick(self):
        """单次监控轮询。"""
        now = datetime.now().strftime("%H:%M:%S")

        if not self.paper_trader.positions:
            return  # 无持仓，无需监控

        # 更新现价
        try:
            quotes = self.market_data.get_realtime_quotes()
            if not quotes.empty:
                price_map = {}
                for symbol in self.paper_trader.positions:
                    match = quotes[quotes["symbol"] == symbol]
                    if not match.empty:
                        price_map[symbol] = float(match.iloc[0].get("price", 0))
                self.paper_trader.update_prices(price_map)
        except Exception:
            pass

        # 止损/止盈检查
        self._check_stops()

    def _check_stops(self):
        """检查持仓止损/止盈条件，触发后自动卖出。"""
        positions_dict = {}
        for symbol, pos in self.paper_trader.positions.items():
            positions_dict[symbol] = {
                "symbol": symbol,
                "avg_cost": pos.avg_cost,
                "shares": pos.shares,
                "current_price": pos.current_price,
                "peak_price": getattr(pos, "peak_price", pos.current_price),
            }

        alerts = self.risk_manager.check_stop_conditions(
            positions=positions_dict,
            price_data={},
        )

        for alert in alerts:
            symbol = alert["symbol"]
            action = alert["action"]
            logger.warning("  🚨 %s %s | %s | 价格 ¥%.2f",
                          symbol, action, alert["reason"], alert["current_price"])

            # 自动执行卖出
            if action in ("STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP"):
                trade = self.paper_trader.execute(
                    symbol=symbol,
                    action="SELL",
                    price=alert["current_price"],
                    reason=alert["reason"],
                )
                if trade:
                    self.session.orders_executed += 1
                    self.session.alerts_triggered += 1
                    logger.info("  ✓ 自动卖出 %s | 盈亏 ¥%+.0f", symbol, trade.realized_pnl)

                    # 触发通知
                    self._send_alert(
                        f"🔔 {'止损' if action == 'STOP_LOSS' else '止盈'}卖出\n"
                        f"{symbol} @ ¥{alert['current_price']:.2f}\n"
                        f"盈亏: {alert['pnl_pct']*100:+.1f}%\n"
                        f"理由: {alert['reason']}"
                    )

    # ── 通知 ────────────────────────────────────────────────

    def _send_daily_report(self):
        """发送每日交易报告。"""
        summary = self.paper_trader.summary()

        # 持仓列表
        positions_text = ""
        for symbol, pos in self.paper_trader.positions.items():
            pnl_pct = pos.unrealized_pnl_pct * 100
            positions_text += f"  {symbol} {pos.name} | {pos.shares}股 | 盈亏 {pnl_pct:+.1f}%\n"

        if not positions_text:
            positions_text = "  (空仓)\n"

        # 今日信号
        signals_text = ""
        for d in self._decisions[:10]:
            emoji = "📈" if d.action == "BUY" else "📉"
            signals_text += f"  {emoji} {d.symbol} {d.name} | {d.signal} | 评分{d.score:.2f}\n"

        if not signals_text:
            signals_text = "  无信号\n"

        message = (
            f"A股自主交易日报 {self.session.date}\n"
            f"\n"
            f"📊 账户概览\n"
            f"  总资产: ¥{summary.equity:,.0f}\n"
            f"  现金: ¥{summary.cash:,.0f}\n"
            f"  市值: ¥{summary.positions_value:,.0f}\n"
            f"  收益: {summary.total_return_pct:+.2f}%\n"
            f"  胜率: {summary.win_rate:.1f}% | 夏普: {summary.sharpe_ratio:.2f}\n"
            f"\n"
            f"💼 持仓 ({summary.active_positions}只)\n"
            f"{positions_text}"
            f"\n"
            f"📡 今日信号\n"
            f"{signals_text}"
            f"\n"
            f"📋 策略: {self.current_strategy} | 市场: {self.session.market_state}\n"
            f"执行: {self.session.orders_executed}笔 | 拒绝: {self.session.orders_rejected}笔\n"
        )

        self.notifier.broadcast(message, title=f"A股日报 {self.session.date}")

        # 同时保存到文件
        report_path = os.path.join(
            os.path.expanduser("~/.a-stock-agent/reports"),
            f"trade_{self.session.date}.md",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        md_report = self._format_markdown_report(summary)
        with open(report_path, "w") as f:
            f.write(md_report)
        logger.info("  📄 报告已保存: %s", report_path)

    def _send_alert(self, message: str):
        """发送即时告警。"""
        self.notifier.broadcast(message, title="⚠ A股交易告警")

    def _format_markdown_report(self, summary: PaperSummary) -> str:
        """格式化 Markdown 报告。"""
        lines = [
            f"# A股自主交易报告",
            f"",
            f"**日期**: {self.session.date}",
            f"**策略**: {self.current_strategy} | **市场**: {self.session.market_state}",
            f"**模式**: {self.mode.value}",
            f"",
            f"## 账户概览",
            f"",
            f"| 指标 | 数值 |",
            f"|---|---|",
            f"| 总资产 | ¥{summary.equity:,.0f} |",
            f"| 现金 | ¥{summary.cash:,.0f} |",
            f"| 持仓市值 | ¥{summary.positions_value:,.0f} |",
            f"| 累计收益 | {summary.total_return_pct:+.2f}% |",
            f"| 胜率 | {summary.win_rate:.1f}% |",
            f"| 夏普比率 | {summary.sharpe_ratio:.2f} |",
            f"| 最大回撤 | {summary.max_drawdown:.2f}% |",
            f"| 累计交易 | {summary.total_trades} 笔 |",
            f"",
            f"## 当前持仓",
            f"",
        ]

        if self.paper_trader.positions:
            lines.append("| 代码 | 名称 | 股数 | 成本 | 现价 | 盈亏 |")
            lines.append("|---|---|---|---|---|---|")
            for symbol, pos in self.paper_trader.positions.items():
                pnl_pct = pos.unrealized_pnl_pct * 100
                lines.append(
                    f"| {symbol} | {pos.name} | {pos.shares} | ¥{pos.avg_cost:.2f} | "
                    f"¥{pos.current_price:.2f} | {pnl_pct:+.1f}% |"
                )
        else:
            lines.append("(空仓)")
            lines.append("")

        lines.extend([
            "",
            f"## 今日决策",
            "",
            f"| 代码 | 名称 | 信号 | 评分 | 动作 | 金额 | 理由 |",
            f"|---|---|---|---|---|---|---|",
        ])

        for d in self._decisions:
            lines.append(
                f"| {d.symbol} | {d.name} | {d.signal} | {d.score:.2f} | "
                f"{d.action} | ¥{d.amount:,.0f} | {d.reason} |"
            )

        if not self._decisions:
            lines.append("| (无) | | | | | |")
            lines.append("")

        lines.extend([
            "",
            f"---",
            f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"a-stock-agent 自主交易引擎*",
        ])

        return "\n".join(lines)

    # ── 状态管理 ────────────────────────────────────────────

    def _save_state(self):
        """保存当前状态到文件。"""
        state = {
            "date": self.session.date if self.session else "",
            "mode": self.mode.value,
            "strategy": self.current_strategy,
            "phase": self.phase.value,
            "cash": self.paper_trader.cash,
            "equity": self.paper_trader.summary().equity,
            "positions": {
                s: {
                    "name": p.name,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                }
                for s, p in self.paper_trader.positions.items()
            },
            "daily_orders": self._daily_orders,
            "updated_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> dict:
        """加载上次状态。"""
        if os.path.exists(self._state_file):
            with open(self._state_file) as f:
                return json.load(f)
        return {}

    def status(self) -> dict:
        """返回当前状态摘要。"""
        summary = self.paper_trader.summary()
        return {
            "mode": self.mode.value,
            "strategy": self.current_strategy,
            "phase": self.phase.value,
            "market_state": self.session.market_state if self.session else "",
            "capital": self.initial_capital,
            "cash": summary.cash,
            "equity": summary.equity,
            "total_return_pct": summary.total_return_pct,
            "total_trades": summary.total_trades,
            "win_rate": summary.win_rate,
            "sharpe": summary.sharpe_ratio,
            "max_drawdown": summary.max_drawdown,
            "positions": len(self.paper_trader.positions),
            "daily_orders": self._daily_orders,
        }

    # ── 工具 ────────────────────────────────────────────────

    def _get_current_phase(self, now: dtime) -> TradePhase:
        """根据当前时间判断交易阶段。"""
        if now < self.PRE_MARKET_SCAN_TIME:
            return TradePhase.IDLE
        elif now < self.TRADING_START:
            return TradePhase.PRE_MARKET
        elif now < self.LUNCH_START:
            return TradePhase.MARKET_OPEN
        elif now < self.LUNCH_END:
            return TradePhase.LUNCH_BREAK
        elif now < self.TRADING_END:
            return TradePhase.MARKET_OPEN
        else:
            return TradePhase.POST_MARKET

    def _seconds_until_market_open(self) -> int:
        """计算距离下一个交易时段开始还有多少秒。"""
        now = datetime.now()
        today_open = now.replace(
            hour=self.PRE_MARKET_SCAN_TIME.hour,
            minute=self.PRE_MARKET_SCAN_TIME.minute,
            second=0, microsecond=0,
        )
        if now > today_open:
            today_open = today_open.replace(day=now.day + 1)
        return int((today_open - now).total_seconds())