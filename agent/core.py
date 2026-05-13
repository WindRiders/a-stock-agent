"""AI 交易 Agent 核心。

整合数据获取、技术分析、基本面评估、策略信号生成、回测验证、
LLM智能分析和风险管理。
提供统一的对外接口。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agent.config import AgentConfig
from agent.llm import LLMAnalyzer
from agent.market_state import MarketStateDetector, MarketState
from data.market import MarketData
from data.fundamental import FundamentalData
from data.news import NewsData
from data.store import DataStore
from analysis.technical import TechnicalAnalyzer
from analysis.fundamental import FundamentalAnalyzer
from analysis.scoring import StockScorer, StockScore
from strategy.base import TradeSignal, Signal
from strategy.factory import StrategyFactory
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import MetricsReport
from risk import RiskManager, RiskLimits, PositionSize, PortfolioRisk

logger = logging.getLogger(__name__)


class TradingAgent:
    """A股量化交易 Agent。

    Usage:
        agent = TradingAgent()
        agent.scan_market()       # 全市场扫描
        agent.analyze("000001")   # 分析单只股票
        report = agent.generate_report()  # 生成投资报告
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.market_data = MarketData()
        self.fund_data = FundamentalData()
        self.news_data = NewsData()
        self.scorer = StockScorer(
            market_data=self.market_data,
            fund_data=self.fund_data,
            news_data=self.news_data,
        )
        self.strategy = StrategyFactory.get(self.config.strategy)
        self.backtest_engine = BacktestEngine()
        self.llm_analyzer = LLMAnalyzer(
            model=self.config.llm_model if self.config.llm_enabled else None,
            llm_config=self.config.get_llm_config(),
        )
        self.market_detector = MarketStateDetector()
        self.risk_manager = RiskManager(RiskLimits(
            max_single_position=self.config.max_position_pct,
            max_total_positions=self.config.max_positions,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        ))
        self.store = DataStore()

        # 缓存
        self._stock_list: Optional[pd.DataFrame] = None
        self._scores: List[StockScore] = []
        self._signals: List[TradeSignal] = []
        self._market_kline: Optional[pd.DataFrame] = None
        self._current_scan_id: Optional[int] = None

    # ── 市场扫描 ──────────────────────────────────────────

    def scan_market(self, top_n: int = None, verbose: bool = True) -> List[StockScore]:
        """扫描全市场，返回评分排序。

        先做全市场技术筛选（取前 N 名），再做深入分析。
        """
        if top_n is None:
            top_n = self.config.scan_top_n

        if verbose:
            logger.info("正在获取股票列表...")

        stock_list = self._get_stock_list()
        if stock_list.empty:
            logger.error("无法获取股票列表")
            return []

        if verbose:
            logger.info("获取实时行情...")
        try:
            quotes = self.market_data.get_realtime_quotes()
        except Exception:
            quotes = pd.DataFrame()

        # 选技术分最高的 N 只股票深入分析
        if verbose:
            logger.info("正在筛选技术面 Top %d...", top_n)

        # 用实时行情做快速筛选：涨跌幅、量比
        if not quotes.empty:
            merged = stock_list.merge(
                quotes[["symbol", "pct_change", "volume_ratio", "turnover_rate"]],
                on="symbol",
                how="left",
            )
        else:
            merged = stock_list

        # 优先分析成交活跃的股票
        if "volume_ratio" in merged.columns:
            merged = merged.sort_values("volume_ratio", ascending=False)
        if "pct_change" in merged.columns:
            # 避免追涨：涨跌幅 > 9% 的跳过（近涨停）
            merged = merged[merged["pct_change"].abs() < 9.5]

        candidates = merged.head(top_n * 3)  # 取 3 倍候选

        if verbose:
            logger.info("正在对 %d 只候选股进行深度分析...", len(candidates))

        scores = []
        for i, row in candidates.iterrows():
            symbol = row["symbol"]
            name = row.get("name", "")
            try:
                score = self.scorer.score(symbol, name)
                if score.rating != "D":
                    scores.append(score)
            except Exception as e:
                logger.debug("分析 %s 失败: %s", symbol, e)

        # 按总分排序
        scores.sort(key=lambda x: x.total_score, reverse=True)
        self._scores = scores[:top_n]

        # 自动持久化
        try:
            self._current_scan_id = self.store.save_scan(
                scores=self._scores,
                strategy=self.config.strategy,
            )
        except Exception as e:
            logger.warning("持久化扫描结果失败: %s", e)

        if verbose:
            logger.info("扫描完成！共分析 %d 只股票，有效评分 %d 只", len(candidates), len(self._scores))

        return self._scores

    # ── 单股分析 ──────────────────────────────────────────

    def analyze(self, symbol: str) -> StockScore:
        """深度分析单只股票。"""
        symbol = self.market_data.normalize_symbol(symbol)
        score = self.scorer.score(symbol)

        # 获取名称
        stock_list = self._get_stock_list()
        match = stock_list[stock_list["symbol"] == symbol]
        if not match.empty:
            score.name = match.iloc[0]["name"]

        return score

    def detail_report(self, symbol: str) -> str:
        """生成单只股票的详细分析报告。"""
        score = self.analyze(symbol)

        # 获取历史K线用于更多数据
        try:
            df = self.market_data.get_daily_kline(symbol)
            ma20 = float(df["close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
            ma60 = float(df["close"].rolling(60).mean().iloc[-1]) if len(df) >= 60 else None
            latest_close = float(df["close"].iloc[-1])
            week_ago_idx = max(0, len(df) - 6)
            week_ago_close = float(df["close"].iloc[week_ago_idx])
            week_change = (latest_close / week_ago_close - 1) * 100

            # 30日最高最低
            recent = df.tail(30)
            high_30 = float(recent["high"].max())
            low_30 = float(recent["low"].min())
        except Exception:
            ma20 = ma60 = latest_close = week_change = high_30 = low_30 = None

        lines = [
            "╔══════════════════════════════════════════════╗",
            f"║  {score.name or '---'} ({score.symbol}) 深度分析",
            "╠══════════════════════════════════════════════╣",
        ]

        # 基本信息
        if latest_close:
            lines.append(f"║  最新价: ¥{latest_close:.2f}")
        if ma20 and latest_close:
            vs_ma20 = (latest_close / ma20 - 1) * 100
            lines.append(f"║  MA20: ¥{ma20:.2f}  ({vs_ma20:+.1f}%)")
        if ma60 and latest_close:
            vs_ma60 = (latest_close / ma60 - 1) * 100
            lines.append(f"║  MA60: ¥{ma60:.2f}  ({vs_ma60:+.1f}%)")
        if week_change is not None:
            lines.append(f"║  周涨跌: {week_change:+.2f}%")

        # 基本面
        if score.pe:
            pe_str = f"{score.pe:.1f}" if score.pe > 0 else "亏损"
            lines.append(f"║  PE(TTM): {pe_str}")
        if score.pb:
            lines.append(f"║  PB: {score.pb:.2f}")

        # 评分
        lines.append("╠══════════════════════════════════════════════╣")
        lines.append(f"║  技术面评分: {score.tech_score:>3d} / +11")
        lines.append(f"║  基本面评分: {score.fund_score:>3d} / +6")
        lines.append(f"║  消息面评分: {score.sentiment_score:>3d} / +2")
        lines.append(f"║  资金面评分: {score.capital_score:>3d} / +2")
        lines.append(f"║  ─────────────────────────────")
        lines.append(f"║  综合评分: {score.total_score:.2f}")
        lines.append(f"║  评级: {score.rating}")
        lines.append(f"║  信号: {score.signal}")

        # 理由
        if score.reasons:
            lines.append("╠══════════════════════════════════════════════╣")
            lines.append("║  看多理由:")
            for r in score.reasons:
                lines.append(f"║    ✅ {r}")

        if score.warnings:
            lines.append("║  风险提示:")
            for w in score.warnings:
                lines.append(f"║    ⚠️  {w}")

        lines.append("╚══════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── 信号生成 ──────────────────────────────────────────

    def generate_signals(self) -> List[TradeSignal]:
        """基于当前评分生成交易信号。"""
        if not self._scores:
            self._scores = self.scan_market(verbose=False)

        market_df = self._get_market_kline()
        self._signals = self.strategy.generate_signals(self._scores, market_df)

        # 按信号强度排序
        signal_rank = {
            Signal.STRONG_BUY: 0,
            Signal.BUY: 1,
            Signal.HOLD: 2,
            Signal.SELL: 3,
            Signal.STRONG_SELL: 4,
        }
        self._signals.sort(key=lambda x: (signal_rank.get(x.signal, 99), -x.score))

        return self._signals

    def get_buy_signals(self) -> List[TradeSignal]:
        """获取买入信号列表。"""
        if not self._signals:
            self.generate_signals()
        return [s for s in self._signals if s.signal in (Signal.BUY, Signal.STRONG_BUY)]

    def get_sell_signals(self) -> List[TradeSignal]:
        """获取卖出信号列表。"""
        if not self._signals:
            self.generate_signals()
        return [s for s in self._signals if s.signal in (Signal.SELL, Signal.STRONG_SELL)]

    # ── 回测 ──────────────────────────────────────────

    def backtest(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """运行回测。"""
        engine = BacktestEngine(initial_capital=initial_capital)

        # 获取所有相关股票的价格数据
        symbols = signals_df["symbol"].unique()
        price_data = {}
        for sym in symbols:
            try:
                df = self.market_data.get_daily_kline(sym)
                df["date"] = pd.to_datetime(df["date"])
                price_data[sym] = df
            except Exception as e:
                logger.warning("获取 %s K线失败: %s", sym, e)

        result = engine.run(signals_df, price_data)
        return result

# ── 报告生成 ──────────────────────────────────────────

    def generate_report(self, use_llm: bool = False) -> str:
        """生成完整的投资分析报告。

        Args:
            use_llm: 是否使用LLM生成AI解读（需要 llm_enabled=True）
        """
        if not self._scores:
            self._scores = self.scan_market(verbose=False)
        if not self._signals:
            self.generate_signals()

        return self.llm_analyzer.generate_full_report(
            scores=self._scores,
            market_df=self._get_market_kline(),
            strategy_name=self.config.strategy,
        )

    def generate_ai_report(self) -> str:
        """生成AI增强版报告（使用LLM深度解读）。

        先用量化规则生成基础报告，再用LLM做深度解读。
        """
        if not self._scores:
            self._scores = self.scan_market(verbose=False)

        # 基础量化报告
        base = self.llm_analyzer.generate_full_report(
            scores=self._scores,
            market_df=self._get_market_kline(),
            strategy_name=self.config.strategy,
        )

        # 如果LLM未启用，直接返回基础报告
        if not self.config.llm_enabled:
            return base + "\n\n[提示] 启用 LLM 分析：在 AgentConfig 中设置 llm_enabled=True"

        # TODO: 实际调用 LLM 进行增强分析
        prompt = self.llm_analyzer.build_market_prompt(self._scores, self._get_market_kline())

        return base + f"\n\n[AI 分析提示词已生成，共 {len(prompt)} 字符]\n请在 LLM 模式下运行以获取 AI 解读。"

    # ── 仓位管理 ──────────────────────────────────────────

    def get_position_suggestions(
        self,
        total_capital: float,
        current_positions: Dict[str, float] = None,
    ) -> List[PositionSize]:
        """获取仓位建议。"""
        if not self._scores:
            self._scores = self.scan_market(verbose=False)

        return self.risk_manager.calculate_position_sizes(
            scores=self._scores,
            total_capital=total_capital,
            current_positions=current_positions,
        )

    def check_risk_alerts(
        self,
        positions: Dict[str, dict],
        price_data: Dict[str, pd.DataFrame],
    ) -> List[Dict]:
        """检查持仓的风险告警（止损/止盈）。"""
        return self.risk_manager.check_stop_conditions(positions, price_data)

    def generate_risk_report(
        self,
        total_capital: float,
        current_positions: Dict[str, float] = None,
    ) -> str:
        """生成风控分析报告。"""
        positions = self.get_position_suggestions(total_capital, current_positions)

        # 从历史扫描结果估算风险指标
        returns = []
        for s in self._scores[:5]:
            try:
                df = self.market_data.get_daily_kline(s.symbol)
                if len(df) > 30:
                    rets = df["close"].pct_change().dropna()
                    returns.append(rets)
            except Exception:
                pass

        risk = PortfolioRisk()
        if returns:
            # 用第一只股票的收益作为组合收益近似
            risk.volatility = float(np.std(returns[0]) * np.sqrt(252)) if returns else 0
            risk.max_drawdown = 0
            risk.sharpe = 0

        return self.risk_manager.generate_risk_report(
            positions=positions,
            risk_metrics=risk,
            total_capital=total_capital,
        )

    # ── 市场状态检测 ──────────────────────────────────────

    def detect_market_state(
        self, index_code: str = "000300"
    ) -> MarketState:
        """检测当前市场状态并推荐策略。

        Args:
            index_code: 指数代码，默认沪深300

        Returns:
            MarketState 包含状态判断和策略推荐
        """
        try:
            index_df = self.market_data.get_index_kline(index_code)
        except Exception as e:
            logger.error("获取指数数据失败: %s", e)
            return MarketState(warnings=[f"无法获取指数 {index_code} 的数据"])

        # 可选：获取市场宽度数据
        breadth = None
        try:
            quotes = self.market_data.get_realtime_quotes()
            if not quotes.empty and "pct_change" in quotes.columns:
                breadth = quotes
        except Exception:
            pass

        state = self.market_detector.detect(index_df, breadth)

        # 自动切换策略
        is_auto = getattr(self.config, 'auto_strategy', False)
        if is_auto and state.strategy_confidence > 0.6:
            old_strategy = self.config.strategy
            new_strategy = state.recommended_strategy
            if old_strategy != new_strategy:
                logger.info(
                    "市场状态触发策略切换: %s → %s (置信度: %.0f%%)",
                    old_strategy, new_strategy, state.strategy_confidence * 100,
                )
                self.config.strategy = new_strategy
                self.strategy = StrategyFactory.get(new_strategy)

        return state

    def generate_market_state_report(self) -> str:
        """生成市场状态诊断报告。"""
        state = self.detect_market_state()
        return self.market_detector.generate_report(state)

    # ── 历史查询 ──────────────────────────────────────────

    def get_history(self, limit: int = 20) -> List[Dict]:
        """获取最近的扫描历史。"""
        return self.store.get_scan_history(limit)

    def get_stock_history(self, symbol: str, limit: int = 30) -> List[Dict]:
        """获取某只股票的评分历史。"""
        return self.store.get_stock_history(symbol, limit)

    # ── 持仓管理 ──────────────────────────────────────────

    def get_portfolio(self) -> List[Dict]:
        """获取持仓列表。"""
        return self.store.get_positions()

    def get_portfolio_summary(self) -> Dict:
        """获取持仓汇总。"""
        return self.store.get_portfolio_summary()

    def record_buy(self, symbol: str, shares: int, price: float, name: str = "", reason: str = ""):
        """记录买入。"""
        self.store.add_position(symbol, shares, price, name)
        self.store.record_trade(symbol, "BUY", shares, price, name, reason=reason)

    def record_sell(self, symbol: str, shares: int, price: float, name: str = "", reason: str = ""):
        """记录卖出。"""
        commission = price * shares * 0.00025
        stamp_tax = price * shares * 0.001
        self.store.record_trade(
            symbol, "SELL", shares, price, name,
            commission=commission, stamp_tax=stamp_tax, reason=reason,
        )
        self.store.close_position(symbol, exit_price=price)

    def refresh_prices(self):
        """刷新持仓市价。"""
        try:
            from data.market import MarketData
            md = MarketData()
            quotes = md.get_realtime_quotes()
            if not quotes.empty:
                price_map = dict(zip(quotes["symbol"], quotes["price"]))
                self.store.update_positions_prices(price_map)
        except Exception as e:
            logger.warning("刷新持仓价格失败: %s", e)

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """获取交易记录。"""
        return self.store.get_trades(limit)

    def get_db_stats(self) -> Dict:
        """获取数据库统计。"""
        return self.store.get_stats()

    # ── 内部方法 ──────────────────────────────────────────

    def _get_stock_list(self) -> pd.DataFrame:
        if self._stock_list is None:
            try:
                self._stock_list = self.market_data.get_stock_list()
            except Exception as e:
                logger.error("获取股票列表失败: %s", e)
                self._stock_list = pd.DataFrame()
        return self._stock_list

    def _get_market_kline(self) -> Optional[pd.DataFrame]:
        if self._market_kline is None:
            try:
                self._market_kline = self.market_data.get_index_kline("000300")
            except Exception:
                pass
        return self._market_kline