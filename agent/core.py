"""AI 交易 Agent 核心。

整合数据获取、技术分析、基本面评估、策略信号生成和回测验证。
提供统一的对外接口。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from agent.config import AgentConfig
from data.market import MarketData
from data.fundamental import FundamentalData
from data.news import NewsData
from analysis.technical import TechnicalAnalyzer
from analysis.fundamental import FundamentalAnalyzer
from analysis.scoring import StockScorer, StockScore
from strategy.base import TradeSignal, Signal
from strategy.factory import StrategyFactory
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import MetricsReport

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

        # 缓存
        self._stock_list: Optional[pd.DataFrame] = None
        self._scores: List[StockScore] = []
        self._signals: List[TradeSignal] = []
        self._market_kline: Optional[pd.DataFrame] = None

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

    def generate_report(self) -> str:
        """生成完整的投资分析报告。"""
        if not self._scores:
            self._scores = self.scan_market(verbose=False)
        if not self._signals:
            self.generate_signals()

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        strategy_name = self.config.strategy
        strategy_desc = self.strategy.description

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║           A 股 投 资 分 析 报 告                     ║",
            f"║  生成时间: {now}                    ║",
            f"║  策略: {strategy_desc}                               ║",
            "╠══════════════════════════════════════════════════════╣",
            "",
            "━━━ 📊 市场概况 ━━━",
        ]

        # 市场指数
        try:
            market_df = self._get_market_kline()
            if market_df is not None and not market_df.empty:
                latest = market_df["close"].iloc[-1]
                pct = float(market_df["pct_change"].iloc[-1]) if "pct_change" in market_df.columns else 0
                lines.append(f"  沪深300: {latest:.2f}  ({pct:+.2f}%)")
        except Exception:
            pass

        lines.append("")
        lines.append("━━━ 🎯 买入信号 ━━━")

        buy_signals = self.get_buy_signals()
        if buy_signals:
            for i, sig in enumerate(buy_signals[:10], 1):
                icon = "🔥" if sig.signal == Signal.STRONG_BUY else "📈"
                lines.append(
                    f"  {i:>2}. {icon} {sig.symbol} {sig.name:<8s} "
                    f"评分:{sig.score:.2f}  {sig.reason}"
                )
        else:
            lines.append("  (暂无符合条件的买入信号)")

        lines.append("")
        lines.append("━━━ 📋 综合评分 Top 10 ━━━")

        for i, score in enumerate(self._scores[:10], 1):
            signal_icon = {
                "STRONG_BUY": "🔥",
                "BUY": "📈",
                "HOLD": "⏸️",
                "SELL": "📉",
                "STRONG_SELL": "💀",
            }.get(score.signal, "❓")

            pe_str = f"PE:{score.pe:.1f}" if score.pe and score.pe > 0 else "PE:N/A"
            lines.append(
                f"  {i:>2}. {signal_icon} {score.symbol} {score.name:<8s} "
                f"总分:{score.total_score:.2f} 评级:{score.rating}  "
                f"技术:{score.tech_score:+d} 基本:{score.fund_score:+d}  "
                f"{pe_str}"
            )

        lines.append("")
        lines.append("━━━ ⚠️ 风险提示 ━━━")
        lines.append("  本报告由算法自动生成，仅供参考，不构成投资建议。")
        lines.append("  股市有风险，投资需谨慎。历史表现不代表未来收益。")
        lines.append("")
        lines.append("╚══════════════════════════════════════════════════════╝")

        return "\n".join(lines)

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