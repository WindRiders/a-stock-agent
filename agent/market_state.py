"""市场状态检测模块。

自动判断当前市场处于什么状态：
- 趋势方向（多头/空头/震荡）
- 波动率水平（高/正常/低）
- 市场宽度（广度）

根据状态自动推荐最合适的交易策略。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态枚举。"""

    BULL_TREND = "bull_trend"         # 牛市趋势
    BEAR_TREND = "bear_trend"         # 熊市趋势
    SIDEWAYS = "sideways"             # 横盘震荡
    HIGH_VOL = "high_volatility"      # 高波动
    PANIC = "panic"                   # 恐慌暴跌
    RECOVERY = "recovery"             # 超跌反弹
    UNKNOWN = "unknown"


@dataclass
class MarketState:
    """市场状态分析结果。"""

    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_cn: str = "未知"

    # 趋势指标
    trend_direction: str = "neutral"  # "up" / "down" / "neutral"
    trend_strength: float = 0.0       # 0~1
    ma_alignment: str = "mixed"       # "bullish" / "bearish" / "mixed"

    # 波动指标
    volatility: float = 0.0           # 年化波动率
    volatility_regime: str = "normal"  # "high" / "normal" / "low"
    max_drawdown_30d: float = 0.0

    # 市场宽度
    above_ma20_pct: float = 0.0       # 在20日均线上方的股票比例
    breadth_signal: str = "neutral"

    # 最新指数数据
    latest_close: float = 0.0
    latest_change_pct: float = 0.0

    # 策略推荐
    recommended_strategy: str = "trend"
    strategy_confidence: float = 0.0
    strategy_reason: str = ""

    # 风险提示
    risk_level: str = "medium"  # "high" / "medium" / "low"
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MarketStateDetector:
    """市场状态检测器。

    通过分析大盘指数K线数据，判断当前市场状态。
    支持：趋势判断、波动率分析、市场宽度、策略推荐。
    """

    # 关键均线周期
    MA_SHORT = 5
    MA_MEDIUM = 20
    MA_LONG = 60

    def __init__(self):
        pass

    # ── 主检测 ──────────────────────────────────────────────

    def detect(
        self,
        index_df: pd.DataFrame,
        market_breadth_data: Optional[pd.DataFrame] = None,
    ) -> MarketState:
        """检测当前市场状态。

        Args:
            index_df: 指数K线数据（需含 close, high, low, volume, pct_change）
            market_breadth_data: 全市场股票涨跌数据（可选，用于市场宽度分析）

        Returns:
            MarketState
        """
        if index_df.empty or len(index_df) < 60:
            return MarketState(warnings=["数据不足，无法判断市场状态（需要至少60个交易日）"])

        state = MarketState()

        # 1. 趋势分析
        self._analyze_trend(index_df, state)

        # 2. 波动率分析
        self._analyze_volatility(index_df, state)

        # 3. 市场宽度
        if market_breadth_data is not None:
            self._analyze_breadth(market_breadth_data, state)

        # 4. 综合判断 regime
        self._determine_regime(state)

        # 5. 策略推荐
        self._recommend_strategy(state)

        # 6. 风险等级
        self._assess_risk(state)

        return state

    # ── 趋势分析 ────────────────────────────────────────────

    def _analyze_trend(self, df: pd.DataFrame, state: MarketState):
        """分析趋势方向和强度。"""
        close = df["close"]

        # 计算均线
        ma5 = float(close.rolling(self.MA_SHORT).mean().iloc[-1])
        ma20 = float(close.rolling(self.MA_MEDIUM).mean().iloc[-1])
        ma60 = float(close.rolling(self.MA_LONG).mean().iloc[-1])
        latest = float(close.iloc[-1])

        state.latest_close = latest
        if "pct_change" in df.columns:
            state.latest_change_pct = float(df["pct_change"].iloc[-1])

        # 均线排列判断
        if latest > ma5 > ma20 > ma60:
            state.ma_alignment = "bullish"
            state.trend_direction = "up"
            state.trend_strength = min(1.0, (latest / ma60 - 1) * 3)  # 偏离60日线幅度
        elif latest < ma5 < ma20 < ma60:
            state.ma_alignment = "bearish"
            state.trend_direction = "down"
            state.trend_strength = min(1.0, (1 - latest / ma60) * 3)
        elif latest > ma20:
            state.ma_alignment = "mixed"
            state.trend_direction = "up"
            state.trend_strength = abs(latest / ma20 - 1) * 2
        elif latest < ma20:
            state.ma_alignment = "mixed"
            state.trend_direction = "down"
            state.trend_strength = abs(latest / ma20 - 1) * 2
        else:
            state.trend_direction = "neutral"
            state.trend_strength = 0.0

        # 连续涨跌天数
        if "pct_change" in df.columns:
            recent = df.tail(10)
            up_days = sum(1 for v in recent["pct_change"] if v > 0)
            down_days = sum(1 for v in recent["pct_change"] if v < 0)

            if up_days >= 7:
                state.trend_strength = min(1.0, state.trend_strength + 0.2)
            elif down_days >= 7:
                state.trend_strength = min(1.0, state.trend_strength + 0.2)

    # ── 波动率分析 ──────────────────────────────────────────

    def _analyze_volatility(self, df: pd.DataFrame, state: MarketState):
        """分析波动率水平。"""
        close = df["close"]

        # 日收益率
        returns = close.pct_change().dropna()

        if len(returns) < 5:
            return

        # 年化波动率
        state.volatility = float(np.std(returns.tail(60)) * np.sqrt(252))

        # 近期 vs 长期波动率对比
        recent_vol = float(np.std(returns.tail(10)) * np.sqrt(252))
        long_vol = float(np.std(returns.tail(60)) * np.sqrt(252))

        if long_vol > 0:
            vol_ratio = recent_vol / long_vol
            if vol_ratio > 2.0:
                state.volatility_regime = "high"
                state.warnings.append(f"近期波动急剧放大（{vol_ratio:.1f}x），注意风险")
            elif vol_ratio > 1.5:
                state.volatility_regime = "high"
            elif vol_ratio < 0.5:
                state.volatility_regime = "low"
            else:
                state.volatility_regime = "normal"
        else:
            state.volatility_regime = "normal"

        # 30日最大回撤
        if len(close) >= 30:
            recent = close.tail(30)
            peak = recent.cummax()
            drawdown = (recent - peak) / peak
            state.max_drawdown_30d = float(drawdown.min())

    # ── 市场宽度 ────────────────────────────────────────────

    def _analyze_breadth(
        self, breadth_df: pd.DataFrame, state: MarketState
    ):
        """分析市场宽度。"""
        # breadth_df 预期包含每只股票的 pct_change
        if breadth_df.empty:
            return

        try:
            # 涨跌比
            up_count = int((breadth_df["pct_change"] > 0).sum())
            total_count = len(breadth_df)
            if total_count > 0:
                state.above_ma20_pct = up_count / total_count

                if state.above_ma20_pct > 0.7:
                    state.breadth_signal = "bullish"
                elif state.above_ma20_pct < 0.3:
                    state.breadth_signal = "bearish"
                else:
                    state.breadth_signal = "neutral"
        except Exception:
            pass

    # ── 综合状态判断 ────────────────────────────────────────

    def _determine_regime(self, state: MarketState):
        """综合判断市场状态。"""
        # 恐慌判断优先：高波动 + 大回撤
        if (
            state.volatility_regime == "high"
            and state.max_drawdown_30d < -0.10
        ):
            state.regime = MarketRegime.PANIC
            state.regime_cn = "恐慌暴跌"
            return

        # 超跌反弹：之前大跌 + 最近反弹
        if state.max_drawdown_30d < -0.08 and state.trend_direction == "up":
            state.regime = MarketRegime.RECOVERY
            state.regime_cn = "超跌反弹"
            return

        # 高波动不确定性
        if state.volatility_regime == "high":
            state.regime = MarketRegime.HIGH_VOL
            state.regime_cn = "高波动"
            return

        # 趋势判断
        if state.ma_alignment == "bullish" and state.trend_strength > 0.3:
            state.regime = MarketRegime.BULL_TREND
            state.regime_cn = "牛市趋势"
        elif state.ma_alignment == "bearish" and state.trend_strength > 0.3:
            state.regime = MarketRegime.BEAR_TREND
            state.regime_cn = "熊市趋势"
        else:
            state.regime = MarketRegime.SIDEWAYS
            state.regime_cn = "横盘震荡"

    # ── 策略推荐 ────────────────────────────────────────────

    def _recommend_strategy(self, state: MarketState):
        """根据市场状态推荐最适合的策略。"""

        recommendations = {
            MarketRegime.BULL_TREND: ("momentum", 0.85, "趋势确认向上，动量策略可捕捉强势股"),
            MarketRegime.BEAR_TREND: ("value", 0.6, "市场下行，防御为主。价值策略关注低估值标的"),
            MarketRegime.SIDEWAYS: ("grid", 0.75, "震荡市适合网格交易，低买高卖积累差价"),
            MarketRegime.HIGH_VOL: ("trend", 0.5, "高波动环境，趋势策略需设宽止损"),
            MarketRegime.PANIC: ("mean_reversion", 0.55, "恐慌后往往有超跌反弹机会，均值回归策略"),
            MarketRegime.RECOVERY: ("mean_reversion", 0.7, "反弹初期，均值回归策略捕捉修复行情"),
        }

        rec = recommendations.get(state.regime, ("trend", 0.5, "默认趋势策略"))
        state.recommended_strategy = rec[0]
        state.strategy_confidence = rec[1]
        state.strategy_reason = rec[2]

    # ── 风险评估 ────────────────────────────────────────────

    def _assess_risk(self, state: MarketState):
        """评估当前风险等级。"""
        risk_score = 0

        if state.volatility_regime == "high":
            risk_score += 3
        elif state.volatility_regime == "low":
            risk_score -= 1

        if state.max_drawdown_30d < -0.05:
            risk_score += 2
        if state.max_drawdown_30d < -0.10:
            risk_score += 2

        if state.trend_direction == "down":
            risk_score += 2

        if state.regime == MarketRegime.PANIC:
            risk_score += 3

        if risk_score >= 5:
            state.risk_level = "high"
            state.warnings.append("⚠️ 当前市场风险较高，建议降低仓位或观望")
        elif risk_score >= 2:
            state.risk_level = "medium"
        else:
            state.risk_level = "low"

    # ── 报告生成 ────────────────────────────────────────────

    def generate_report(self, state: MarketState) -> str:
        """生成市场状态分析报告。"""
        lines = [
            "╔══════════════════════════════════╗",
            "║      市 场 状 态 诊 断          ║",
            "╠══════════════════════════════════╣",
            f"║  状态: {state.regime_cn}",
            f"║  风险: {state.risk_level}",
            "╠══════════════════════════════════╣",
            "",
            "━━━ 📈 趋势 ━━━",
            f"  方向: {state.trend_direction}",
            f"  强度: {state.trend_strength:.2f}",
            f"  均线排列: {state.ma_alignment}",
            "",
            "━━━ 📊 波动 ━━━",
            f"  年化波动率: {state.volatility*100:.1f}%",
            f"  波动状态: {state.volatility_regime}",
            f"  30日最大回撤: {state.max_drawdown_30d*100:.1f}%",
            "",
            "━━━ 🎯 策略推荐 ━━━",
            f"  推荐策略: {state.recommended_strategy}",
            f"  置信度: {state.strategy_confidence:.0%}",
            f"  理由: {state.strategy_reason}",
        ]

        if state.warnings:
            lines.append("")
            lines.append("━━━ ⚠️ 风险提示 ━━━")
            for w in state.warnings:
                lines.append(f"  {w}")

        lines.append("")
        lines.append("╚══════════════════════════════════╝")
        return "\n".join(lines)