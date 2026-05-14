"""多周期分析模块。

在日线分析基础上，叠加周线和月线维度。
核心思想：大周期定方向，小周期找买卖点。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from analysis.technical import TechnicalAnalyzer, TechResult

logger = logging.getLogger(__name__)


@dataclass
class MultiTimeframeResult:
    """多周期分析结果。"""

    symbol: str

    # 日线
    daily: Optional[TechResult] = None
    daily_signal: str = "HOLD"

    # 周线
    weekly: Optional[TechResult] = None
    weekly_signal: str = "HOLD"

    # 月线
    monthly: Optional[TechResult] = None
    monthly_signal: str = "HOLD"

    # 共振评分
    resonance_score: float = 0.0  # -1 ~ 1，越高越多周期看多
    resonance_grade: str = "C"    # A/B/C/D 共振等级

    # 综合信号（多周期加权）
    final_signal: str = "HOLD"
    signal_confidence: float = 0.0

    # 说明
    description: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MultiTimeframeAnalyzer:
    """多周期分析器。

    同时分析日线、周线、月线，检测多周期共振。

    权重：
    - 月线趋势：40%（大趋势决定方向）
    - 周线趋势：35%（中期趋势）
    - 日线信号：25%（买卖时机）
    """

    def __init__(self, market_data=None):
        self.market_data = market_data
        self.tech_analyzer = TechnicalAnalyzer()

        # 周期权重
        self.weights = {
            "monthly": 0.40,
            "weekly": 0.35,
            "daily": 0.25,
        }

    def analyze(
        self,
        daily_df: pd.DataFrame,
        symbol: str = "",
    ) -> MultiTimeframeResult:
        """执行多周期分析。

        Args:
            daily_df: 日线K线数据（需要足够的数据量以聚合周线和月线）
            symbol: 股票代码

        Returns:
            MultiTimeframeResult
        """
        result = MultiTimeframeResult(symbol=symbol)

        if daily_df.empty or len(daily_df) < 20:
            result.warnings.append("数据不足（需要至少20个交易日）")
            return result

        # 1. 日线分析
        result.daily = self.tech_analyzer.analyze(daily_df, symbol)
        result.daily_signal = result.daily.signal

        # 2. 周线聚合
        try:
            weekly_df = self._resample_to_weekly(daily_df)
            if len(weekly_df) >= 30:
                result.weekly = self.tech_analyzer.analyze(weekly_df, symbol)
                result.weekly_signal = result.weekly.signal
            else:
                result.warnings.append("周线数据不足（需要至少30周）")
        except Exception as e:
            logger.debug("周线分析失败: %s", e)
            result.warnings.append("周线分析失败")

        # 3. 月线聚合
        try:
            monthly_df = self._resample_to_monthly(daily_df)
            if len(monthly_df) >= 12:
                result.monthly = self.tech_analyzer.analyze(monthly_df, symbol)
                result.monthly_signal = result.monthly.signal
            else:
                result.warnings.append("月线数据不足（需要至少12个月）")
        except Exception as e:
            logger.debug("月线分析失败: %s", e)
            result.warnings.append("月线分析失败")

        # 4. 计算共振
        self._calc_resonance(result)

        # 5. 生成综合信号
        self._generate_final_signal(result)

        return result

    def _resample_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线→周线聚合。"""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # 计算周涨跌幅
        weekly["pct_change"] = weekly["close"].pct_change() * 100

        weekly = weekly.reset_index()
        weekly = weekly.rename(columns={"index": "date"})
        return weekly

    def _resample_to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """日线→月线聚合。"""
        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        monthly = df.resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        monthly["pct_change"] = monthly["close"].pct_change() * 100

        monthly = monthly.reset_index()
        return monthly

    def _calc_resonance(self, r: MultiTimeframeResult):
        """计算多周期共振评分。

        共振逻辑：
        - 三个周期全部看多 → 强共振 +1
        - 两个周期看多 → 中等共振 +0.5
        - 一个周期看多 → 弱共振 0
        - 全部看空 → 反向共振 -1
        """

        def signal_value(sig: str) -> float:
            mapping = {
                "STRONG_BUY": 2.0,
                "BUY": 1.0,
                "HOLD": 0.0,
                "SELL": -1.0,
                "STRONG_SELL": -2.0,
            }
            return mapping.get(sig, 0.0)

        monthly_val = signal_value(r.monthly_signal) if r.monthly else 0
        weekly_val = signal_value(r.weekly_signal) if r.weekly else 0
        daily_val = signal_value(r.daily_signal)

        # 加权共振评分
        r.resonance_score = (
            monthly_val * self.weights["monthly"]
            + weekly_val * self.weights["weekly"]
            + daily_val * self.weights["daily"]
        ) / 2.0  # 归一化到 -1~1

        # 共振等级
        if r.resonance_score >= 0.7:
            r.resonance_grade = "A"
            r.description = "多周期强烈共振看多，趋势明确"
        elif r.resonance_score >= 0.3:
            r.resonance_grade = "B"
            r.description = "多周期偏多，趋势向上但存在分歧"
        elif r.resonance_score >= -0.3:
            r.resonance_grade = "C"
            r.description = "多周期分歧，方向不明确"
        elif r.resonance_score >= -0.7:
            r.resonance_grade = "D"
            r.description = "多周期偏空，趋势向下"
        else:
            r.resonance_grade = "D"
            r.description = "多周期强烈共振看空，规避为主"

    def _generate_final_signal(self, r: MultiTimeframeResult):
        """根据共振生成最终信号。"""

        # 买条件：至少周线和日线不冲突
        if r.resonance_score >= 0.5:
            r.final_signal = "STRONG_BUY"
            r.signal_confidence = min(1.0, r.resonance_score)
        elif r.resonance_score >= 0.2:
            r.final_signal = "BUY"
            r.signal_confidence = min(1.0, r.resonance_score)
        elif r.resonance_score <= -0.5:
            r.final_signal = "STRONG_SELL"
            r.signal_confidence = min(1.0, abs(r.resonance_score))
        elif r.resonance_score <= -0.2:
            r.final_signal = "SELL"
            r.signal_confidence = min(1.0, abs(r.resonance_score))
        else:
            r.final_signal = "HOLD"
            r.signal_confidence = 0.3

        # 额外检查：月线空头+日线多头 → 警惕反弹陷阱
        if r.monthly and r.monthly_signal in ("SELL", "STRONG_SELL"):
            if r.daily_signal in ("BUY", "STRONG_BUY"):
                r.warnings.append("月线空头趋势中，日线买入可能是反弹陷阱")
                if r.final_signal in ("BUY", "STRONG_BUY"):
                    r.final_signal = "HOLD"
                    r.signal_confidence *= 0.5

        # 月线多头+日线超卖 → 黄金坑
        if r.monthly and r.monthly_signal in ("BUY", "STRONG_BUY"):
            if r.daily_signal in ("SELL", "STRONG_SELL"):
                r.description += "。月线多头中日线超卖，可能是加仓良机"

    def generate_report(self, r: MultiTimeframeResult) -> str:
        """生成多周期分析报告。"""
        lines = [
            "╔══════════════════════════════════╗",
            f"║   多周期分析 — {r.symbol}",
            "╠══════════════════════════════════╣",
            "",
            "━━━ 📊 周期信号 ━━━",
        ]

        signal_icons = {
            "STRONG_BUY": "🔥", "BUY": "📈", "HOLD": "⏸️",
            "SELL": "📉", "STRONG_SELL": "💀",
        }

        lines.append(f"  月线: {signal_icons.get(r.monthly_signal, '❓')} {r.monthly_signal}" if r.monthly else "  月线: 数据不足")
        lines.append(f"  周线: {signal_icons.get(r.weekly_signal, '❓')} {r.weekly_signal}" if r.weekly else "  周线: 数据不足")
        lines.append(f"  日线: {signal_icons.get(r.daily_signal, '❓')} {r.daily_signal}")

        lines.append("")
        lines.append("━━━ 🔗 共振分析 ━━━")
        lines.append(f"  共振评分: {r.resonance_score:+.2f}")
        lines.append(f"  共振等级: {r.resonance_grade}")
        lines.append(f"  {r.description}")

        lines.append("")
        lines.append("━━━ 🎯 综合信号 ━━━")
        lines.append(f"  信号: {signal_icons.get(r.final_signal, '')} {r.final_signal}")
        lines.append(f"  置信度: {r.signal_confidence:.0%}")

        if r.warnings:
            lines.append("")
            lines.append("━━━ ⚠️ 提醒 ━━━")
            for w in r.warnings:
                lines.append(f"  • {w}")

        lines.append("")
        lines.append("╚══════════════════════════════════╝")
        return "\n".join(lines)