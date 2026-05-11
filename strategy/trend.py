"""趋势跟踪策略。

识别均线多头排列的股票，跟踪趋势。
"""

from typing import List

import pandas as pd

from .base import BaseStrategy, TradeSignal, Signal
from analysis.scoring import StockScore


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略。

    核心逻辑：
    - 技术评分 > 0 且有基本面支撑 + 消息面不差
    - 综合评分 >= 0.5 时买入
    """

    name = "trend"
    description = "趋势跟踪：均线多头+量价配合的趋势策略"

    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        signals = []

        for s in scores:
            signal_type = Signal.HOLD
            reason = ""

            if s.total_score >= 0.7:
                signal_type = Signal.STRONG_BUY
                reason = f"综合评分优秀({s.total_score:.2f})"
            elif s.total_score >= 0.55:
                signal_type = Signal.BUY
                reason = f"综合评分良好({s.total_score:.2f})"
            elif s.total_score < 0.15:
                if s.fund_score <= -2:
                    signal_type = Signal.SELL
                    reason = "基本面差且综合评分低"

            signals.append(
                TradeSignal(
                    symbol=s.symbol,
                    name=s.name,
                    signal=signal_type,
                    score=s.total_score,
                    confidence=s.total_score,
                    reason=reason,
                )
            )

        return self.filter_by_market_condition(signals, market_df)