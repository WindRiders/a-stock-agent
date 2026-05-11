"""动量策略：强者恒强。

买入技术面强势、成交活跃的股票。
"""

from typing import List

import pandas as pd

from .base import BaseStrategy, TradeSignal, Signal
from analysis.scoring import StockScore


class MomentumStrategy(BaseStrategy):
    """动量策略。

    核心逻辑：
    - 买入技术评分高（total_score > 8）且有放量配合的股票
    - 持有到技术面转弱（total_score < 0）
    """

    name = "momentum"
    description = "动量策略：买入技术面强势+放量的股票，捕捉趋势延续"

    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        signals = []

        for s in scores:
            signal_type = Signal.HOLD
            reason = ""

            if s.tech_score >= 8 and s.capital_score >= 1:
                signal_type = Signal.STRONG_BUY
                reason = f"技术面强势(score={s.tech_score})，资金活跃"
            elif s.tech_score >= 5 and s.capital_score >= 1:
                signal_type = Signal.BUY
                reason = f"技术面转强(score={s.tech_score})"
            elif s.tech_score <= -9:
                signal_type = Signal.STRONG_SELL
                reason = f"技术面严重走弱(score={s.tech_score})"
            elif s.tech_score <= -6:
                signal_type = Signal.SELL
                reason = f"技术面转弱(score={s.tech_score})"

            signals.append(
                TradeSignal(
                    symbol=s.symbol,
                    name=s.name,
                    signal=signal_type,
                    score=s.total_score,
                    confidence=min(1.0, max(0.0, s.tech_score / 11.0)),
                    reason=reason,
                )
            )

        return self.filter_by_market_condition(signals, market_df)