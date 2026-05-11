"""价值策略：低估值 + 高质量。

买入基本面优秀且估值合理的股票。
"""

from typing import List

import pandas as pd

from .base import BaseStrategy, TradeSignal, Signal
from analysis.scoring import StockScore


class ValueStrategy(BaseStrategy):
    """价值投资策略。

    核心逻辑：
    - 买入 PE 合理、ROE 高、基本面评分高的股票
    - 在技术面共振时加强信号
    """

    name = "value"
    description = "价值策略：低估值+高ROE的基本面选股"

    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        signals = []

        for s in scores:
            signal_type = Signal.HOLD
            reason = ""

            # 价值策略：基本面好 + 估值合理
            if s.fund_score >= 4 and s.tech_score >= 2:
                signal_type = Signal.STRONG_BUY
                reason = "基本面优秀，估值合理，技术面共振"
            elif s.fund_score >= 3:
                signal_type = Signal.BUY
                reason = f"基本面良好(fund_score={s.fund_score})"
            elif s.fund_score <= -3:
                signal_type = Signal.SELL
                reason = f"基本面恶化(fund_score={s.fund_score})"

            signals.append(
                TradeSignal(
                    symbol=s.symbol,
                    name=s.name,
                    signal=signal_type,
                    score=s.total_score,
                    confidence=min(1.0, max(0.0, (s.fund_score + 6) / 12.0)),
                    reason=reason,
                )
            )

        return self.filter_by_market_condition(signals, market_df)