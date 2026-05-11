"""均值回归策略。

基于"涨多了会跌，跌多了会涨"的均值回归原理。
寻找超卖反弹或超买回调的机会。
"""

from typing import List

import pandas as pd

from .base import BaseStrategy, TradeSignal, Signal
from analysis.scoring import StockScore


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略。

    核心逻辑：
    - 寻找短期超卖（RSI < 30 或股价远离均线）的标的
    - 预期回归均值 → 买入
    - 回归到均线附近 → 卖出
    - 跌破支撑不回归 → 止损

    适用环境：震荡市或慢趋势中回调时
    风险：趋势市中可能逆势操作，需要严格止损
    """

    name = "mean_reversion"
    description = "均值回归：超卖反弹+超买回调，捕捉短期回归机会"

    def __init__(
        self,
        oversold_threshold: float = 0.25,  # 评分低于此视为超卖
        overbought_threshold: float = 0.75,  # 评分高于此视为超买
        hold_days: int = 5,  # 预期持有天数
        strict_stop: bool = True,  # 严格止损模式
    ):
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.hold_days = hold_days
        self.strict_stop = strict_stop

    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        signals = []

        for s in scores:
            signal_type = Signal.HOLD
            reason = ""

            # 均值回归寻找超卖机会
            if s.total_score <= self.oversold_threshold:
                # 基本面不能太差（避免价值陷阱）
                if s.fund_score >= -2:
                    # 技术面超卖 + 基本面OK → 均值回归买入
                    if s.tech_score <= -2:
                        signal_type = Signal.STRONG_BUY
                        reason = f"严重超卖（评分{s.total_score:.2f}），基本面尚可，均值回归机会"
                    else:
                        signal_type = Signal.BUY
                        reason = f"偏弱（评分{s.total_score:.2f}），关注回归可能"

                else:
                    # 基本面差 + 超卖 → 可能是价值陷阱
                    signal_type = Signal.HOLD
                    reason = "超卖但基本面差，警惕价值陷阱"

            elif s.total_score >= self.overbought_threshold:
                # 超买 → 卖出/减仓
                if s.tech_score >= 6:
                    signal_type = Signal.STRONG_SELL
                    reason = f"严重超买（评分{s.total_score:.2f}），回调风险大"
                else:
                    signal_type = Signal.SELL
                    reason = f"偏高（评分{s.total_score:.2f}），建议减仓"

            else:
                # 回归到合理区间 → 持仓或平仓
                if 0.4 <= s.total_score <= 0.6:
                    signal_type = Signal.HOLD
                    reason = "已回归合理估值区间"
                else:
                    signal_type = Signal.HOLD
                    reason = ""

            signals.append(
                TradeSignal(
                    symbol=s.symbol,
                    name=s.name,
                    signal=signal_type,
                    score=s.total_score,
                    confidence=1.0 - abs(s.total_score - 0.5) * 2,
                    reason=reason,
                )
            )

        return self.filter_by_market_condition(signals, market_df)