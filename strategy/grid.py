"""网格交易策略。

在震荡市中，设定价格区间，在区间内低买高卖。
适合横盘震荡行情，不依赖趋势判断。
"""

from typing import List, Optional

import pandas as pd

from .base import BaseStrategy, TradeSignal, Signal
from analysis.scoring import StockScore


class GridTradingStrategy(BaseStrategy):
    """网格交易策略。

    核心逻辑：
    - 确定震荡区间（基于布林带或近期高低点）
    - 在区间下沿买入，上沿卖出
    - 配合技术评分过滤标的

    适用环境：震荡市（趋势策略在震荡市容易反复止损）
    """

    name = "grid"
    description = "网格交易：在震荡区间内低买高卖，适合横盘行情"

    def __init__(
        self,
        grid_levels: int = 5,
        grid_spacing_pct: float = 0.02,  # 网格间距 2%
    ):
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct

    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        signals = []

        for s in scores:
            signal_type = Signal.HOLD
            reason = ""

            # 网格策略更适合评分中等的震荡股
            # 太强（趋势股）不适合网格，太弱也不适合
            if 0.2 <= s.total_score <= 0.6:
                # 技术面偏中性（震荡特征）
                if -2 < s.tech_score < 4:
                    # 结合布林带位置判断（通过量比和评分间接判断）
                    if s.volume_ratio and s.volume_ratio < 1.2:
                        # 低成交量+中等评分 → 可能是震荡区间底部
                        signal_type = Signal.BUY
                        reason = "震荡区间底部，适合网格布局"
                    elif s.volume_ratio and s.volume_ratio > 2.0:
                        # 突发放量 → 可能突破，不适合网格
                        signal_type = Signal.HOLD
                        reason = "放量异动，网格暂停观察"
                    else:
                        signal_type = Signal.HOLD
                        reason = "震荡区间中部，等待更好位置"

            elif s.total_score < 0.2:
                # 评分过低，跌出网格区间 → 止损
                if s.tech_score <= -4:
                    signal_type = Signal.SELL
                    reason = "跌破震荡区间，止损退出"

            elif s.total_score > 0.7:
                # 太强，可能突破 → 不适合网格，转为趋势
                signal_type = Signal.HOLD
                reason = "评分过高，可能即将突破，不适合网格"

            signals.append(
                TradeSignal(
                    symbol=s.symbol,
                    name=s.name,
                    signal=signal_type,
                    score=s.total_score,
                    confidence=0.5 + abs(s.total_score - 0.4) * 0.5,
                    reason=reason,
                )
            )

        return self.filter_by_market_condition(signals, market_df)