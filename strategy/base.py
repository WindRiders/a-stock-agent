"""策略基类。

定义策略接口和通用方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from analysis.scoring import StockScore


class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradeSignal:
    """交易信号。"""

    symbol: str
    name: str = ""
    signal: Signal = Signal.HOLD
    score: float = 0.0
    confidence: float = 0.0  # 0~1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reason: str = ""


class BaseStrategy(ABC):
    """交易策略抽象基类。"""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def generate_signals(
        self, scores: List[StockScore], market_df: pd.DataFrame = None
    ) -> List[TradeSignal]:
        """根据评分生成交易信号。

        Args:
            scores: 股票评分列表
            market_df: 市场指数K线数据（可选，用于风控）

        Returns:
            交易信号列表
        """
        pass

    def filter_by_market_condition(
        self, signals: List[TradeSignal], market_df: pd.DataFrame
    ) -> List[TradeSignal]:
        """根据市场环境过滤信号。

        例如大盘下跌趋势中抑制买入信号。
        """
        if market_df is None or market_df.empty:
            return signals

        # 简单判断：大盘在20日均线上方
        if "close" in market_df.columns:
            ma20 = market_df["close"].rolling(20).mean().iloc[-1]
            latest = market_df["close"].iloc[-1]
            is_bullish = latest > ma20
        else:
            is_bullish = True

        if not is_bullish:
            # 市场弱势，降低买入信号强度
            for s in signals:
                if s.signal in (Signal.BUY, Signal.STRONG_BUY):
                    s.signal = Signal.HOLD
                    s.reason += " (大盘弱势，暂缓买入)"

        return signals