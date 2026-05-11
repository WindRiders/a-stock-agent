"""技术分析模块。

计算常用技术指标：MA, MACD, RSI, KDJ, BOLL, ATR, 成交量分析等。
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


@dataclass
class TechResult:
    """单只股票的技术分析结果。"""

    symbol: str
    # 趋势
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    trend_score: int = 0  # -2 ~ +2

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_score: int = 0  # -2 ~ +2

    # RSI
    rsi_6: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_score: int = 0  # -2 ~ +2

    # KDJ
    k: Optional[float] = None
    d: Optional[float] = None
    j: Optional[float] = None
    kdj_score: int = 0  # -2 ~ +2

    # 布林带
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: Optional[float] = None
    bb_score: int = 0  # -2 ~ +2

    # 成交量
    volume_ratio_5: Optional[float] = None  # 5日均量比
    volume_score: int = 0  # -1 ~ +1

    # 综合
    total_score: int = 0  # -11 ~ +11
    signal: str = "HOLD"  # BUY / STRONG_BUY / HOLD / SELL / STRONG_SELL

    # 最近价格
    latest_close: Optional[float] = None
    latest_date: Optional[str] = None


class TechnicalAnalyzer:
    """技术指标分析与信号生成。"""

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> TechResult:
        """对一只股票的K线数据做全面技术分析。

        Args:
            df: K线数据，需包含 close, high, low, volume 列
            symbol: 股票代码

        Returns:
            TechResult 包含所有技术指标值和评分
        """
        if df.empty or len(df) < 60:
            logger.warning("数据量不足（需要至少60个交易日）")
            return TechResult(symbol=symbol)

        result = TechResult(symbol=symbol)
        result.latest_close = float(df["close"].iloc[-1])
        result.latest_date = str(df["date"].iloc[-1])[:10]

        # 计算所有指标
        self._calc_ma(df, result)
        self._calc_macd(df, result)
        self._calc_rsi(df, result)
        self._calc_kdj(df, result)
        self._calc_bollinger(df, result)
        self._calc_volume(df, result)

        # 综合评分
        result.total_score = (
            result.trend_score
            + result.macd_score
            + result.rsi_score
            + result.kdj_score
            + result.bb_score
            + result.volume_score
        )

        # 生成买卖信号
        result.signal = self._generate_signal(result)

        return result

    def _calc_ma(self, df: pd.DataFrame, r: TechResult):
        """移动均线分析。"""
        close = df["close"]
        r.ma5 = float(close.rolling(5).mean().iloc[-1]) if len(df) >= 5 else None
        r.ma10 = float(close.rolling(10).mean().iloc[-1]) if len(df) >= 10 else None
        r.ma20 = float(close.rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
        r.ma60 = float(close.rolling(60).mean().iloc[-1]) if len(df) >= 60 else None

        if r.ma5 and r.ma10 and r.ma20 and r.ma60:
            # 多头排列：短期均线在长期之上
            if r.ma5 > r.ma10 > r.ma20 > r.ma60:
                r.trend_score = 2
            elif r.ma5 > r.ma10 > r.ma20:
                r.trend_score = 1
            # 空头排列
            elif r.ma5 < r.ma10 < r.ma20 < r.ma60:
                r.trend_score = -2
            elif r.ma5 < r.ma10 < r.ma20:
                r.trend_score = -1
            else:
                r.trend_score = 0

    def _calc_macd(self, df: pd.DataFrame, r: TechResult):
        """MACD 分析。"""
        macd_indicator = ta.trend.MACD(df["close"])
        r.macd = float(macd_indicator.macd().iloc[-1])
        r.macd_signal = float(macd_indicator.macd_signal().iloc[-1])
        r.macd_hist = float(macd_indicator.macd_diff().iloc[-1])

        # 金叉/死叉判断
        macd_vals = macd_indicator.macd().values
        signal_vals = macd_indicator.macd_signal().values

        if len(macd_vals) >= 2:
            golden_cross = (macd_vals[-2] <= signal_vals[-2]) and (macd_vals[-1] > signal_vals[-1])
            dead_cross = (macd_vals[-2] >= signal_vals[-2]) and (macd_vals[-1] < signal_vals[-1])

            if golden_cross and r.macd_hist > 0:
                r.macd_score = 2  # 零轴上金叉
            elif golden_cross:
                r.macd_score = 1  # 零轴下金叉
            elif dead_cross and r.macd_hist < 0:
                r.macd_score = -2  # 零轴下死叉
            elif dead_cross:
                r.macd_score = -1  # 零轴上死叉
            elif r.macd_hist > 0:
                r.macd_score = 1
            elif r.macd_hist < 0:
                r.macd_score = -1

    def _calc_rsi(self, df: pd.DataFrame, r: TechResult):
        """RSI 分析。"""
        r.rsi_6 = float(ta.momentum.RSIIndicator(df["close"], window=6).rsi().iloc[-1])
        r.rsi_14 = float(ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1])

        if r.rsi_14 is not None:
            if r.rsi_14 < 20:
                r.rsi_score = 2  # 超卖，反弹机会
            elif r.rsi_14 < 30:
                r.rsi_score = 1
            elif r.rsi_14 < 50:
                r.rsi_score = 0
            elif r.rsi_14 < 70:
                r.rsi_score = 0
            elif r.rsi_14 < 80:
                r.rsi_score = -1
            else:
                r.rsi_score = -2  # 超买

    def _calc_kdj(self, df: pd.DataFrame, r: TechResult):
        """KDJ 分析。"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 手动计算 KDJ（ta 库的 StochasticOscillator 与标准 KDJ 略有不同）
        low_list = low.rolling(9).min()
        high_list = high.rolling(9).max()
        rsv = (close - low_list) / (high_list - low_list) * 100

        k_vals = [50.0]
        d_vals = [50.0]
        for i in range(1, len(rsv)):
            if pd.notna(rsv.iloc[i]):
                k_vals.append(2 / 3 * k_vals[-1] + 1 / 3 * rsv.iloc[i])
                d_vals.append(2 / 3 * d_vals[-1] + 1 / 3 * k_vals[-1])
            else:
                k_vals.append(k_vals[-1])
                d_vals.append(d_vals[-1])

        r.k = round(k_vals[-1], 2)
        r.d = round(d_vals[-1], 2)
        r.j = round(3 * k_vals[-1] - 2 * d_vals[-1], 2)

        if r.j is not None:
            if r.j < 0:
                r.kdj_score = 2  # 严重超卖
            elif r.j < 20:
                r.kdj_score = 1
            elif r.j > 100:
                r.kdj_score = -2  # 严重超买
            elif r.j > 80:
                r.kdj_score = -1

    def _calc_bollinger(self, df: pd.DataFrame, r: TechResult):
        """布林带分析。"""
        bb = ta.volatility.BollingerBands(df["close"])
        r.bb_upper = float(bb.bollinger_hband().iloc[-1])
        r.bb_middle = float(bb.bollinger_mavg().iloc[-1])
        r.bb_lower = float(bb.bollinger_lband().iloc[-1])

        if r.latest_close and r.bb_upper and r.bb_lower:
            r.bb_position = (r.latest_close - r.bb_lower) / (r.bb_upper - r.bb_lower)
            if r.bb_position < 0.05:
                r.bb_score = 2  # 接触下轨，反弹概率大
            elif r.bb_position < 0.2:
                r.bb_score = 1
            elif r.bb_position > 0.95:
                r.bb_score = -2  # 接触上轨
            elif r.bb_position > 0.8:
                r.bb_score = -1

    def _calc_volume(self, df: pd.DataFrame, r: TechResult):
        """成交量分析。"""
        vol = df["volume"]
        vol_ma5 = vol.rolling(5).mean()
        r.volume_ratio_5 = (
            float(vol.iloc[-1] / vol_ma5.iloc[-1]) if vol_ma5.iloc[-1] > 0 else 1.0
        )

        # 放量上涨 / 缩量下跌 等判断
        pct_change = float(df["pct_change"].iloc[-1]) if "pct_change" in df.columns else 0

        if r.volume_ratio_5:
            if pct_change > 2 and r.volume_ratio_5 > 1.5:
                r.volume_score = 1  # 放量上涨
            elif pct_change > 0 and r.volume_ratio_5 > 2:
                r.volume_score = 1
            elif pct_change < -2 and r.volume_ratio_5 > 1.5:
                r.volume_score = -1  # 放量下跌
            elif pct_change < -1 and r.volume_ratio_5 < 0.5:
                r.volume_score = 1  # 缩量下跌（可能企稳）

    def _generate_signal(self, r: TechResult) -> str:
        """根据总分生成交易信号。"""
        if r.total_score >= 8:
            return "STRONG_BUY"
        elif r.total_score >= 4:
            return "BUY"
        elif r.total_score <= -8:
            return "STRONG_SELL"
        elif r.total_score <= -4:
            return "SELL"
        return "HOLD"

    def scan_market(
        self, market_data, stock_list: pd.DataFrame
    ) -> pd.DataFrame:
        """全市场扫描，对每只股票做技术分析并排序。

        Args:
            market_data: MarketData 实例
            stock_list: 股票列表

        Returns:
            按 total_score 降序排列的分析结果
        """
        results = []
        total = len(stock_list)
        for i, row in stock_list.iterrows():
            symbol = row["symbol"]
            try:
                df = market_data.get_daily_kline(symbol)
                if len(df) < 60:
                    continue
                tech = self.analyze(df, symbol)
                results.append(
                    {
                        "symbol": tech.symbol,
                        "name": row.get("name", ""),
                        "latest_close": tech.latest_close,
                        "trend_score": tech.trend_score,
                        "macd_score": tech.macd_score,
                        "rsi_score": tech.rsi_score,
                        "kdj_score": tech.kdj_score,
                        "bb_score": tech.bb_score,
                        "volume_score": tech.volume_score,
                        "total_score": tech.total_score,
                        "signal": tech.signal,
                        "rsi_14": tech.rsi_14,
                        "volume_ratio_5": tech.volume_ratio_5,
                    }
                )
            except Exception as e:
                logger.debug("分析 %s 失败: %s", symbol, e)
                continue

            if (i + 1) % 500 == 0:
                logger.info("扫描进度: %d/%d", i + 1, total)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("total_score", ascending=False)
        return result_df