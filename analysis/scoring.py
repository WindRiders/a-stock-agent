"""综合评分模块。

整合技术面、基本面、资金面、消息面的多维度评分。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from analysis.technical import TechnicalAnalyzer, TechResult
from analysis.fundamental import FundamentalAnalyzer, FundamentalResult
from data.market import MarketData
from data.fundamental import FundamentalData
from data.news import NewsData

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """股票综合评分。"""

    symbol: str
    name: str = ""

    # 各维度得分
    tech_score: int = 0        # -11 ~ +11
    fund_score: int = 0        # -6 ~ +6
    sentiment_score: int = 0   # -2 ~ +2
    capital_score: int = 0     # -2 ~ +2  资金面

    # 综合
    total_score: float = 0.0
    rating: str = "C"          # A/B/C/D

    # 详细信息
    signal: str = "HOLD"
    latest_price: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    volume_ratio: Optional[float] = None

    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class StockScorer:
    """综合评分引擎。

    权重分配：
    - 技术面：50%
    - 基本面：25%
    - 消息面：15%
    - 资金面：10%
    """

    def __init__(
        self,
        market_data: MarketData = None,
        fund_data: FundamentalData = None,
        news_data: NewsData = None,
    ):
        self.market_data = market_data or MarketData()
        self.fund_data = fund_data or FundamentalData()
        self.news_data = news_data or NewsData()
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer(self.fund_data)

    def score(self, symbol: str, name: str = "") -> StockScore:
        """对单只股票做综合评分。

        Args:
            symbol: 股票代码
            name: 股票名称（可选）

        Returns:
            StockScore 综合评分结果
        """
        score = StockScore(symbol=symbol, name=name)

        # 1. 获取K线数据
        try:
            df = self.market_data.get_daily_kline(symbol)
        except Exception as e:
            logger.error("获取 %s K线失败: %s", symbol, e)
            return score

        if df.empty or len(df) < 60:
            score.warnings.append("数据量不足（<60个交易日），评分可能不准确")
            return score

        # 2. 技术面分析（权重 50%）
        tech: TechResult = self.tech_analyzer.analyze(df, symbol)
        score.tech_score = tech.total_score
        score.signal = tech.signal
        score.latest_price = tech.latest_close
        score.volume_ratio = tech.volume_ratio_5

        # 记录技术面理由
        if tech.trend_score >= 2:
            score.reasons.append("均线多头排列，趋势向上")
        elif tech.trend_score <= -2:
            score.warnings.append("均线空头排列，趋势向下")

        if tech.macd_score >= 2:
            score.reasons.append("MACD 零轴上金叉")
        elif tech.macd_score <= -2:
            score.warnings.append("MACD 零轴下死叉")

        if tech.rsi_score >= 2:
            score.reasons.append("RSI 超卖区，存在反弹需求")
        elif tech.rsi_score <= -2:
            score.warnings.append("RSI 超买区，回调风险大")

        if tech.kdj_score >= 2:
            score.reasons.append("KDJ 超卖区")
        elif tech.kdj_score <= -2:
            score.warnings.append("KDJ 超买区")

        # 3. 基本面分析（权重 25%）
        try:
            fund: FundamentalResult = self.fund_analyzer.analyze(symbol)
            score.fund_score = fund.total_score
            score.pe = fund.pe
            score.pb = fund.pb

            if fund.value_score >= 2:
                score.reasons.append(f"估值较低（PE={fund.pe}，PB={fund.pb}）")
            elif fund.value_score <= -2:
                score.warnings.append("估值偏高")

            if fund.quality_score >= 2:
                score.reasons.append(f"盈利能力优秀（ROE={fund.roe}%）")
            elif fund.quality_score <= -2:
                score.warnings.append("盈利能力堪忧")

        except Exception as e:
            logger.warning("基本面分析 %s 失败: %s", symbol, e)

        # 4. 消息面分析（权重 15%）
        try:
            news = self.news_data.get_stock_news(symbol, limit=10)
            if not news.empty:
                texts = news.iloc[:, 0].tolist() if len(news.columns) > 0 else []
                sent = self.news_data.analyze_sentiment(texts)
                if sent > 0.3:
                    score.sentiment_score = 2
                    score.reasons.append("近期消息面偏正面")
                elif sent > 0.1:
                    score.sentiment_score = 1
                elif sent < -0.3:
                    score.sentiment_score = -2
                    score.warnings.append("近期消息面偏负面")
                elif sent < -0.1:
                    score.sentiment_score = -1
        except Exception as e:
            logger.debug("新闻分析 %s 失败: %s", symbol, e)

        # 5. 资金面分析（权重 10%）-- 基于成交量
        if tech.volume_ratio_5:
            if tech.volume_ratio_5 > 2:
                score.capital_score = 2
                score.reasons.append("近期显著放量，资金关注度高")
            elif tech.volume_ratio_5 > 1.5:
                score.capital_score = 1
            elif tech.volume_ratio_5 < 0.3:
                score.capital_score = -1
                score.warnings.append("成交量萎缩严重")

        # 6. 计算综合得分
        score.total_score = round(
            self._normalize(score.tech_score, -11, 11) * 0.50
            + self._normalize(score.fund_score, -6, 6) * 0.25
            + self._normalize(score.sentiment_score, -2, 2) * 0.15
            + self._normalize(score.capital_score, -2, 2) * 0.10,
            2,
        )

        # 7. 评级
        if score.total_score >= 0.7:
            score.rating = "A"
        elif score.total_score >= 0.4:
            score.rating = "B"
        elif score.total_score >= 0.1:
            score.rating = "C"
        else:
            score.rating = "D"

        return score

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """归一化到 0~1。"""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)