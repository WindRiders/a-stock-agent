"""基本面分析模块。

评估估值合理性、成长性、盈利质量、机构关注度等。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from data.fundamental import FundamentalData

logger = logging.getLogger(__name__)


@dataclass
class FundamentalResult:
    """基本面分析结果。"""

    symbol: str
    pe: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    eps: Optional[float] = None
    total_market_cap: Optional[float] = None
    net_profit_margin: Optional[float] = None
    debt_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None

    # 评分
    value_score: int = 0    # -2 ~ +2  估值
    quality_score: int = 0  # -2 ~ +2  质量
    growth_score: int = 0   # -2 ~ +2  成长
    total_score: int = 0    # -6 ~ +6

    notes: list = field(default_factory=list)


class FundamentalAnalyzer:
    """基本面评估器。"""

    def __init__(self, fund_data: FundamentalData = None):
        self.fund_data = fund_data or FundamentalData()

    def analyze(self, symbol: str) -> FundamentalResult:
        """分析单只股票的基本面。"""
        result = FundamentalResult(symbol=symbol)

        try:
            summary = self.fund_data.get_financial_summary(symbol)
        except Exception as e:
            logger.warning("获取 %s 基本面数据失败: %s", symbol, e)
            return result

        # 解析关键指标
        try:
            result.pe = float(summary.get("市盈率-动态", summary.get("pe", 0)) or 0)
        except (ValueError, TypeError):
            result.pe = None
        try:
            result.pb = float(summary.get("市净率", summary.get("pb", 0)) or 0)
        except (ValueError, TypeError):
            result.pb = None
        try:
            result.roe = float(summary.get("roe", 0) or 0)
        except (ValueError, TypeError):
            result.roe = None
        try:
            result.eps = float(summary.get("eps", summary.get("基本每股收益", 0)) or 0)
        except (ValueError, TypeError):
            result.eps = None
        try:
            cap_str = str(summary.get("总市值", "0"))
            cap_str = cap_str.replace("亿", "e8").replace("万", "e4")
            result.total_market_cap = float(cap_str) if cap_str else None
        except (ValueError, TypeError):
            result.total_market_cap = None
        try:
            result.net_profit_margin = float(
                summary.get("net_profit_margin", 0) or 0
            )
        except (ValueError, TypeError):
            result.net_profit_margin = None
        try:
            result.debt_ratio = float(summary.get("debt_ratio", 0) or 0)
        except (ValueError, TypeError):
            result.debt_ratio = None

        # ── 估值评分 ──
        self._score_valuation(result)

        # ── 质量评分 ──
        self._score_quality(result)

        # ── 成长评分 ──
        self._score_growth(result)

        result.total_score = result.value_score + result.quality_score + result.growth_score
        return result

    def _score_valuation(self, r: FundamentalResult):
        """估值评分：低估值得分高。"""
        if r.pe is not None and r.pe > 0:
            if r.pe < 10:
                r.value_score += 2
            elif r.pe < 20:
                r.value_score += 1
            elif r.pe > 100:
                r.value_score -= 2
            elif r.pe > 50:
                r.value_score -= 1

        if r.pb is not None and r.pb > 0:
            if r.pb < 1:
                r.value_score += 1
            elif r.pb > 10:
                r.value_score -= 1

        # 限制范围
        r.value_score = max(-2, min(2, r.value_score))

    def _score_quality(self, r: FundamentalResult):
        """质量评分：高ROE、低负债、高利润率得分高。"""
        if r.roe is not None:
            if r.roe > 20:
                r.quality_score += 2
            elif r.roe > 15:
                r.quality_score += 1
            elif r.roe < 5 and r.roe >= 0:
                r.quality_score -= 1
            elif r.roe < 0:
                r.quality_score -= 2

        if r.net_profit_margin is not None:
            if r.net_profit_margin > 20:
                r.quality_score += 1
            elif r.net_profit_margin < 5 and r.net_profit_margin >= 0:
                r.quality_score -= 1

        if r.debt_ratio is not None:
            if r.debt_ratio > 80:
                r.quality_score -= 1
            elif r.debt_ratio < 30:
                r.quality_score += 1

        r.quality_score = max(-2, min(2, r.quality_score))

    def _score_growth(self, r: FundamentalResult):
        """成长评分：高EPS得分高。"""
        if r.eps is not None:
            if r.eps > 2:
                r.growth_score += 2
            elif r.eps > 1:
                r.growth_score += 1
            elif r.eps < 0.1 and r.eps >= 0:
                r.growth_score -= 1
            elif r.eps < 0:
                r.growth_score -= 2

        r.growth_score = max(-2, min(2, r.growth_score))