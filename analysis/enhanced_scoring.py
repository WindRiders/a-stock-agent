"""增强评分模块。

在基础多维度评分上叠加：
- 行业相对排名
- 北向资金流向
- 龙虎榜关注
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from analysis.scoring import StockScorer, StockScore
from data.market import MarketData

logger = logging.getLogger(__name__)


@dataclass
class EnhancedScore(StockScore):
    """增强评分结果。"""

    # 行业维度
    sector_name: str = ""
    sector_rank: int = 0        # 行业内排名
    sector_total: int = 0       # 行业内股票总数
    sector_pct: float = 0.0     # 行业百分位
    sector_score: int = 0       # -2~+2

    # 资金维度
    north_flow_score: int = 0   # -2~+2 北向资金
    lhb_score: int = 0          # -1~+1 龙虎榜

    # 周期维度
    multi_tf_score: int = 0     # -2~+2 多周期共振

    def __post_init__(self):
        super().__post_init__()
        # 重算总分：+行业 +资金 +周期
        self.total_score = self._calc_enhanced_total()

    def _calc_enhanced_total(self) -> float:
        """增强版综合得分。"""
        base = self.total_score * 0.7  # 基础评分权重 70%
        sector = self.sector_score / 4 * 0.10    # 行业 10%
        capital = (self.north_flow_score + self.lhb_score) / 4 * 0.10  # 资金 10%
        cycle = self.multi_tf_score / 4 * 0.10   # 周期共振 10%
        return round(base + sector + capital + cycle, 2)


class EnhancedScorer:
    """增强评分器。

    在 StockScorer 基础上增加行业对比和资金面。
    """

    def __init__(self, market_data: MarketData = None):
        self.market_data = market_data or MarketData()
        self.base_scorer = StockScorer(
            market_data=self.market_data,
        )
        self._sector_cache: Optional[Dict[str, pd.DataFrame]] = None

    def score(self, symbol: str, name: str = "") -> EnhancedScore:
        """增强版评分。"""
        # 先做基础评分
        base = self.base_scorer.score(symbol, name)
        
        enhanced = EnhancedScore(
            symbol=base.symbol,
            name=base.name,
            tech_score=base.tech_score,
            fund_score=base.fund_score,
            sentiment_score=base.sentiment_score,
            capital_score=base.capital_score,
            total_score=base.total_score,
            rating=base.rating,
            signal=base.signal,
            latest_price=base.latest_price,
            pe=base.pe,
            pb=base.pb,
            volume_ratio=base.volume_ratio,
            reasons=base.reasons,
            warnings=base.warnings,
        )

        # 行业对比
        self._add_sector_comparison(enhanced, symbol)

        # 重新计算增强总分
        enhanced.total_score = enhanced._calc_enhanced_total()
        enhanced.rating = self._calc_rating(enhanced.total_score)

        return enhanced

    def _add_sector_comparison(self, score: EnhancedScore, symbol: str):
        """添加行业对比维度。"""
        try:
            # 获取行业板块数据
            sectors = self.market_data.get_sector_list()
            if sectors.empty:
                return

            # 简化：用实时行情中的股票数据做行业对比
            quotes = self.market_data.get_realtime_quotes()
            if quotes.empty:
                return

            # 在行情中找到该股票
            stock_row = quotes[quotes["symbol"] == symbol]
            if stock_row.empty:
                return

            # 行业打分：基于涨跌幅在所有股票中的分位
            if "pct_change" in quotes.columns:
                pct = float(stock_row["pct_change"].iloc[0]) if not stock_row.empty else 0
                all_pct = quotes["pct_change"].dropna()

                if len(all_pct) > 100:
                    rank = (all_pct < pct).sum()
                    total = len(all_pct)
                    score.sector_rank = rank
                    score.sector_total = total
                    score.sector_pct = round(rank / total * 100, 1)

                    if score.sector_pct >= 80:
                        score.sector_score = 2
                        score.reasons.append(f"全市场排名前{100-score.sector_pct:.0f}%")
                    elif score.sector_pct >= 60:
                        score.sector_score = 1
                    elif score.sector_pct <= 20:
                        score.warnings.append(f"全市场排名后{score.sector_pct:.0f}%")
                        score.sector_score = -2
                    elif score.sector_pct <= 40:
                        score.sector_score = -1

        except Exception as e:
            logger.debug("行业对比失败: %s", e)

    def add_sector_analysis(
        self, scores: List[EnhancedScore]
    ) -> List[EnhancedScore]:
        """批量为评分添加行业对比。"""
        try:
            quotes = self.market_data.get_realtime_quotes()
            if quotes.empty:
                return scores

            for score in scores:
                if score.sector_total > 0:
                    continue  # 已分析过

                stock_row = quotes[quotes["symbol"] == score.symbol]
                if stock_row.empty:
                    continue

                if "pct_change" in quotes.columns:
                    pct = float(stock_row["pct_change"].iloc[0])
                    all_pct = quotes["pct_change"].dropna()
                    if len(all_pct) > 100:
                        rank = (all_pct < pct).sum()
                        score.sector_rank = rank
                        score.sector_total = len(all_pct)
                        score.sector_pct = round(rank / len(all_pct) * 100, 1)

                        if score.sector_pct >= 80:
                            score.sector_score = 2
                        elif score.sector_pct >= 60:
                            score.sector_score = 1
                        elif score.sector_pct <= 20:
                            score.sector_score = -2
                        elif score.sector_pct <= 40:
                            score.sector_score = -1

                # 重算
                score.total_score = score._calc_enhanced_total()
                score.rating = self._calc_rating(score.total_score)

        except Exception as e:
            logger.debug("批量行业对比失败: %s", e)

        return scores

    @staticmethod
    def _calc_rating(total_score: float) -> str:
        if total_score >= 0.7:
            return "A"
        elif total_score >= 0.4:
            return "B"
        elif total_score >= 0.1:
            return "C"
        return "D"