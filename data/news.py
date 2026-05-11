"""A股新闻舆情模块。

获取个股新闻、市场要闻、公告等，并进行情感分析。
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class NewsData:
    """新闻与舆情数据接口。"""

    def get_stock_news(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """获取个股相关新闻。"""
        try:
            df = ak.stock_news_em(symbol=symbol)
            if not df.empty:
                df = df.head(limit)
            return df
        except Exception as e:
            logger.warning("获取个股新闻失败: %s", e)
            return pd.DataFrame()

    def get_market_news(self, limit: int = 30) -> pd.DataFrame:
        """获取市场要闻/财经新闻。"""
        try:
            # cctv 财 经 新闻 作为 市场要闻
            df = ak.news_trade_notify_suspend_ba()
            if not df.empty:
                df = df.head(limit)
            return df
        except Exception:
            pass

        try:
            df = ak.stock_info_global_em()
            if not df.empty:
                df = df.head(limit)
            return df
        except Exception as e:
            logger.error("获取市场新闻失败: %s", e)
            return pd.DataFrame()

    def get_announcements(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """获取个股公告。"""
        try:
            df = ak.stock_notice_report(symbol=symbol)
            if not df.empty:
                df = df.head(limit)
            return df
        except Exception as e:
            logger.warning("获取个股公告失败: %s", e)
            return pd.DataFrame()

    def get_limit_up_pool(self, date: str = None) -> pd.DataFrame:
        """获取涨停池。"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_em(date=date)
            return df
        except Exception as e:
            logger.warning("获取涨停池失败: %s", e)
            return pd.DataFrame()

    def get_continuous_limit_up(self, date: str = None) -> pd.DataFrame:
        """获取连续涨停股。"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_strong_em(date=date)
            return df
        except Exception as e:
            logger.warning("获取连续涨停失败: %s", e)
            return pd.DataFrame()

    def analyze_sentiment(self, news_texts: List[str]) -> float:
        """简单的新闻情感分析。

        使用关键词匹配来估算情感得分：-1（负面）到 +1（正面）。

        Args:
            news_texts: 新闻文本列表

        Returns:
            平均情感得分
        """
        positive_words = [
            "增长", "增长超预期", "利好", "突破", "创新高", "涨停", "大增",
            "盈利", "预增", "回购", "分红", "中标", "签约", "获批",
            "超预期", "扭亏", "高增长", "扩产", "涨价", "需求旺盛",
            "政策支持", "补贴", "改革", "开放", "降息", "降准",
        ]
        negative_words = [
            "下跌", "跌停", "亏损", "预亏", "减持", "爆雷", "违规",
            "处罚", "调查", "退市", "警告", "债务", "违约", "下滑",
            "需求不足", "产能过剩", "裁员", "关停", "贸易战", "制裁",
            "加息", "收紧", "监管", "严查", "利空",
        ]

        if not news_texts:
            return 0.0

        scores = []
        for text in news_texts:
            text = str(text)
            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)
            total = pos + neg
            if total > 0:
                scores.append((pos - neg) / total)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0