"""资金面深度分析模块。

利用已有的北向资金、龙虎榜接口，
为评分系统提供真实的资金面数据。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from data.market import MarketData

logger = logging.getLogger(__name__)


class CapitalFlowAnalyzer:
    """资金面分析器。

    分析维度：
    1. 北向资金（沪股通+深股通）净流入趋势
    2. 龙虎榜上榜情况
    """

    def __init__(self, market_data: MarketData = None):
        self.market_data = market_data or MarketData()
        self._north_cache: Optional[Dict] = None
        self._lhb_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    # ── 北向资金 ────────────────────────────────────────────

    def get_north_flow_score(self) -> int:
        """获取北向资金情绪评分。

        Returns:
            -2: 持续流出（看空）
            -1: 小幅流出  
             0: 无数据/中性
            +1: 小幅流入
            +2: 持续流入（看多）
        """
        try:
            df = self.market_data.get_north_flow()
            if df.empty:
                return 0

            # 找近期数据列
            flow_col = None
            for col in ["net_flow", "net_amount", "value", "北向资金", "资金"]:
                if col in df.columns:
                    flow_col = col
                    break

            if flow_col is None:
                # 尝试数值列
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    flow_col = numeric_cols[0]
                else:
                    return 0

            # 近期5天趋势
            recent = df.tail(5)
            values = pd.to_numeric(recent[flow_col], errors="coerce").dropna()

            if len(values) < 3:
                return 0

            # 5日净流入总和
            total_flow = values.sum()
            # 5日趋势（线性回归斜率）
            import numpy as np
            x = np.arange(len(values))
            y = values.values
            if len(values) >= 3:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0

            # 评分逻辑
            if total_flow > 0 and slope > 0:
                return 2  # 持续净流入
            elif total_flow > 0:
                return 1  # 有流入但趋势不确定
            elif total_flow < 0 and slope < 0:
                return -2  # 持续净流出
            elif total_flow < 0:
                return -1
            else:
                return 0

        except Exception as e:
            logger.debug("北向资金分析失败: %s", e)
            return 0

    # ── 龙虎榜 ──────────────────────────────────────────────

    def get_lhb_score(self, symbol: str) -> int:
        """检查股票是否在龙虎榜上。

        Returns:
            -1: 龙虎榜卖出
             0: 不在龙虎榜
            +1: 龙虎榜买入
        """
        try:
            today = datetime.now().strftime("%Y%m%d")
            # 尝试今天，失败则昨天
            for attempt in range(3):
                date = (datetime.now() - timedelta(days=attempt)).strftime("%Y%m%d")
                try:
                    df = self.market_data.get_lhb_top_list(trade_date=date)
                    if not df.empty and "代码" in df.columns:
                        stock_rows = df[df["代码"] == symbol]
                        if not stock_rows.empty:
                            # 判断买卖方向
                            # 龙虎榜数据通常有"买方金额"和"卖方金额"
                            buy_cols = [c for c in df.columns if "买" in c and "金额" in c]
                            sell_cols = [c for c in df.columns if "卖" in c and "金额" in c]

                            if buy_cols or sell_cols:
                                buy_amt = 0
                                sell_amt = 0
                                row = stock_rows.iloc[0]
                                for bc in buy_cols:
                                    try:
                                        buy_amt += float(row[bc])
                                    except Exception:
                                        pass
                                for sc in sell_cols:
                                    try:
                                        sell_amt += float(row[sc])
                                    except Exception:
                                        pass

                                if buy_amt > sell_amt * 1.5:
                                    return 1
                                elif sell_amt > buy_amt * 1.5:
                                    return -1
                            return 1  # 上榜默认偏多关注
                        return 0
                except Exception:
                    continue
            return 0

        except Exception as e:
            logger.debug("龙虎榜分析失败: %s", e)
            return 0

    # ── 综合资金评分 ────────────────────────────────────────

    def analyze(self, symbol: str = None) -> Dict:
        """综合分析资金面。

        Returns:
            {
                "north_flow_score": int,  # -2~+2
                "lhb_score": int,         # -1~+1
                "total_score": int,       # -3~+3
                "north_trend": str,
                "lhb_status": str,
            }
        """
        north = self.get_north_flow_score()
        lhb = self.get_lhb_score(symbol) if symbol else 0

        trend_map = {
            2: "持续流入", 1: "小幅流入", 0: "中性",
            -1: "小幅流出", -2: "持续流出",
        }
        lhb_map = {1: "买入上榜", 0: "未上榜", -1: "卖出上榜"}

        return {
            "north_flow_score": north,
            "lhb_score": lhb,
            "total_score": north + lhb,
            "north_trend": trend_map.get(north, "未知"),
            "lhb_status": lhb_map.get(lhb, "未知"),
        }