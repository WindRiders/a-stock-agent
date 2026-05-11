"""A股基本面数据模块。

获取财务数据、估值指标、盈利能力等。
"""

import logging
from typing import Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalData:
    """基本面数据接口。"""

    def get_financial_summary(self, symbol: str) -> dict:
        """获取个股核心财务指标摘要。

        Returns dict with: pe, pb, roe, revenue_growth, profit_growth, etc.
        """
        symbol_clean = symbol.replace(".SH", "").replace(".SZ", "")
        result = {"symbol": symbol_clean}

        try:
            # 个股基本面指标
            info = ak.stock_individual_info_em(symbol=symbol_clean)
            info_dict = dict(zip(info["item"], info["value"]))
            result.update(info_dict)
        except Exception as e:
            logger.warning("获取个股信息失败: %s", e)

        try:
            # 财务指标
            fin = ak.stock_financial_analysis_indicator(symbol=symbol_clean)
            if not fin.empty:
                latest = fin.iloc[0]
                result["roe"] = latest.get("净资产收益率", None)
                result["net_profit_margin"] = latest.get("销售净利率", None)
                result["debt_ratio"] = latest.get("资产负债率", None)
                result["current_ratio"] = latest.get("流动比率", None)
                result["eps"] = latest.get("摊薄每股收益", None)
                result["bps"] = latest.get("每股净资产", None)
        except Exception as e:
            logger.warning("获取财务指标失败: %s", e)

        return result

    def get_valuation(self, symbol: str) -> pd.DataFrame:
        """获取历史估值数据（PE/PB分位）。"""
        symbol_clean = symbol.replace(".SH", "").replace(".SZ", "")
        try:
            df = ak.stock_a_lg_indicator(symbol=symbol_clean)
            df = df.rename(
                columns={
                    "trade_date": "date",
                    "pe": "pe_ttm",
                    "pe_ttm": "pe_ttm",
                    "pb": "pb",
                }
            )
            return df
        except Exception as e:
            logger.error("获取估值数据失败: %s", e)
            raise

    def get_profit_forecast(self, symbol: str) -> pd.DataFrame:
        """获取盈利预测。"""
        try:
            df = ak.stock_profit_forecast_em(symbol=symbol)
            return df
        except Exception as e:
            logger.warning("获取盈利预测失败: %s", e)
            return pd.DataFrame()

    def get_institutional_holdings(self, symbol: str) -> pd.DataFrame:
        """获取机构持仓。"""
        try:
            df = ak.stock_institute_hold_em(
                symbol=symbol,
                date="2024-12-31",
            )
            return df
        except Exception as e:
            logger.warning("获取机构持仓失败: %s", e)
            return pd.DataFrame()

    def get_shareholder_changes(self, symbol: str) -> pd.DataFrame:
        """获取股东人数变化。"""
        try:
            df = ak.stock_zh_a_gdhs(symbol=symbol)
            return df
        except Exception as e:
            logger.warning("获取股东人数失败: %s", e)
            return pd.DataFrame()