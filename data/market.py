"""A股行情数据获取模块。

支持实时行情、历史K线、行业板块、龙虎榜等数据。
数据源：akshare（主力）+ baostock（备用）。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import baostock as bs
import pandas as pd

logger = logging.getLogger(__name__)


class MarketData:
    """A股市场行情数据接口。

    封装 akshare 和 baostock，提供统一的市场数据查询接口。
    """

    def __init__(self):
        self._bs_logged_in = False

    # ── 股票列表 ──────────────────────────────────────────────

    def get_stock_list(self) -> pd.DataFrame:
        """获取沪深A股列表。

        Returns:
            DataFrame with columns: code, name, industry, area, market
        """
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "symbol", "name": "name"})
            df["market"] = df["symbol"].apply(
                lambda x: "SH" if x.startswith("6") else "SZ"
            )
            logger.info("获取 A 股列表成功，共 %d 只股票", len(df))
            return df
        except Exception as e:
            logger.warning("akshare 获取股票列表失败: %s，尝试 baostock", e)
            return self._get_stock_list_bs()

    def _get_stock_list_bs(self) -> pd.DataFrame:
        self._bs_login()
        rs = bs.query_stock_basic(code_name="")
        bs.logout()
        records = []
        while (rs.error_code == "0") & rs.next():
            row = rs.get_row_data()
            # baostock 返回 6 列: code, code_name, ipoDate, outDate, type, status
            records.append(row[:4])  # 只取前4列
        df = pd.DataFrame(records, columns=["code", "name", "ipoDate", "type"])
        df = df[df["type"] == "1"]  # 只取股票
        return df.rename(columns={"code": "symbol"})

    # ── 实时行情 ──────────────────────────────────────────────

    def get_realtime_quotes(self) -> pd.DataFrame:
        """获取全市场实时行情。

        Returns:
            DataFrame with price, change%, volume, turnover, etc.
        """
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.rename(
                columns={
                    "代码": "symbol",
                    "名称": "name",
                    "最新价": "price",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "振幅": "amplitude",
                    "最高": "high",
                    "最低": "low",
                    "今开": "open",
                    "昨收": "pre_close",
                    "量比": "volume_ratio",
                    "换手率": "turnover_rate",
                    "市盈率-动态": "pe_dynamic",
                    "市净率": "pb",
                    "总市值": "total_market_cap",
                    "流通市值": "circulating_market_cap",
                }
            )
            return df
        except Exception as e:
            logger.error("获取实时行情失败: %s", e)
            raise

    # ── 历史K线 ───────────────────────────────────────────────

    def get_daily_kline(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "daily",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取个股历史日K线数据。

        Args:
            symbol: 股票代码，如 '000001'（平安银行）
            start_date: 起始日期 'YYYYMMDD'
            end_date: 截止日期 'YYYYMMDD'
            period: 'daily' / 'weekly' / 'monthly'
            adjust: 'qfq'(前复权) / 'hfq'(后复权) / ''(不复权)

        Returns:
            DataFrame with OHLCV and turnover
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover_rate",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date")
        except Exception as e:
            logger.warning("akshare K线获取失败: %s，尝试 baostock", e)
            return self._get_kline_bs(symbol, start_date, end_date, period, adjust)

    def _get_kline_bs(
        self, symbol: str, start_date: str, end_date: str, period: str, adjust: str
    ) -> pd.DataFrame:
        self._bs_login()
        freq_map = {"daily": "d", "weekly": "w", "monthly": "m"}
        adjust_map = {"qfq": "2", "hfq": "1", "": "3"}
        code = f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
        # baostock 需要 YYYY-MM-DD 格式
        start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_fmt,
            end_date=end_fmt,
            frequency=freq_map.get(period, "d"),
            adjustflag=adjust_map.get(adjust, "2"),
        )
        if rs is None:
            self._bs_logged_in = False
            bs.login()
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_fmt,
                end_date=end_fmt,
                frequency=freq_map.get(period, "d"),
                adjustflag=adjust_map.get(adjust, "2"),
            )
        bs.logout()
        if rs is None or rs.error_code != "0":
            raise RuntimeError(f"baostock 查询失败: {rs.error_msg if rs else '无响应'}")
        records = []
        while (rs.error_code == "0") & rs.next():
            records.append(rs.get_row_data())
        df = pd.DataFrame(
            records,
            columns=[
                "date", "open", "high", "low", "close",
                "volume", "turnover", "turnover_rate", "pct_change",
            ],
        )
        for col in ["open", "high", "low", "close", "volume", "turnover", "pct_change"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    # ── 指数数据 ──────────────────────────────────────────────

    def get_index_kline(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """获取指数日K线。

        Args:
            index_code: 指数代码如 '000300'(沪深300), '000001'(上证指数)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                                     "high": "high", "low": "low", "volume": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            return df[mask].sort_values("date")
        except Exception as e:
            logger.error("获取指数K线失败: %s", e)
            raise

    # ── 板块/行业 ─────────────────────────────────────────────

    def get_sector_list(self) -> pd.DataFrame:
        """获取行业板块列表及涨跌幅。"""
        try:
            df = ak.stock_board_industry_name_em()
            return df
        except Exception as e:
            logger.error("获取板块列表失败: %s", e)
            raise

    def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """获取某行业板块的成分股。"""
        try:
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
            return df
        except Exception as e:
            logger.error("获取板块成分股失败: %s", e)
            raise

    # ── 龙虎榜 ────────────────────────────────────────────────

    def get_lhb_top_list(self, trade_date: str = None) -> pd.DataFrame:
        """获取龙虎榜数据。"""
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_lhb_detail_em(date=trade_date)
            return df
        except Exception as e:
            logger.error("获取龙虎榜失败: %s", e)
            raise

    # ── 北向资金 ──────────────────────────────────────────────

    def get_north_flow(self) -> pd.DataFrame:
        """获取北向资金（沪股通+深股通）流向。"""
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            return df
        except AttributeError:
            try:
                df = ak.stock_hsgt_hist_em(symbol="沪股通")
                return df
            except Exception:
                pass
        except Exception as e:
            logger.error("获取北向资金失败: %s", e)
            raise

    # ── 内部工具 ──────────────────────────────────────────────

    def _bs_login(self):
        if not self._bs_logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                logger.warning("baostock 登录失败: %s", lg.error_msg)
            self._bs_logged_in = True

    def normalize_symbol(self, symbol: str) -> str:
        """标准化股票代码为6位数字。"""
        return symbol.replace(".SH", "").replace(".SZ", "").replace("sh", "").replace("sz", "").zfill(6)