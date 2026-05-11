"""测试数据层：MarketData 行情数据获取。"""

import pytest
import pandas as pd
from data.market import MarketData

pytestmark = pytest.mark.network


class TestMarketData:
    """MarketData 测试套件。"""

    def setup_method(self):
        self.market = MarketData()

    def test_normalize_symbol(self):
        """测试代码标准化。"""
        assert self.market.normalize_symbol("000001") == "000001"
        assert self.market.normalize_symbol("sh000001") == "000001"
        assert self.market.normalize_symbol("000001.SZ") == "000001"
        assert self.market.normalize_symbol("600000.SH") == "600000"
        assert self.market.normalize_symbol("1") == "000001"

    def test_get_stock_list(self):
        """测试获取股票列表。"""
        df = self.market.get_stock_list()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "symbol" in df.columns
        assert "name" in df.columns
        # 至少应该有 4000+ 只股票
        assert len(df) > 4000

    def test_get_realtime_quotes(self):
        """测试获取实时行情。"""
        df = self.market.get_realtime_quotes()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        required_cols = ["symbol", "name", "price", "pct_change"]
        for col in required_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_get_daily_kline(self):
        """测试获取日K线。"""
        df = self.market.get_daily_kline("000001")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"缺少列: {col}"
        # 至少有200个交易日
        assert len(df) >= 200

    def test_get_daily_kline_with_baostock_fallback(self):
        """测试 baostock 备用数据源。"""
        # 用一只可能引起 akshare 失败的股票测试 fallback
        df = self.market._get_kline_bs(
            "000001", "20240101", "20240630", "daily", "qfq"
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "date" in df.columns

    def test_get_index_kline(self):
        """测试获取指数K线。"""
        df = self.market.get_index_kline("000300")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_get_sector_list(self):
        """测试获取板块列表。"""
        df = self.market.get_sector_list()
        assert isinstance(df, pd.DataFrame)
        # 至少有几十个行业板块
        assert len(df) > 20

    def test_get_north_flow(self):
        """测试获取北向资金。"""
        df = self.market.get_north_flow()
        assert isinstance(df, pd.DataFrame)


class TestMarketDataEdgeCases:
    """边界情况测试。"""

    def setup_method(self):
        self.market = MarketData()

    def test_invalid_symbol(self):
        """无效股票代码应该优雅处理。"""
        with pytest.raises(Exception):
            # akshare 可能会抛异常
            self.market.get_daily_kline("999999")

    def test_kline_date_range(self):
        """自定义日期范围。"""
        df = self.market.get_daily_kline(
            "000001", start_date="20240101", end_date="20240131"
        )
        assert len(df) > 0
        assert int(str(df["date"].iloc[0])[:7].replace("-", "")) >= 202401

    def test_kline_adjust(self):
        """测试复权模式。"""
        df_qfq = self.market.get_daily_kline("000001", adjust="qfq")
        df_no = self.market.get_daily_kline("000001", adjust="")
        # 复权价格应该不同
        if not df_qfq.empty and not df_no.empty:
            # 找最早可比较的日期
            assert df_qfq["close"].iloc[0] != df_no["close"].iloc[0]