"""测试数据持久化模块。"""

import os
import tempfile
import pytest

from data.store import DataStore
from analysis.scoring import StockScore


def make_score(symbol="000001", **kwargs):
    defaults = {
        "symbol": symbol,
        "name": "测试股",
        "tech_score": 5,
        "fund_score": 3,
        "sentiment_score": 1,
        "capital_score": 1,
        "total_score": 0.65,
        "rating": "B",
        "signal": "BUY",
        "latest_price": 10.5,
        "pe": 15.0,
        "pb": 1.5,
        "volume_ratio": 1.2,
        "reasons": ["均线多头排列", "MACD金叉"],
        "warnings": ["成交量萎缩"],
    }
    defaults.update(kwargs)
    return StockScore(**defaults)


class TestDataStore:
    """DataStore 测试套件。"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = DataStore(self.db_path)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_retrieve_scan(self):
        """保存并读取扫描记录。"""
        scores = [
            make_score("000001", total_score=0.8, signal="STRONG_BUY"),
            make_score("000002", total_score=0.6, signal="BUY"),
            make_score("000003", total_score=0.3, signal="HOLD"),
        ]
        scan_id = self.store.save_scan(scores, strategy="trend")

        # 读取历史
        history = self.store.get_scan_history(limit=5)
        assert len(history) == 1
        assert history[0]["strategy"] == "trend"
        assert history[0]["total_stocks"] == 3

        # 读取明细
        detail = self.store.get_scan_detail(scan_id)
        assert len(detail) == 3
        assert detail[0]["symbol"] == "000001"  # 按评分降序
        assert detail[0]["total_score"] == 0.8

    def test_multiple_scans(self):
        """多次扫描。"""
        for i in range(3):
            scores = [make_score(f"{i:06d}", total_score=0.5 + i * 0.1)]
            self.store.save_scan(scores, strategy="trend")

        history = self.store.get_scan_history(limit=10)
        assert len(history) == 3

    def test_stock_history(self):
        """单只股票历史记录。"""
        # 模拟多次扫描中同一只股票的不同评分
        for score_val in [0.8, 0.6, 0.4]:
            scores = [make_score("000001", total_score=score_val)]
            self.store.save_scan(scores, strategy="trend")

        records = self.store.get_stock_history("000001", limit=10)
        assert len(records) == 3
        # 按时间降序，最新的在前
        assert records[0]["total_score"] == 0.4
        assert records[2]["total_score"] == 0.8

    def test_save_signals(self):
        """保存信号。"""
        from strategy.base import TradeSignal, Signal

        signals = [
            TradeSignal(symbol="000001", name="测试A", signal=Signal.STRONG_BUY, score=0.9, confidence=0.85, reason="趋势强劲"),
            TradeSignal(symbol="000002", name="测试B", signal=Signal.BUY, score=0.6, confidence=0.7, reason="估值合理"),
        ]
        count = self.store.save_signals(signals, strategy="trend")
        assert count == 2

        recent = self.store.get_recent_signals(limit=10)
        assert len(recent) == 2

        buys = self.store.get_recent_signals(limit=10, signal_filter="STRONG_BUY")
        assert len(buys) == 1

    def test_portfolio_crud(self):
        """持仓增删改。"""
        # 开仓
        pos_id = self.store.add_position("000001", 1000, 10.0, name="测试", stop_loss=9.2, take_profit=12.5)
        assert pos_id > 0

        # 读持仓
        positions = self.store.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "000001"
        assert positions[0]["shares"] == 1000

        # 更新价格
        self.store.update_positions_prices({"000001": 11.0})
        positions = self.store.get_positions()
        assert positions[0]["current_price"] == 11.0
        assert positions[0]["profit"] == 1000.0  # (11-10)*1000

        # 加仓
        self.store.add_position("000001", 500, 10.5)
        positions = self.store.get_positions()
        assert positions[0]["shares"] == 1500
        assert positions[0]["avg_cost"] == pytest.approx(10.1667, 0.01)

        # 平仓
        assert self.store.close_position("000001", 11.0)
        active = self.store.get_positions("active")
        assert len(active) == 0
        closed = self.store.get_positions("closed")
        assert len(closed) == 1

    def test_trade_records(self):
        """交易记录。"""
        self.store.record_trade("000001", "BUY", 1000, 10.0, "测试", commission=2.5, reason="测试买入")
        self.store.record_trade("000001", "SELL", 1000, 11.0, "测试", commission=2.75, stamp_tax=11.0, reason="止盈")

        records = self.store.get_trades(10)
        assert len(records) == 2
        assert records[0]["action"] == "SELL"  # 最新在前
        assert records[1]["action"] == "BUY"

    def test_portfolio_summary(self):
        """持仓汇总。"""
        self.store.add_position("000001", 1000, 10.0)
        self.store.add_position("000002", 500, 20.0)
        self.store.update_positions_prices({"000001": 11.0, "000002": 22.0})

        summary = self.store.get_portfolio_summary()
        assert summary["active_count"] == 2
        assert summary["total_market_value"] == 22000.0  # 11*1000 + 22*500
        assert summary["active_profit"] == 2000.0  # 1000 + 1000

    def test_stats(self):
        """数据库统计。"""
        scores = [make_score("000001")]
        self.store.save_scan(scores, strategy="trend")

        stats = self.store.get_stats()
        assert stats["total_scans"] == 1
        assert stats["total_items"] == 1
        assert stats["last_scan"] is not None
        assert stats["db_path"] == self.db_path

    def test_empty_state(self):
        """空状态查询不应报错。"""
        assert self.store.get_scan_history() == []
        assert self.store.get_positions() == []
        assert self.store.get_trades() == []
        assert self.store.get_recent_signals() == []
        assert self.store.get_portfolio_summary()["active_count"] == 0