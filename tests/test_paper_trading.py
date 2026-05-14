"""测试纸面交易/模拟盘模块。"""

import os
import tempfile
import pytest

from paper_trading import PaperTrader, PaperTrade, PaperPosition, PaperSummary


class TestPaperTrader:
    """模拟盘交易测试。"""

    def setup_method(self):
        """每个测试前创建新的模拟盘。"""
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.trader = PaperTrader(initial_capital=100000, db_path=self.tmp_db.name)

    def teardown_method(self):
        """清理临时数据库。"""
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    def test_initial_state(self):
        """初始状态：全额现金，无持仓。"""
        assert self.trader.cash == 100000
        assert len(self.trader.positions) == 0
        assert len(self.trader.trades) == 0

    def test_buy_creates_position(self):
        """买入创建持仓。"""
        trade = self.trader.execute("000001", "BUY", 10.0, 1000, name="平安银行")
        assert trade is not None
        assert trade.action == "BUY"
        assert trade.shares == 1000
        assert "000001" in self.trader.positions
        assert self.trader.positions["000001"].shares == 1000
        assert self.trader.cash < 100000  # 资金被扣

    def test_buy_insufficient_cash(self):
        """资金不足时买入失败。"""
        trade = self.trader.execute("000001", "BUY", 50000.0, 1000)
        assert trade is None  # 无法买入（金额 > 资金）

    def test_buy_by_amount(self):
        """按金额买入。"""
        trade = self.trader.execute("000001", "BUY", 10.0, amount=20000, name="平安银行")
        assert trade is not None
        assert trade.shares == 2000  # 20000 / 10 = 2000 股

    def test_buy_below_one_lot(self):
        """不足1手时买入失败。"""
        trade = self.trader.execute("000001", "BUY", 10.0, 50)  # 50 股 < 100
        assert trade is None

    def test_sell_reduces_position(self):
        """卖出减少持仓。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        trade = self.trader.execute("000001", "SELL", 11.0, 500)
        assert trade is not None
        assert trade.action == "SELL"
        assert trade.shares == 500
        pos = self.trader.positions["000001"]
        assert pos.shares == 500
        assert trade.realized_pnl != 0

    def test_sell_all_clears_position(self):
        """卖出全部清除持仓。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        trade = self.trader.execute("000001", "SELL", 11.0, 1000)
        assert trade is not None
        assert "000001" not in self.trader.positions

    def test_sell_no_position(self):
        """无持仓卖出失败。"""
        trade = self.trader.execute("000001", "SELL", 10.0, 1000)
        assert trade is None

    def test_execute_signal_buy(self):
        """根据信号买入。"""
        trade = self.trader.execute_signal("000001", "STRONG_BUY", 0.85, 10.0, name="测试")
        assert trade is not None
        assert trade.action == "BUY"

    def test_execute_signal_already_held(self):
        """已有持仓时信号不重复买入。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        trade = self.trader.execute_signal("000001", "BUY", 0.85, 10.0)
        assert trade is None

    def test_execute_signal_sell(self):
        """根据信号卖出。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        trade = self.trader.execute_signal("000001", "SELL", 0.30, 11.0)
        assert trade is not None
        assert trade.action == "SELL"

    def test_execute_signal_sell_no_position(self):
        """无持仓不卖。"""
        trade = self.trader.execute_signal("000001", "SELL", 0.30, 10.0)
        assert trade is None

    def test_update_prices(self):
        """更新持仓现价。"""
        self.trader.execute("000001", "BUY", 10.0, 1000, name="平安")
        self.trader.update_prices({"000001": 11.0})
        pos = self.trader.positions["000001"]
        assert pos.current_price == 11.0
        assert pos.market_value == 11000

    def test_take_snapshot(self):
        """记录快照。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        self.trader.update_prices({"000001": 11.0})
        snap = self.trader.take_snapshot()
        assert snap.equity > 100000  # 盈利
        assert snap.position_count == 1
        assert len(self.trader.snapshots) == 1

    def test_summary_metrics(self):
        """汇总指标。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        self.trader.update_prices({"000001": 11.0})
        self.trader.take_snapshot()

        s = self.trader.summary()
        assert s.initial_capital == 100000
        assert s.active_positions == 1
        assert s.total_trades == 1

    def test_reset_clears_all(self):
        """重置清空所有状态。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        self.trader.take_snapshot()

        self.trader.reset()
        assert self.trader.cash == 100000
        assert len(self.trader.positions) == 0
        assert len(self.trader.trades) == 0
        assert len(self.trader.snapshots) == 0

    def test_positions_list(self):
        """持仓列表。"""
        self.trader.execute("000001", "BUY", 10.0, 1000, name="平安")
        self.trader.update_prices({"000001": 11.0})

        positions = self.trader.positions_list()
        assert len(positions) == 1
        p = positions[0]
        assert p["symbol"] == "000001"
        assert p["shares"] == 1000
        assert p["unrealized_pnl"] > 0

    def test_generate_report(self):
        """生成报告。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        report = self.trader.generate_report()
        assert "初始资金" in report
        assert "100" in report

    def test_commission_calculated(self):
        """佣金正确计算。"""
        trade = self.trader.execute("000001", "BUY", 10.0, 1000)
        assert trade.commission > 0
        # 万2.5 × 10000 = 2.5，最低5元
        assert trade.commission >= 5.0

    def test_stamp_tax_only_on_sell(self):
        """印花税仅卖出收取。"""
        self.trader.execute("000001", "BUY", 10.0, 1000)
        trade = self.trader.execute("000001", "SELL", 11.0, 1000)
        assert trade.stamp_tax > 0  # 千1

    def test_multiple_positions(self):
        """多只持仓。"""
        self.trader.execute("000001", "BUY", 10.0, 1000, name="A")
        self.trader.execute("000002", "BUY", 15.0, 500, name="B")
        assert len(self.trader.positions) == 2
        positions = self.trader.positions_list()
        assert len(positions) == 2

    def test_partial_sell_multiple(self):
        """多次部分卖出。"""
        self.trader.execute("000001", "BUY", 10.0, 2000)
        self.trader.execute("000001", "SELL", 11.0, 500)
        assert self.trader.positions["000001"].shares == 1500
        self.trader.execute("000001", "SELL", 11.0, 500)
        assert self.trader.positions["000001"].shares == 1000
        self.trader.execute("000001", "SELL", 11.0, 1000)
        assert "000001" not in self.trader.positions