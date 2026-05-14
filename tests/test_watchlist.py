"""测试自选股监视列表。"""

import json
import os
import tempfile
import pytest

from watchlist import Watchlist, WatchItem, WatchAlert, WatchScanResult


class TestWatchlist:
    """自选股列表 CRUD 测试。"""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.wl = Watchlist(path=self.tmp.name)

    def teardown_method(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_empty_on_init(self):
        """初始为空。"""
        assert len(self.wl.list()) == 0
        assert self.wl.stats()["total"] == 0

    def test_add_item(self):
        """添加自选股。"""
        self.wl.add("000001", name="平安银行")
        assert len(self.wl.list()) == 1
        assert self.wl.get("000001").name == "平安银行"

    def test_add_update_existing(self):
        """重复添加更新已有。"""
        self.wl.add("000001", name="旧名")
        self.wl.add("000001", name="平安银行", tags=["银行"])
        assert self.wl.get("000001").name == "平安银行"
        assert "银行" in self.wl.get("000001").tags

    def test_remove(self):
        """移除自选股。"""
        self.wl.add("000001")
        self.wl.add("000002")
        self.wl.remove("000001")
        assert len(self.wl.list()) == 1
        assert self.wl.get("000001") is None

    def test_clear(self):
        """清空列表。"""
        self.wl.add("000001")
        self.wl.add("000002")
        self.wl.clear()
        assert len(self.wl.list()) == 0

    def test_persistence_roundtrip(self):
        """保存再加载，数据一致。"""
        self.wl.add("000001", name="平安", tags=["银行"], alerts={"pct_change_gt": 5})
        self.wl.add("000002", name="万科", tags=["地产"])

        # 新建实例加载
        wl2 = Watchlist(path=self.tmp.name)
        assert len(wl2.list()) == 2
        item = wl2.get("000001")
        assert item.name == "平安"
        assert item.tags == ["银行"]
        assert item.alerts == {"pct_change_gt": 5}

    def test_tags(self):
        """标签管理。"""
        self.wl.add("000001")
        self.wl.tag("000001", "银行", "龙头")
        assert self.wl.get("000001").tags == ["银行", "龙头"]

        self.wl.untag("000001", "银行")
        assert self.wl.get("000001").tags == ["龙头"]

    def test_by_tag(self):
        """按标签筛选。"""
        self.wl.add("000001", tags=["科技"])
        self.wl.add("000002", tags=["银行"])
        self.wl.add("000858", tags=["科技", "消费"])

        tech = self.wl.by_tag("科技")
        assert len(tech) == 2
        assert tech[0].symbol in ("000001", "000858")

    def test_alerts_set_and_clear(self):
        """告警设置和清除。"""
        self.wl.add("000001")
        self.wl.set_alert("000001", pct_change_gt=5, price_lt=10)

        assert self.wl.get("000001").alerts["pct_change_gt"] == 5
        assert self.wl.get("000001").alerts["price_lt"] == 10

        self.wl.clear_alerts("000001")
        assert self.wl.get("000001").alerts == {}

    def test_stats(self):
        """统计信息。"""
        self.wl.add("000001", alerts={"pct_change_gt": 5})
        self.wl.add("000002")
        self.wl.add("000858", alerts={"price_lt": 10})

        stats = self.wl.stats()
        assert stats["total"] == 3
        assert stats["with_alerts"] == 2

    def test_list_returns_items(self):
        """list() 返回 WatchItem 列表。"""
        self.wl.add("000001", name="平安")
        items = self.wl.list()
        assert len(items) == 1
        assert isinstance(items[0], WatchItem)

    def test_non_existent_get(self):
        """不存在返回 None。"""
        assert self.wl.get("999999") is None

    def test_corrupted_file(self):
        """损坏的文件不崩溃。"""
        with open(self.tmp.name, "w") as f:
            f.write("not valid json")

        wl = Watchlist(path=self.tmp.name)
        assert len(wl.list()) == 0


class TestWatchAlerts:
    """告警检查测试。"""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.wl = Watchlist(path=self.tmp.name)

    def teardown_method(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_no_alerts_configured(self):
        """无告警条件时返回空。"""
        self.wl.add("000001")
        scan_results = [WatchScanResult(symbol="000001", name="平安", pct_change=5.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 0

    def test_pct_change_gt_triggered(self):
        """涨幅超限触发告警。"""
        self.wl.add("000001", alerts={"pct_change_gt": 3})
        scan_results = [WatchScanResult(symbol="000001", name="平安", pct_change=5.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "pct_change_gt"
        assert alerts[0].severity == "warning"

    def test_pct_change_gt_not_triggered(self):
        """涨幅未达阈值不触发。"""
        self.wl.add("000001", alerts={"pct_change_gt": 5})
        scan_results = [WatchScanResult(symbol="000001", name="平安", pct_change=2.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 0

    def test_pct_change_lt_triggered(self):
        """跌幅超限触发告警。"""
        self.wl.add("000001", alerts={"pct_change_lt": -3})
        scan_results = [WatchScanResult(symbol="000001", name="平安", pct_change=-5.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1
        assert alerts[0].alert_type == "pct_change_lt"
        assert alerts[0].severity == "critical"

    def test_price_gt_triggered(self):
        """价格突破触发。"""
        self.wl.add("000001", alerts={"price_gt": 50})
        scan_results = [WatchScanResult(symbol="000001", name="平安", latest_price=55.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_price_lt_triggered(self):
        """价格跌破触发。"""
        self.wl.add("000001", alerts={"price_lt": 5})
        scan_results = [WatchScanResult(symbol="000001", name="平安", latest_price=4.5, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_volume_ratio_triggered(self):
        """量比告警。"""
        self.wl.add("000001", alerts={"volume_ratio_gt": 3})
        scan_results = [WatchScanResult(symbol="000001", name="平安", volume_ratio=5.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_rating_below_triggered(self):
        """评级告警。"""
        self.wl.add("000001", alerts={"rating_below": "B"})
        scan_results = [WatchScanResult(symbol="000001", name="平安", rating="C", score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_signal_triggered(self):
        """信号告警。"""
        self.wl.add("000001", alerts={"signal": "STRONG_BUY"})
        scan_results = [WatchScanResult(symbol="000001", name="平安", signal="STRONG_BUY", score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_score_triggered(self):
        """评分告警。"""
        self.wl.add("000001", alerts={"score_gt": 0.8})
        scan_results = [WatchScanResult(symbol="000001", name="平安", score=0.85)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 1

    def test_multiple_alerts(self):
        """多个告警条件同时触发。"""
        self.wl.add("000001", alerts={"pct_change_gt": 3, "volume_ratio_gt": 2})
        scan_results = [WatchScanResult(
            symbol="000001", name="平安", pct_change=5.0, volume_ratio=4.0, score=0.5,
        )]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 2

    def test_symbol_not_in_watchlist(self):
        """不在列表中的股票不触发告警。"""
        self.wl.add("000001", alerts={"pct_change_gt": 3})
        scan_results = [WatchScanResult(symbol="000002", name="", pct_change=10.0, score=0.5)]
        alerts = self.wl.check_alerts(scan_results)
        assert len(alerts) == 0