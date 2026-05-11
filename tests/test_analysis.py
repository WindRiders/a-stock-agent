"""测试基本面分析和评分模块。"""

import pytest
from analysis.fundamental import FundamentalAnalyzer, FundamentalResult
from analysis.scoring import StockScorer, StockScore


class TestFundamentalAnalyzer:
    """基本面分析测试。"""

    def setup_method(self):
        self.analyzer = FundamentalAnalyzer()

    def test_result_dataclass(self):
        """测试基本数据结构。"""
        r = FundamentalResult(symbol="000001")
        assert r.symbol == "000001"
        assert r.total_score == 0

    def test_score_valuation_low_pe(self):
        """低PE应得高分。"""
        r = FundamentalResult(symbol="000001", pe=8.0)
        self.analyzer._score_valuation(r)
        assert r.value_score > 0

    def test_score_valuation_high_pe(self):
        """高PE应得低分。"""
        r = FundamentalResult(symbol="000001", pe=120.0)
        self.analyzer._score_valuation(r)
        assert r.value_score < 0

    def test_score_quality_high_roe(self):
        """高ROE应得高分。"""
        r = FundamentalResult(symbol="000001", roe=25.0)
        self.analyzer._score_quality(r)
        assert r.quality_score > 0

    def test_score_quality_negative_roe(self):
        """负ROE应得低分。"""
        r = FundamentalResult(symbol="000001", roe=-10.0)
        self.analyzer._score_quality(r)
        assert r.quality_score < 0

    def test_score_valuation_bounds(self):
        """评分应在 -2 到 +2 之间。"""
        r = FundamentalResult(symbol="000001", pe=5.0, pb=0.5)
        self.analyzer._score_valuation(r)
        assert -2 <= r.value_score <= 2

        r2 = FundamentalResult(symbol="000002", pe=500.0, pb=50.0)
        self.analyzer._score_valuation(r2)
        assert -2 <= r2.value_score <= 2

    def test_score_quality_bounds(self):
        """质量评分应在范围内。"""
        r = FundamentalResult(symbol="000001", roe=50.0, net_profit_margin=30.0, debt_ratio=10.0)
        self.analyzer._score_quality(r)
        assert -2 <= r.quality_score <= 2

    def test_score_growth_bounds(self):
        """成长评分应在范围内。"""
        r = FundamentalResult(symbol="000001", eps=5.0)
        self.analyzer._score_growth(r)
        assert -2 <= r.growth_score <= 2


class TestStockScorer:
    """综合评分测试。"""

    def setup_method(self):
        self.scorer = StockScorer()

    def test_normalize(self):
        """归一化计算。"""
        assert self.scorer._normalize(0, -10, 10) == 0.5
        assert self.scorer._normalize(10, -10, 10) == 1.0
        assert self.scorer._normalize(-10, -10, 10) == 0.0

    def test_score_realtime_stock(self):
        """实时股票评分（需要网络）。"""
        score = self.scorer.score("000001", "平安银行")
        assert isinstance(score, StockScore)
        assert score.symbol == "000001"
        assert score.name == "平安银行"
        # 评分应在合理范围
        assert 0.0 <= score.total_score <= 1.0
        assert score.rating in ("A", "B", "C", "D")