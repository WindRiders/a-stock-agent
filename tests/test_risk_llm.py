"""测试风险管理和LLM分析模块。"""

import pytest
import numpy as np
import pandas as pd

from risk import RiskManager, RiskLimits, PositionSize, PortfolioRisk
from agent.llm import LLMAnalyzer
from analysis.scoring import StockScore


def make_score(**kwargs) -> StockScore:
    defaults = {
        "symbol": "000001",
        "name": "测试股",
        "tech_score": 0,
        "fund_score": 0,
        "sentiment_score": 0,
        "capital_score": 0,
        "total_score": 0.5,
        "latest_price": 10.0,
        "pe": 15.0,
        "pb": 1.5,
        "volume_ratio": 1.0,
        "signal": "HOLD",
        "rating": "C",
    }
    defaults.update(kwargs)
    return StockScore(**defaults)


class TestRiskManager:
    """风险管理器测试。"""

    def setup_method(self):
        self.rm = RiskManager()

    def test_position_sizing(self):
        """仓位计算。"""
        scores = [
            make_score(symbol="000001", total_score=0.8, signal="STRONG_BUY", name="测试A"),
            make_score(symbol="000002", total_score=0.6, signal="BUY", name="测试B"),
            make_score(symbol="000003", total_score=0.1, signal="HOLD", name="测试C"),
        ]
        positions = self.rm.calculate_position_sizes(scores, 100000)
        # 只有前两只有买入信号
        assert len(positions) == 2
        # 仓位应在合理范围
        for p in positions:
            assert 0 < p.suggested_pct <= 0.20

    def test_position_sizing_scores_sorted(self):
        """高分股票应获得不低于低分股票的仓位。"""
        scores = [
            make_score(symbol="000001", total_score=0.9, signal="STRONG_BUY"),
            make_score(symbol="000002", total_score=0.55, signal="BUY"),
        ]
        positions = self.rm.calculate_position_sizes(scores, 100000)
        # 高分应有 >= 低分的仓位（都可能碰到 20% 上限）
        assert positions[0].suggested_pct >= positions[1].suggested_pct

    def test_max_positions_limit(self):
        """最多持仓数限制。"""
        rm = RiskManager(RiskLimits(max_total_positions=3))
        scores = [
            make_score(symbol=f"{i:06d}", total_score=0.6 + 0.01 * i, signal="BUY")
            for i in range(10)
        ]
        positions = rm.calculate_position_sizes(scores, 100000)
        assert len(positions) <= 3

    def test_stop_loss_calculation(self):
        """止损价计算。"""
        scores = [
            make_score(symbol="000001", total_score=0.8, signal="STRONG_BUY", latest_price=10.0),
        ]
        positions = self.rm.calculate_position_sizes(scores, 100000)
        assert positions[0].stop_loss_price == pytest.approx(9.2, 0.1)  # -8%
        assert positions[0].take_profit_price == pytest.approx(12.5, 0.1)  # +25%

    def test_portfolio_risk_calculation(self):
        """组合风险指标计算。"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        risk = self.rm.calculate_portfolio_risk(returns)
        assert risk.var_95 < 0
        assert risk.cvar_95 < 0
        assert risk.volatility > 0

    def test_diversification_score(self):
        """分散化评分。"""
        np.random.seed(42)
        # 不相关资产
        uncorr = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(0.001, 0.02, 100),
            "C": np.random.normal(0.001, 0.02, 100),
        })
        score = self.rm.calc_diversification_score(uncorr)
        assert score > 0.5  # 不相关资产应该有高分散化得分


class TestLLMAnalyzer:
    """LLM分析器测试。"""

    def setup_method(self):
        self.analyzer = LLMAnalyzer()

    def test_market_summary_empty(self):
        """空数据。"""
        result = self.analyzer.generate_market_summary([])
        assert "暂无" in result

    def test_market_summary_with_data(self):
        """正常数据。"""
        scores = [
            make_score(symbol="000001", total_score=0.8, signal="STRONG_BUY"),
            make_score(symbol="000002", total_score=0.6, signal="BUY"),
            make_score(symbol="000003", total_score=0.3, signal="HOLD"),
            make_score(symbol="000004", total_score=0.1, signal="SELL"),
        ]
        result = self.analyzer.generate_market_summary(scores)
        assert "买入信号" in result
        assert "卖出信号" in result

    def test_stock_commentary(self):
        """个股点评生成。"""
        score = make_score(
            symbol="000001", name="平安银行", total_score=0.75,
            tech_score=6, fund_score=4, signal="BUY", rating="B",
            pe=8.0, pb=0.9, volume_ratio=1.8,
            reasons=["均线多头排列"], warnings=[]
        )
        commentary = self.analyzer.generate_stock_commentary(score)
        assert "平安银行" in commentary
        assert "000001" in commentary
        assert "均线多头排列" in commentary

    def test_build_market_prompt(self):
        """测试LLM prompt构建。"""
        scores = [
            make_score(symbol="000001", total_score=0.8, signal="STRONG_BUY", name="A"),
            make_score(symbol="000002", total_score=0.6, signal="BUY", name="B"),
        ]
        prompt = self.analyzer.build_market_prompt(scores)
        assert "STRONG_BUY" in prompt or "强烈买入" in prompt
        assert len(prompt) > 100

    def test_full_report(self):
        """完整报告生成。"""
        scores = [
            make_score(symbol="000001", total_score=0.8, signal="STRONG_BUY", name="A"),
            make_score(symbol="000002", total_score=0.6, signal="BUY", name="B"),
            make_score(symbol="000003", total_score=0.2, signal="SELL", name="C"),
        ]
        report = self.analyzer.generate_full_report(scores)
        assert "A股智能分析报告" in report
        assert len(report) > 500