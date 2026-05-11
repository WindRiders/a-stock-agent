"""回测指标计算与报告。"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass
class MetricsReport:
    """完整的回测绩效报告。"""

    result: BacktestResult

    def to_dict(self) -> dict:
        return {
            "初始资金": f"¥{self.result.initial_capital:,.0f}",
            "最终权益": f"¥{self.result.final_equity:,.2f}",
            "总收益率": f"{self.result.total_return:.2f}%",
            "年化收益率": f"{self.result.annual_return:.2f}%",
            "最大回撤": f"{self.result.max_drawdown:.2f}%",
            "夏普比率": f"{self.result.sharpe_ratio:.2f}",
            "胜率": f"{self.result.win_rate:.1f}%",
            "交易次数": f"{self.result.total_trades}",
        }

    def summary(self) -> str:
        """格式化绩效摘要。"""
        d = self.to_dict()
        lines = [
            "══════════════════════════════",
            "      回 测 绩 效 报 告",
            "══════════════════════════════",
        ]
        for key, val in d.items():
            lines.append(f"  {key:　<10s} : {val}")
        lines.append("══════════════════════════════")
        return "\n".join(lines)


class BacktestMetrics:
    """回测绩效指标计算工具。"""

    @staticmethod
    def calc_sharpe(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率。"""
        if len(returns) < 2:
            return 0.0
        r = np.array(returns)
        excess = np.mean(r) - risk_free_rate / 252
        std = np.std(r, ddof=1)
        return (excess / std * math.sqrt(252)) if std > 0 else 0.0

    @staticmethod
    def calc_max_drawdown(equity_curve: pd.Series) -> float:
        """计算最大回撤。"""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def calc_calmar(annual_return: float, max_drawdown: float) -> float:
        """Calmar 比率 = 年化收益 / 最大回撤。"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)

    @staticmethod
    def calc_sortino(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Sortino 比率（只看下行波动）。"""
        if len(returns) < 2:
            return 0.0
        r = np.array(returns)
        downside = r[r < 0]
        if len(downside) == 0:
            return 10.0
        excess = np.mean(r) - risk_free_rate / 252
        std_down = np.std(downside, ddof=1)
        return (excess / std_down * math.sqrt(252)) if std_down > 0 else 0.0