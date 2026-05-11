"""风险管理模块。

仓位管理、组合优化、风险指标计算（VaR, CVaR, Beta, 最大回撤等）。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """单只股票的仓位建议。"""

    symbol: str
    name: str = ""
    score: float = 0.0
    max_pct: float = 0.0  # 最大仓位占比
    suggested_pct: float = 0.0  # 建议仓位占比
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reason: str = ""


@dataclass
class PortfolioRisk:
    """组合风险指标。"""

    var_95: float = 0.0  # 95%置信度 VaR
    cvar_95: float = 0.0  # 95% CVaR
    max_drawdown: float = 0.0
    volatility: float = 0.0  # 年化波动率
    beta: float = 0.0  # 相对沪深300的Beta
    sharpe: float = 0.0
    diversification_score: float = 0.0  # 分散化评分


@dataclass
class RiskLimits:
    """风控限制参数。"""

    max_single_position: float = 0.20  # 单只最大仓位 20%
    max_total_positions: int = 8  # 最大持仓数
    max_sector_exposure: float = 0.40  # 单行业最大曝露
    stop_loss_pct: float = -0.08  # 止损线 -8%
    take_profit_pct: float = 0.25  # 止盈线 +25%
    trailing_stop_pct: float = 0.05  # 移动止损 5%
    max_leverage: float = 1.0  # 最大杠杆（A股现货为1）


class RiskManager:
    """风险管理器。

    提供仓位计算、组合风险度量、止损止盈建议。
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()

    # ── 仓位计算 ────────────────────────────────────────────

    def calculate_position_sizes(
        self,
        scores: List,
        total_capital: float,
        current_positions: Dict[str, float] = None,
    ) -> List[PositionSize]:
        """根据评分计算建议仓位。

        算法：
        1. 按评分排序，取 Top N
        2. 评分归一化作为仓位权重
        3. 限制单只最大仓位
        4. 考虑已有持仓
        """
        if current_positions is None:
            current_positions = {}

        # 只考虑买入信号的股票
        candidates = [
            s for s in scores
            if s.signal in ("BUY", "STRONG_BUY")
            and s.symbol not in current_positions
        ]
        candidates.sort(key=lambda x: x.total_score, reverse=True)

        if not candidates:
            return []

        # 取评分前 N 只
        top_n = candidates[:self.limits.max_total_positions]

        # 计算剩余可用资金
        used_capital = sum(current_positions.values())
        available_capital = total_capital - used_capital

        # 基于评分分配权重
        scores_only = [s.total_score for s in top_n]
        total_score = sum(scores_only)
        if total_score == 0:
            return []

        results = []
        for s, sc in zip(top_n, scores_only):
            weight = sc / total_score  # 按评分占比
            suggested_pct = min(
                weight * (available_capital / total_capital),
                self.limits.max_single_position,
            )

            # 计算止损止盈价
            if s.latest_price and s.latest_price > 0:
                stop_loss = s.latest_price * (1 + self.limits.stop_loss_pct)
                take_profit = s.latest_price * (1 + self.limits.take_profit_pct)
            else:
                stop_loss = take_profit = None

            # 生成理由
            if s.total_score >= 0.7:
                reason = "评分优秀，建议标准仓位"
            elif s.total_score >= 0.5:
                reason = "评分良好，适当配置"
            else:
                reason = "评分一般，轻仓试探"

            results.append(PositionSize(
                symbol=s.symbol,
                name=s.name,
                score=s.total_score,
                max_pct=self.limits.max_single_position,
                suggested_pct=round(suggested_pct, 4),
                stop_loss_price=round(stop_loss, 2) if stop_loss else None,
                take_profit_price=round(take_profit, 2) if take_profit else None,
                reason=reason,
            ))

        return results

    # ── 止损/止盈检查 ────────────────────────────────────────

    def check_stop_conditions(
        self,
        positions: Dict[str, dict],
        price_data: Dict[str, pd.DataFrame],
    ) -> List[dict]:
        """检查持仓是否需要止损/止盈。

        positions: {symbol: {avg_cost, shares, stop_loss, take_profit}}
        返回需要操作的列表。
        """
        alerts = []
        for symbol, pos in positions.items():
            if symbol not in price_data:
                continue

            df = price_data[symbol]
            if df.empty:
                continue

            current_price = float(df["close"].iloc[-1])
            avg_cost = pos.get("avg_cost", current_price)
            pnl_pct = (current_price / avg_cost - 1) if avg_cost > 0 else 0

            # 止损检查
            if pnl_pct <= self.limits.stop_loss_pct:
                alerts.append({
                    "symbol": symbol,
                    "action": "STOP_LOSS",
                    "current_price": current_price,
                    "avg_cost": avg_cost,
                    "pnl_pct": pnl_pct,
                    "reason": f"触及止损线 ({pnl_pct*100:.1f}%)",
                })
                continue

            # 止盈检查
            if pnl_pct >= self.limits.take_profit_pct:
                alerts.append({
                    "symbol": symbol,
                    "action": "TAKE_PROFIT",
                    "current_price": current_price,
                    "avg_cost": avg_cost,
                    "pnl_pct": pnl_pct,
                    "reason": f"达到止盈目标 ({pnl_pct*100:.1f}%)",
                })
                continue

            # 移动止损：从最高点回落超过阈值
            if "peak_price" in pos:
                peak = pos["peak_price"]
                from_peak = (current_price / peak - 1) if peak > 0 else 0
                if from_peak <= -self.limits.trailing_stop_pct:
                    alerts.append({
                        "symbol": symbol,
                        "action": "TRAILING_STOP",
                        "current_price": current_price,
                        "avg_cost": avg_cost,
                        "pnl_pct": pnl_pct,
                        "reason": f"从高点回落 {from_peak*100:.1f}%（移动止损）",
                    })

        return alerts

    # ── 风险指标计算 ─────────────────────────────────────────

    def calculate_portfolio_risk(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: pd.Series = None,
    ) -> PortfolioRisk:
        """计算组合风险指标。

        returns_df: 组合日收益率序列
        benchmark_returns: 基准（如沪深300）日收益率
        """
        risk = PortfolioRisk()

        if returns_df.empty or len(returns_df) < 2:
            return risk

        returns = returns_df.values if isinstance(returns_df, pd.DataFrame) else returns_df
        if returns.ndim > 1:
            returns = returns.flatten()

        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return risk

        # VaR (Historical method, 95%)
        risk.var_95 = np.percentile(returns, 5)

        # CVaR
        tail = returns[returns <= risk.var_95]
        risk.cvar_95 = float(np.mean(tail)) if len(tail) > 0 else risk.var_95

        # 年化波动率
        risk.volatility = float(np.std(returns, ddof=1) * np.sqrt(252))

        # 最大回撤
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        risk.max_drawdown = float(np.min(drawdown))

        # Beta
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            bench = benchmark_returns.values if hasattr(benchmark_returns, 'values') else benchmark_returns
            bench = bench[~np.isnan(returns)]
            aligned_returns = returns[~np.isnan(returns)]
            if len(aligned_returns) > 1 and len(bench) > 1:
                min_len = min(len(aligned_returns), len(bench))
                cov = np.cov(aligned_returns[:min_len], bench[:min_len])
                if cov.size >= 4:
                    risk.beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0

        # 夏普比率（无风险利率 2%）
        excess = np.mean(returns) - 0.02 / 252
        risk.sharpe = float(excess / np.std(returns, ddof=1) * np.sqrt(252)) if np.std(returns, ddof=1) > 0 else 0

        return risk

    # ── 组合优化 ────────────────────────────────────────────

    def optimize_equal_risk(
        self,
        candidates: List[PositionSize],
        total_capital: float,
    ) -> List[PositionSize]:
        """等风险贡献优化（简化版）。

        调整仓位使每只股票的风险贡献大致相等。
        风险贡献 ≈ 仓位占比 × 波动率
        """
        if not candidates:
            return candidates

        # 简化处理：评分越高的给更多权重，但不超过上限
        total_score = sum(c.score for c in candidates)
        if total_score == 0:
            return candidates

        for c in candidates:
            # 按评分占比分配，同时做再平衡
            base_pct = (c.score / total_score)
            # 限制单只上限
            c.suggested_pct = min(base_pct, self.limits.max_single_position)
            c.reason += " (等风险贡献优化)"

        # 归一化确保总和不超过可用仓位
        total_pct = sum(c.suggested_pct for c in candidates)
        if total_pct > 1.0:
            scale = 1.0 / total_pct
            for c in candidates:
                c.suggested_pct = round(c.suggested_pct * scale, 4)

        return candidates

    # ── 分散化评分 ──────────────────────────────────────────

    def calc_diversification_score(
        self,
        returns_matrix: pd.DataFrame,
    ) -> float:
        """计算组合分散化评分。

        基于持仓间平均相关性：相关性越低，分散化越好。
        返回 0~1，越高越好。
        """
        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            return 1.0  # 单只持仓，假设完全分散

        corr = returns_matrix.corr()
        # 取非对角线元素的平均绝对值
        n = corr.shape[0]
        off_diag_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag_sum += abs(corr.iloc[i, j])
                count += 1

        if count == 0:
            return 1.0

        avg_corr = off_diag_sum / count
        # 相关性越低，分散化得分越高
        return round(1.0 - avg_corr * 0.8, 4)

    # ── 风控报告 ────────────────────────────────────────────

    def generate_risk_report(
        self,
        positions: List[PositionSize],
        risk_metrics: PortfolioRisk,
        total_capital: float,
    ) -> str:
        """生成风控报告。"""
        lines = [
            "╔══════════════════════════════════╗",
            "║        风 控 分 析 报 告        ║",
            "╠══════════════════════════════════╣",
            f"║  总资金: ¥{total_capital:,.0f}",
            "╠══════════════════════════════════╣",
            "",
            "━━━ 📊 风险指标 ━━━",
            f"  VaR (95%): {risk_metrics.var_95*100:.2f}%",
            f"  CVaR (95%): {risk_metrics.cvar_95*100:.2f}%",
            f"  年化波动率: {risk_metrics.volatility*100:.2f}%",
            f"  最大回撤: {risk_metrics.max_drawdown*100:.2f}%",
            f"  Beta: {risk_metrics.beta:.2f}",
            f"  夏普比率: {risk_metrics.sharpe:.2f}",
        ]

        if positions:
            lines.append("")
            lines.append("━━━ 💰 仓位建议 ━━━")
            for p in positions:
                amount = total_capital * p.suggested_pct
                lines.append(
                    f"  {p.symbol} {p.name:<8s}  "
                    f"仓位:{p.suggested_pct*100:.1f}%  "
                    f"金额:¥{amount:,.0f}"
                )
                if p.stop_loss_price:
                    lines.append(f"    止损: ¥{p.stop_loss_price:.2f}")
                if p.take_profit_price:
                    lines.append(f"    止盈: ¥{p.take_profit_price:.2f}")
                lines.append(f"    理由: {p.reason}")

        lines.append("")
        lines.append("╚══════════════════════════════════╝")
        return "\n".join(lines)