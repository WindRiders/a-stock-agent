"""增强回测模块。

支持：多资产组合回测、参数扫描优化、Walk-Forward 分析。
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """参数优化结果。"""

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_result: Optional[BacktestResult] = None
    best_score: float = 0.0
    all_results: List[Dict] = field(default_factory=list)
    param_importance: Dict[str, float] = field(default_factory=dict)


@dataclass 
class WalkForwardResult:
    """Walk-Forward 分析结果。"""

    windows: List[Dict] = field(default_factory=list)
    avg_insample_return: float = 0.0
    avg_outsample_return: float = 0.0
    robustness_score: float = 0.0  # 样本外稳定性评分


class EnhancedBacktestEngine(BacktestEngine):
    """增强版回测引擎。

    扩展基础引擎，添加：
    - 多资产组合回测
    - 参数网格搜索
    - Walk-Forward 分析
    """

    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)

    # ── 多资产组合回测 ─────────────────────────────────────

    def run_portfolio(
        self,
        signals_by_symbol: Dict[str, pd.DataFrame],
        price_data: Dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """多资产组合回测。

        同时回测多只股票，模拟组合交易。
        """
        # 合并所有信号
        all_signals = []
        for symbol, df in signals_by_symbol.items():
            df = df.copy()
            df["symbol"] = symbol
            all_signals.append(df)

        if not all_signals:
            return self._empty_result()

        combined = pd.concat(all_signals, ignore_index=True)
        combined = combined.sort_values("date")

        return self.run(combined, price_data)

    # ── 参数网格搜索 ───────────────────────────────────────

    def optimize_parameters(
        self,
        param_grid: Dict[str, List[Any]],
        signal_generator: Callable,
        price_data: Dict[str, pd.DataFrame],
        score_metric: str = "sharpe_ratio",
        verbose: bool = True,
    ) -> OptimizationResult:
        """参数网格搜索优化。

        Args:
            param_grid: 参数搜索空间，如 {"stop_loss": [-0.05, -0.08, -0.10], "take_profit": [0.15, 0.20, 0.25]}
            signal_generator: 信号生成函数，接受 params dict，返回 signals_df
            price_data: 价格数据
            score_metric: 优化目标（sharpe_ratio / total_return / calmar_ratio）
            verbose: 是否打印进度

        Returns:
            OptimizationResult
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        if verbose:
            logger.info("参数网格搜索: %d 组参数", len(combinations))

        best_score = -float("inf")
        best_params = None
        best_result = None
        all_results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # 生成信号
            try:
                signals_df = signal_generator(params)
            except Exception as e:
                logger.debug("参数 %s 信号生成失败: %s", params, e)
                continue

            if signals_df.empty:
                continue

            # 运行回测
            self.cash = self.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_history = []

            result = self.run(signals_df, price_data)

            # 评优
            score = getattr(result, score_metric, 0)
            if score_metric == "max_drawdown":
                score = -abs(score)  # 回撤越小越好

            all_results.append({
                "params": params,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

            if verbose and len(all_results) % 10 == 0:
                logger.info("  已测试 %d/%d 组", len(all_results), len(combinations))

        # 计算参数重要性（每个参数对目标的影响）
        importance = self._calc_param_importance(
            all_results, param_names, score_metric
        )

        return OptimizationResult(
            best_params=best_params or {},
            best_result=best_result,
            best_score=best_score,
            all_results=all_results,
            param_importance=importance,
        )

    def _calc_param_importance(
        self,
        results: List[Dict],
        param_names: List[str],
        metric: str,
    ) -> Dict[str, float]:
        """计算参数重要性。

        对于每个参数，比较取不同值时的平均得分变化幅度。
        """
        if len(results) < 2:
            return {}

        importance = {}
        for param in param_names:
            # 按参数值分组
            grouped = {}
            for r in results:
                val = str(r["params"][param])
                if val not in grouped:
                    grouped[val] = []
                grouped[val].append(r["score"])

            # 计算组间差异
            group_means = [np.mean(scores) for scores in grouped.values()]
            if len(group_means) > 1:
                importance[param] = round(np.std(group_means) / (abs(np.mean(group_means)) + 1e-8), 4)
            else:
                importance[param] = 0.0

        return importance

    # ── Walk-Forward 分析 ──────────────────────────────────

    def walk_forward(
        self,
        signal_generator: Callable,
        price_data: Dict[str, pd.DataFrame],
        train_periods: int = 120,
        test_periods: int = 20,
        step_size: int = 20,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """Walk-Forward 分析。

        滑动窗口训练+测试，评估策略在样本外的稳定性。

        Args:
            signal_generator: 信号生成函数 (start_date, end_date, params) -> signals_df
            price_data: 完整价格数据
            train_periods: 训练期长度（交易日数）
            test_periods: 测试期长度
            step_size: 窗口滑动步长
            verbose: 是否打印进度

        Returns:
            WalkForwardResult
        """
        # 获取全部交易日
        all_dates = set()
        for df in price_data.values():
            if "date" in df.columns:
                all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        if len(all_dates) < train_periods + test_periods:
            logger.warning("数据不足：需要 %d 交易日，实际 %d",
                          train_periods + test_periods, len(all_dates))
            return WalkForwardResult()

        windows = []
        start_idx = 0

        while start_idx + train_periods + test_periods <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end = all_dates[start_idx + train_periods - 1]
            test_start = all_dates[start_idx + train_periods]
            test_end = all_dates[start_idx + train_periods + test_periods - 1]

            # 样本内训练
            try:
                train_signals = signal_generator(train_start, train_end)
            except Exception as e:
                logger.debug("训练期信号生成失败: %s", e)
                start_idx += step_size
                continue

            # 样本内回测
            self.cash = self.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_history = []
            in_result = self.run(train_signals, price_data)

            # 样本外测试
            try:
                test_signals = signal_generator(test_start, test_end)
            except Exception:
                start_idx += step_size
                continue

            self.cash = self.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_history = []
            out_result = self.run(test_signals, price_data)

            windows.append({
                "train": (str(train_start)[:10], str(train_end)[:10]),
                "test": (str(test_start)[:10], str(test_end)[:10]),
                "insample_return": in_result.total_return,
                "outsample_return": out_result.total_return,
                "insample_sharpe": in_result.sharpe_ratio,
                "outsample_sharpe": out_result.sharpe_ratio,
            })

            if verbose:
                logger.info(
                    "窗口 [%s ~ %s] → [%s ~ %s] | 样本内: %.1f%% | 样本外: %.1f%%",
                    str(train_start)[:10], str(train_end)[:10],
                    str(test_start)[:10], str(test_end)[:10],
                    in_result.total_return, out_result.total_return,
                )

            start_idx += step_size

        if not windows:
            return WalkForwardResult()

        # 汇总
        insample_rets = [w["insample_return"] for w in windows]
        outsample_rets = [w["outsample_return"] for w in windows]

        # 稳健性评分：样本外收益的稳定性
        if outsample_rets:
            mean_out = np.mean(outsample_rets)
            std_out = np.std(outsample_rets, ddof=1)
            robustness = mean_out / (std_out + 1e-8) if std_out > 0 else 10.0
        else:
            robustness = 0.0

        return WalkForwardResult(
            windows=windows,
            avg_insample_return=round(np.mean(insample_rets), 2) if insample_rets else 0,
            avg_outsample_return=round(np.mean(outsample_rets), 2) if outsample_rets else 0,
            robustness_score=round(robustness, 2),
        )

    # ── 优化报告 ───────────────────────────────────────────

    def optimization_summary(self, result: OptimizationResult) -> str:
        """格式化参数优化结果。"""
        lines = [
            "╔══════════════════════════════════╗",
            "║      参 数 优 化 报 告          ║",
            "╠══════════════════════════════════╣",
            f"║  测试参数组: {len(result.all_results)}",
            f"║  最优得分: {result.best_score:.4f}",
            "╠══════════════════════════════════╣",
            "",
            "  📌 最优参数:",
        ]
        for k, v in result.best_params.items():
            lines.append(f"    {k}: {v}")

        if result.best_result:
            lines.append("")
            lines.append("  📊 最优结果:")
            lines.append(f"    总收益: {result.best_result.total_return:.2f}%")
            lines.append(f"    夏普: {result.best_result.sharpe_ratio:.2f}")
            lines.append(f"    最大回撤: {result.best_result.max_drawdown:.2f}%")
            lines.append(f"    胜率: {result.best_result.win_rate:.1f}%")

        if result.param_importance:
            lines.append("")
            lines.append("  📈 参数重要性:")
            for k, v in sorted(result.param_importance.items(), key=lambda x: -x[1]):
                lines.append(f"    {k}: {v:.4f}")

        lines.append("")
        lines.append("╚══════════════════════════════════╝")
        return "\n".join(lines)