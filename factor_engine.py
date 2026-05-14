"""
因子挖掘引擎。

提供：
- 内置因子库（动量/波动/量价/质量/估值 5大类 30+因子）
- 因子计算（基于K线+基本面数据）
- IC/IR 分析（Rank IC / Pearson IC）
- 因子相关性矩阵
- 因子分层回测（多空分组收益）
- 自动因子合成（运算符组合发现新因子）
- 因子筛选与加权合成

用法:
    from factor_engine import FactorEngine

    engine = FactorEngine()
    factors = engine.compute_all(kline_df, fundamental_data)
    ic = engine.analyze_ic(factors, forward_returns)
    best = engine.select_factors(ic_results, min_ic=0.03)
    combined = engine.combine_factors(best)
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 因子定义 ──────────────────────────────────────────────

@dataclass
class FactorDef:
    """因子定义。"""
    name: str
    category: str  # momentum / volatility / volume / quality / value
    description: str = ""
    formula: str = ""  # 计算公式描述

# 内置因子库
FACTOR_LIBRARY = [
    # ── 动量因子 ────────────────────
    FactorDef("ret_1d", "momentum", "1日收益率", "close / close.shift(1) - 1"),
    FactorDef("ret_5d", "momentum", "5日收益率", "close / close.shift(5) - 1"),
    FactorDef("ret_20d", "momentum", "20日收益率", "close / close.shift(20) - 1"),
    FactorDef("ret_60d", "momentum", "60日收益率", "close / close.shift(60) - 1"),
    FactorDef("ma_ratio_5_20", "momentum", "MA5/MA20", "MA(5) / MA(20)"),
    FactorDef("ma_ratio_10_60", "momentum", "MA10/MA60", "MA(10) / MA(60)"),
    FactorDef("rsi_14", "momentum", "RSI(14)", "经典RSI"),
    FactorDef("macd_dif", "momentum", "MACD DIF", "EMA12-EMA26"),
    FactorDef("macd_hist", "momentum", "MACD柱", "DIF-DEA"),
    FactorDef("price_vs_52w_high", "momentum", "距52周高点", "close / 52周最高 - 1"),

    # ── 波动因子 ────────────────────
    FactorDef("vol_5d", "volatility", "5日波动率", "std(ret, 5)"),
    FactorDef("vol_20d", "volatility", "20日波动率", "std(ret, 20)"),
    FactorDef("vol_ratio_5_20", "volatility", "短/长波动比", "vol(5) / vol(20)"),
    FactorDef("atr_14", "volatility", "ATR(14)", "Average True Range"),
    FactorDef("boll_width", "volatility", "布林带宽", "(upper-lower)/MA"),
    FactorDef("max_dd_20d", "volatility", "20日最大回撤", "max drawdown 20d"),
    FactorDef("hl_ratio", "volatility", "日内振幅", "(high-low)/open"),

    # ── 量价因子 ────────────────────
    FactorDef("volume_ratio_5", "volume", "5日量比", "V / MA(V,5)"),
    FactorDef("volume_ratio_20", "volume", "20日量比", "V / MA(V,20)"),
    FactorDef("volume_trend", "volume", "量趋势", "V递增天数"),
    FactorDef("turnover_5d", "volume", "5日换手率均值", "MA(turnover,5)"),
    FactorDef("vwap_deviation", "volume", "VWAP偏离", "close / VWAP - 1"),
    FactorDef("obv_ratio", "volume", "OBV变化率", "OBV / OBV.shift(20) - 1"),
    FactorDef("money_flow_20d", "volume", "20日资金流向", "sum(money_flow, 20)"),

    # ── 质量因子 ────────────────────
    FactorDef("roe", "quality", "ROE", "净资产收益率"),
    FactorDef("roa", "quality", "ROA", "总资产收益率"),
    FactorDef("gross_margin", "quality", "毛利率", "(收入-成本)/收入"),
    FactorDef("net_margin", "quality", "净利率", "净利润/收入"),
    FactorDef("debt_ratio", "quality", "资产负债率", "总负债/总资产"),
    FactorDef("eps_growth", "quality", "EPS增长率", "EPS同比"),
    FactorDef("revenue_growth", "quality", "营收增长率", "营收同比"),

    # ── 估值因子 ────────────────────
    FactorDef("pe_ttm", "value", "PE(TTM)", "市盈率"),
    FactorDef("pb", "value", "PB", "市净率"),
    FactorDef("ps", "value", "PS", "市销率"),
    FactorDef("pe_percentile", "value", "PE分位数", "当前PE在历史分位"),
    FactorDef("pb_percentile", "value", "PB分位数", "当前PB在历史分位"),
    FactorDef("dividend_yield", "value", "股息率", "每股分红/股价"),
]


@dataclass
class ICResult:
    """IC分析结果。"""
    factor_name: str
    rank_ic: float = 0.0  # Rank IC 均值
    pearson_ic: float = 0.0  # Pearson IC 均值
    ic_std: float = 0.0  # IC标准差
    ir: float = 0.0  # Information Ratio = IC_mean / IC_std
    ic_series: pd.Series = None  # 逐期IC序列
    ic_positive_ratio: float = 0.0  # IC>0 的比例
    long_return: float = 0.0  # 多头组收益
    short_return: float = 0.0  # 空头组收益
    long_short_spread: float = 0.0  # 多空收益差


@dataclass
class FactorResult:
    """单因子计算结果。"""
    name: str
    values: pd.Series
    category: str = ""

    @property
    def valid_count(self) -> int:
        return self.values.notna().sum()

    @property
    def coverage(self) -> float:
        return self.valid_count / len(self.values) if len(self.values) > 0 else 0


# ── 主引擎 ─────────────────────────────────────────────────

class FactorEngine:
    """因子挖掘引擎。

    计算 → 分析 → 筛选 → 合成 → 发现

    用法:
        engine = FactorEngine()
        factors = engine.compute_all(kline_df)
        ic = engine.analyze_ic(factors, forward_returns)
        best = engine.select_factors(ic, min_ir=0.3)
        combined = engine.synthesize(best)
        new_factors = engine.discover(factors, ic, max_ops=2)
    """

    def __init__(self):
        self._factors: Dict[str, FactorResult] = {}
        self._ic_results: Dict[str, ICResult] = {}

    # ── 因子计算 ──────────────────────────────────────────────

    def compute_all(
        self,
        kline: pd.DataFrame,
        fundamental: dict = None,
    ) -> Dict[str, FactorResult]:
        """计算所有内置因子。

        Args:
            kline: OHLCV DataFrame, 必须含 close/high/low/open/volume
            fundamental: 可选基本面数据 dict

        Returns:
            {factor_name: FactorResult}
        """
        df = kline.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        returns = close.pct_change()

        factors = {}

        # ── 动量因子 ────────────────────
        factors["ret_1d"] = self._make(returns, "ret_1d", "momentum")
        factors["ret_5d"] = self._make(close.pct_change(5), "ret_5d", "momentum")
        factors["ret_20d"] = self._make(close.pct_change(20), "ret_20d", "momentum")
        factors["ret_60d"] = self._make(close.pct_change(60), "ret_60d", "momentum")

        ma5 = close.rolling(5).mean()  # noqa: F841
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        factors["ma_ratio_5_20"] = self._make(close.rolling(5).mean() / ma20 - 1, "ma_ratio_5_20", "momentum")
        factors["ma_ratio_10_60"] = self._make(ma10 / ma60 - 1, "ma_ratio_10_60", "momentum")

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        factors["rsi_14"] = self._make(100 - 100 / (1 + rs), "rsi_14", "momentum")
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        factors["macd_dif"] = self._make(dif, "macd_dif", "momentum")
        factors["macd_hist"] = self._make(dif - dea, "macd_hist", "momentum")

        # 52周高点
        high_52w = close.rolling(252).max()
        factors["price_vs_52w_high"] = self._make(close / high_52w - 1, "price_vs_52w_high", "momentum")

        # ── 波动因子 ────────────────────
        factors["vol_5d"] = self._make(returns.rolling(5).std(), "vol_5d", "volatility")
        factors["vol_20d"] = self._make(returns.rolling(20).std(), "vol_20d", "volatility")
        factors["vol_ratio_5_20"] = self._make(
            returns.rolling(5).std() / returns.rolling(20).std().replace(0, np.nan),
            "vol_ratio_5_20", "volatility",
        )

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        factors["atr_14"] = self._make(tr.rolling(14).mean(), "atr_14", "volatility")

        # 布林带宽度
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_width = (bb_std * 2) / bb_mid
        factors["boll_width"] = self._make(bb_width, "boll_width", "volatility")

        # 20日最大回撤
        peak_20 = close.rolling(20).max()
        dd_20 = close / peak_20 - 1
        factors["max_dd_20d"] = self._make(dd_20, "max_dd_20d", "volatility")

        # 日内振幅
        factors["hl_ratio"] = self._make((high - low) / open_, "hl_ratio", "volatility")

        # ── 量价因子 ────────────────────
        vol_ma5 = volume.rolling(5).mean()
        factors["volume_ratio_5"] = self._make(volume / vol_ma5, "volume_ratio_5", "volume")
        factors["volume_ratio_20"] = self._make(volume / volume.rolling(20).mean(), "volume_ratio_20", "volume")

        # 量趋势（连续放量天数）
        vol_increasing = (volume > volume.shift()).astype(int)
        factors["volume_trend"] = self._make(
            vol_increasing.rolling(10).sum(), "volume_trend", "volume"
        )

        # 换手率（如果有）
        if "turnover" in df.columns:
            factors["turnover_5d"] = self._make(df["turnover"].rolling(5).mean(), "turnover_5d", "volume")

        # VWAP偏离
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(10).sum() / volume.rolling(10).sum()
        factors["vwap_deviation"] = self._make(close / vwap - 1, "vwap_deviation", "volume")

        # OBV
        obv = (np.sign(close.diff()) * volume).cumsum()
        factors["obv_ratio"] = self._make(obv / obv.shift(20) - 1, "obv_ratio", "volume")

        # 资金流向
        mf = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
        factors["money_flow_20d"] = self._make(mf.rolling(20).sum(), "money_flow_20d", "volume")

        self._factors = factors
        return factors

    def compute_fundamental_factors(
        self,
        fundamental_data: Dict[str, float],
    ) -> Dict[str, FactorResult]:
        """从基本面数据计算质量/估值因子。

        Args:
            fundamental_data: 包含 ROE/ROA/PE/PB/PS 等字段的dict
        """
        factors = {}

        mapping = {
            "roe": "quality",
            "roa": "quality",
            "gross_margin": "quality",
            "net_margin": "quality",
            "debt_ratio": "quality",
            "eps_growth": "quality",
            "revenue_growth": "quality",
            "pe_ttm": "value",
            "pb": "value",
            "ps": "value",
            "pe_percentile": "value",
            "pb_percentile": "value",
            "dividend_yield": "value",
        }

        for name, category in mapping.items():
            value = fundamental_data.get(name)
            if value is not None and not np.isnan(value):
                factors[name] = FactorResult(
                    name=name,
                    values=pd.Series([value]),
                    category=category,
                )

        self._factors.update(factors)
        return factors

    def _make(self, series: pd.Series, name: str, category: str) -> FactorResult:
        """包装因子结果。"""
        return FactorResult(name=name, values=series, category=category)

    # ── IC 分析 ────────────────────────────────────────────────

    def analyze_ic(
        self,
        factors: Dict[str, FactorResult],
        forward_returns: pd.Series,
        periods: List[int] = None,
    ) -> Dict[str, ICResult]:
        """计算各因子的 IC（信息系数）。

        Args:
            factors: compute_all() 的结果
            forward_returns: 未来N日收益率（对齐日期）
            periods: IC计算周期列表，默认 [1, 5, 20]

        Returns:
            {factor_name: ICResult}
        """
        if periods is None:
            periods = [1, 5, 20]

        results = {}

        for name, fr in factors.items():
            if fr.values is None or len(fr.values.dropna()) < 30:
                continue

            # 对齐因子值和未来收益
            common_idx = fr.values.dropna().index.intersection(forward_returns.dropna().index)
            if len(common_idx) < 30:
                continue

            f_vals = fr.values.loc[common_idx].rank(pct=True)  # 转为横截面排名
            fwd_rets = forward_returns.loc[common_idx]

            # Rank IC — use numpy-based approximation (避免依赖scipy)
            try:
                rank_ic = f_vals.corr(fwd_rets.rank(pct=True), method="spearman")
            except Exception:
                # scipy missing fallback: Pearson on ranks ≈ Spearman for large N
                rank_ic = f_vals.rank().corr(fwd_rets.rank(), method="pearson")

            try:
                pearson_ic = f_vals.corr(fwd_rets, method="pearson")
            except Exception:
                pearson_ic = np.corrcoef(f_vals.fillna(0), fwd_rets.fillna(0))[0, 1]

            # 滚动IC序列（计算稳定性）
            win_size = min(60, len(common_idx) // 3)
            ic_series = []
            for i in range(win_size, len(common_idx)):
                sub_f = f_vals.iloc[i - win_size:i]
                sub_r = fwd_rets.iloc[i - win_size:i]
                ic = sub_f.corr(sub_r.rank(pct=True), method="spearman")
                if not np.isnan(ic):
                    ic_series.append(ic)

            ic_series = pd.Series(ic_series)
            ic_std = ic_series.std() if len(ic_series) > 0 else 0
            ir_value = float(rank_ic / ic_std) if ic_std > 0 else 0

            # IC > 0 的比例
            ic_pos_ratio = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0

            # 分层回测（Top/Bottom 20%）
            n = len(f_vals)
            top_n = max(5, n // 5)
            top_idx = f_vals.nlargest(top_n).index
            bot_idx = f_vals.nsmallest(top_n).index

            long_ret = float(fwd_rets.loc[top_idx].mean()) if len(top_idx) > 0 else 0
            short_ret = float(fwd_rets.loc[bot_idx].mean()) if len(bot_idx) > 0 else 0

            results[name] = ICResult(
                factor_name=name,
                rank_ic=float(rank_ic) if not np.isnan(rank_ic) else 0,
                pearson_ic=float(pearson_ic) if not np.isnan(pearson_ic) else 0,
                ic_std=float(ic_std),
                ir=float(ir_value),
                ic_series=ic_series if len(ic_series) > 0 else None,
                ic_positive_ratio=float(ic_pos_ratio),
                long_return=long_ret,
                short_return=short_ret,
                long_short_spread=long_ret - short_ret,
            )

        self._ic_results = results
        return results

    # ── 因子筛选 ────────────────────────────────────────────────

    def select_factors(
        self,
        ic_results: Dict[str, ICResult] = None,
        min_ic: float = 0.02,
        min_ir: float = 0.3,
        min_ic_pos_ratio: float = 0.55,
        max_correlation: float = 0.7,
    ) -> List[str]:
        """筛选有效因子。

        Criteria:
        - |Rank IC| > min_ic
        - IR > min_ir
        - IC>0比例 > min_ic_pos_ratio
        - 因子间去相关（保留IR更高的）

        Returns:
            筛选后的因子名列表
        """
        if ic_results is None:
            ic_results = self._ic_results

        # 第一轮：单因子筛选
        candidates = []
        for name, ic in ic_results.items():
            if abs(ic.rank_ic) < min_ic:
                continue
            if abs(ic.ir) < min_ir:
                continue
            if ic.ic_positive_ratio < min_ic_pos_ratio:
                continue
            candidates.append((name, abs(ic.ir)))

        if not candidates:
            return []

        candidates.sort(key=lambda x: -x[1])

        # 第二轮：去相关
        selected = []
        for name, ir_val in candidates:
            if name not in self._factors:
                continue

            too_correlated = False
            for sel_name in selected:
                if sel_name in self._factors:
                    corr = self._factors[name].values.corr(
                        self._factors[sel_name].values, method="spearman"
                    )
                    if abs(corr) > max_correlation:
                        too_correlated = True
                        break

            if not too_correlated:
                selected.append(name)

        return selected

    # ── 因子合成 ────────────────────────────────────────────────

    def synthesize(self, factor_names: List[str], method: str = "ic_weighted") -> FactorResult:
        """合成因子（多因子加权组合）。

        Args:
            factor_names: select_factors() 返回的因子列表
            method: 加权方式
                - "ic_weighted": IC加权
                - "equal": 等权
                - "ir_weighted": IR加权

        Returns:
            合成后的因子结果
        """
        if not factor_names:
            return FactorResult(name="combined", values=pd.Series(dtype=float))

        weights = {}
        for name in factor_names:
            if name in self._ic_results:
                ic = self._ic_results[name]
                if method == "ic_weighted":
                    weights[name] = abs(ic.rank_ic)
                elif method == "ir_weighted":
                    weights[name] = abs(ic.ir)
                else:
                    weights[name] = 1.0
            else:
                weights[name] = 1.0

        total_w = sum(weights.values())
        if total_w == 0:
            return FactorResult(name="combined", values=pd.Series(dtype=float))

        # 标准化各因子到 Z-score
        combined = pd.Series(0.0, index=self._factors[factor_names[0]].values.index)
        for name in factor_names:
            fr = self._factors[name]
            z = (fr.values - fr.values.mean()) / fr.values.std(ddof=1).replace(0, 1)
            combined += z * (weights[name] / total_w)

        return FactorResult(
            name="combined",
            values=combined,
            category="combined",
        )

    # ── 因子相关性矩阵 ──────────────────────────────────────────

    def correlation_matrix(self, factor_names: List[str] = None) -> pd.DataFrame:
        """计算因子间相关性矩阵。"""
        if factor_names is None:
            factor_names = list(self._factors.keys())

        data = {}
        for name in factor_names:
            if name in self._factors:
                data[name] = self._factors[name].values.dropna()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df.corr(method="spearman")

    # ── 自动因子发现 ────────────────────────────────────────────

    def discover(
        self,
        factor_names: List[str] = None,
        max_ops: int = 1,
        min_ic_improvement: float = 0.01,
    ) -> List[Tuple[str, str, float]]:
        """自动因子发现：通过运算符组合发现新因子。

        运算符: +, -, *, /, rank

        例如: ret_5d * volume_ratio_5 可能比单独的因子有更高IC。

        Args:
            factor_names: 基础因子池（默认全部已计算因子）
            max_ops: 最大运算符数量
            min_ic_improvement: 最小IC提升阈值

        Returns:
            [(新因子名, 表达式, IC), ...]
        """
        if factor_names is None:
            factor_names = list(self._factors.keys())

        operators = ["add", "sub", "mul", "div"]
        discoveries = []
        seen = set()

        for op in operators:
            for n1, n2 in combinations(factor_names, 2):
                if n1 not in self._factors or n2 not in self._factors:
                    continue

                name = f"{n1}_{op}_{n2}"
                if name in seen:
                    continue
                seen.add(name)

                f1 = self._factors[n1].values.rank(pct=True)
                f2 = self._factors[n2].values.rank(pct=True)

                if op == "add":
                    combined = f1 + f2
                elif op == "sub":
                    combined = f1 - f2
                elif op == "mul":
                    combined = f1 * f2
                elif op == "div":
                    combined = f1 / f2.replace(0, np.nan)
                else:
                    continue

                # 快速IC估算
                if "ret_1d" in self._factors:
                    fwd = self._factors["ret_1d"].values.shift(-1)
                    common = combined.dropna().index.intersection(fwd.dropna().index)
                    if len(common) > 30:
                        ic = combined.loc[common].corr(
                            fwd.loc[common].rank(pct=True), method="spearman"
                        )
                        if abs(ic) > 0:
                            # 对比基础因子中最好的
                            best_base_ic = max(
                                abs(self._ic_results.get(n1, ICResult(factor_name=n1)).rank_ic),
                                abs(self._ic_results.get(n2, ICResult(factor_name=n2)).rank_ic),
                            )
                            if abs(ic) > best_base_ic + min_ic_improvement:
                                discoveries.append((name, f"{n1} {op} {n2}", abs(ic)))

        discoveries.sort(key=lambda x: -x[2])
        return discoveries[:20]  # Top 20

    # ── 因子报告 ────────────────────────────────────────────────

    def generate_report(self, ic_results: Dict[str, ICResult] = None) -> str:
        """生成因子分析报告。"""
        if ic_results is None:
            ic_results = self._ic_results

        if not ic_results:
            return "暂无IC分析结果。请先运行 analyze_ic()"

        lines = [
            "╔══════════════════════════════════════╗",
            "║         因 子 挖 掘 报 告          ║",
            "╠══════════════════════════════════════╣",
        ]

        # 按IR排序
        ranked = sorted(ic_results.items(), key=lambda x: -abs(x[1].ir))

        lines.append("║  排名 | 因子          | RankIC |  IR  | IC>0%")
        lines.append("║  ─────┼──────────────┼────────┼──────┼───────")

        for i, (name, icr) in enumerate(ranked[:15], 1):
            sign = "+" if icr.rank_ic > 0 else ""
            lines.append(
                f"║  {i:3d}  | {name:<12s} | {sign}{icr.rank_ic:.3f} | "
                f"{icr.ir:.2f} | {icr.ic_positive_ratio:.0%}"
            )

        lines.append("╠══════════════════════════════════════╣")

        # 各分类汇总
        categories = {}
        for name, icr in ic_results.items():
            for fd in FACTOR_LIBRARY:
                if fd.name == name:
                    cat = fd.category
                    categories.setdefault(cat, []).append(abs(icr.rank_ic))

        for cat, ics in sorted(categories.items()):
            avg_ic = np.mean(ics) if ics else 0
            lines.append(f"║  {cat:<10s}: 均IC={avg_ic:.4f}  ({len(ics)} 因子)")

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)