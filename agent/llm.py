"""LLM 智能分析模块。

使用大语言模型对股票评分、市场数据和新闻进行深度解读，
生成自然语言投资报告和市场评论。
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from analysis.scoring import StockScore

logger = logging.getLogger(__name__)


# ── 系统提示词 ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位资深的A股量化分析师，擅长技术分析、基本面研究和市场情绪判断。
你的分析风格：专业、客观、数据驱动，同时给出明确的投资逻辑。
用中文回复，要点清晰，避免空洞的套话。

分析框架：
- 技术面：趋势、动能、量价关系、关键支撑阻力
- 基本面：估值水平、盈利能力、成长性、行业对比
- 风险面：市场环境、政策风险、流动性风险
- 操作建议：基于以上分析给出具体建议（仅供参考）"""


# ── 数据类 ──────────────────────────────────────────────────


@dataclass
class LLMAnalysis:
    """LLM分析结果。"""

    # 市场总评
    market_summary: str = ""
    market_sentiment: str = ""  # bullish / neutral / bearish

    # 个股点评
    stock_commentaries: List[Dict] = None

    # 策略建议
    strategy_advice: str = ""
    risk_warnings: List[str] = None

    # 报告全文
    full_report: str = ""

    # 元信息
    model: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if self.stock_commentaries is None:
            self.stock_commentaries = []
        if self.risk_warnings is None:
            self.risk_warnings = []


class LLMAnalyzer:
    """LLM驱动的智能分析器。

    将量化数据转化为自然语言分析报告。
    支持两种模式：
    1. 本地规则模式（无需LLM，基于模板生成）
    2. LLM模式（调用大模型生成深度分析）
    """

    def __init__(self, model: str = None):
        self.model = model

    # ── 市场总评（本地规则） ─────────────────────────────────

    def generate_market_summary(
        self,
        scores: List[StockScore],
        market_df: pd.DataFrame = None,
    ) -> str:
        """基于数据生成市场总评（无需LLM）。"""
        if not scores:
            return "暂无有效数据，无法生成市场分析。"

        lines = ["## 市场概况\n"]

        # 统计信号分布
        buys = sum(1 for s in scores if s.signal in ("BUY", "STRONG_BUY"))
        sells = sum(1 for s in scores if s.signal in ("SELL", "STRONG_SELL"))
        holds = sum(1 for s in scores if s.signal == "HOLD")

        lines.append(f"- 分析股票数: {len(scores)}")
        lines.append(f"- 买入信号: {buys} 只 ({buys/len(scores)*100:.0f}%)")
        lines.append(f"- 卖出信号: {sells} 只 ({sells/len(scores)*100:.0f}%)")
        lines.append(f"- 持有/观望: {holds} 只 ({holds/len(scores)*100:.0f}%)")

        # 评分统计
        avg_score = sum(s.total_score for s in scores) / len(scores)
        lines.append(f"- 平均综合评分: {avg_score:.2f}")

        # 市场情绪判断
        if buys > sells * 2:
            sentiment = "偏乐观"
            lines.append(f"- 市场情绪: 🔥 {sentiment}（买入信号显著多于卖出）")
        elif sells > buys * 2:
            sentiment = "偏谨慎"
            lines.append(f"- 市场情绪: ⚠️ {sentiment}（卖出信号显著多于买入）")
        else:
            sentiment = "中性"
            lines.append(f"- 市场情绪: ⏸️ {sentiment}（多空力量均衡）")

        # 大盘指数参考
        if market_df is not None and not market_df.empty:
            try:
                latest = float(market_df["close"].iloc[-1])
                ma20 = float(market_df["close"].rolling(20).mean().iloc[-1])
                if latest > ma20:
                    lines.append(f"- 大盘参考: 沪深300在20日均线上方，趋势偏多")
                else:
                    lines.append(f"- 大盘参考: 沪深300在20日均线下方，注意系统性风险")
            except Exception:
                pass

        return "\n".join(lines)

    # ── 个股点评（本地规则） ─────────────────────────────────

    def generate_stock_commentary(self, score: StockScore) -> str:
        """为单只股票生成分析点评。"""
        lines = [f"\n### {score.name or '---'} ({score.symbol})\n"]

        # 价格
        if score.latest_price:
            lines.append(f"**最新价**: ¥{score.latest_price:.2f}")

        # 评级和信号
        rating_emoji = {"A": "🌟", "B": "📈", "C": "⏸️", "D": "⚠️"}
        signal_emoji = {
            "STRONG_BUY": "🔥 强烈推荐",
            "BUY": "📈 建议关注",
            "HOLD": "⏸️ 观望",
            "SELL": "📉 建议回避",
            "STRONG_SELL": "💀 强烈回避",
        }
        lines.append(f"**评级**: {rating_emoji.get(score.rating, '')}{score.rating}")
        lines.append(f"**信号**: {signal_emoji.get(score.signal, score.signal)}")

        # 评分拆解
        lines.append(f"\n评分明细:")
        lines.append(f"  技术面: {score.tech_score:+d}/+11")
        lines.append(f"  基本面: {score.fund_score:+d}/+6")
        lines.append(f"  消息面: {score.sentiment_score:+d}/+2")
        lines.append(f"  资金面: {score.capital_score:+d}/+2")
        lines.append(f"  综合: {score.total_score:.2f}")

        # 估值
        if score.pe is not None:
            pe_text = f"{score.pe:.1f}" if score.pe > 0 else "亏损"
            lines.append(f"\n**PE(TTM)**: {pe_text}")
        if score.pb is not None:
            lines.append(f"**PB**: {score.pb:.2f}")
        if score.volume_ratio is not None:
            vol_text = "放量" if score.volume_ratio > 1.5 else ("缩量" if score.volume_ratio < 0.5 else "正常")
            lines.append(f"**量比**: {score.volume_ratio:.2f} ({vol_text})")

        # 看多理由
        if score.reasons:
            lines.append(f"\n**看多理由**:")
            for r in score.reasons:
                lines.append(f"  ✅ {r}")

        # 风险提示
        if score.warnings:
            lines.append(f"\n**风险提示**:")
            for w in score.warnings:
                lines.append(f"  ⚠️ {w}")

        # 投资逻辑总结
        if score.total_score >= 0.7:
            logic = "各项指标均表现优异，技术面和基本面共振，可重点跟踪。"
        elif score.total_score >= 0.4:
            logic = "综合表现良好，有一定投资价值，建议持续关注。"
        elif score.total_score >= 0.1:
            logic = "表现中规中矩，缺乏明显催化剂，建议观望为主。"
        else:
            logic = "多项指标不达预期，建议回避，等待基本面或技术面改善信号。"
        lines.append(f"\n**投资逻辑**: {logic}")

        return "\n".join(lines)

    # ── LLM 调用接口 ────────────────────────────────────────

    def build_market_prompt(self, scores: List[StockScore], market_df: pd.DataFrame = None) -> str:
        """构建市场分析提示词。"""
        lines = [
            "请基于以下A股量化扫描结果，生成一份专业的市场分析报告。",
            f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"分析股票数: {len(scores)}",
            "",
            "## 综合评分排名 Top 10\n",
        ]

        for i, s in enumerate(scores[:10], 1):
            signal_map = {"STRONG_BUY": "🔥强烈买入", "BUY": "📈买入", "HOLD": "⏸️持有", "SELL": "📉卖出", "STRONG_SELL": "💀强烈卖出"}
            lines.append(
                f"{i}. {s.symbol} {s.name or ''} "
                f"总分:{s.total_score:.2f} 评级:{s.rating} "
                f"技术:{s.tech_score:+d} 基本:{s.fund_score:+d} "
                f"PE:{s.pe or 'N/A'} "
                f"信号:{signal_map.get(s.signal, s.signal)}"
            )

        # 信号统计
        buys = sum(1 for s in scores if s.signal in ("BUY", "STRONG_BUY"))
        sells = sum(1 for s in scores if s.signal in ("SELL", "STRONG_SELL"))
        lines.append(f"\n买入信号: {buys} | 卖出信号: {sells} | 合计: {len(scores)}")

        lines.append(f"\n请从以下角度分析：")
        lines.append("1. 当前市场情绪和整体判断")
        lines.append("2. 值得重点关注的前3只股票及理由")
        lines.append("3. 主要风险因素")
        lines.append("4. 操作建议（仅供参考）")

        return "\n".join(lines)

    def build_stock_deep_prompt(self, score: StockScore, kline_summary: str = "") -> str:
        """构建个股深度分析提示词。"""
        return f"""请对以下A股股票进行深度分析：

股票代码: {score.symbol}
股票名称: {score.name or '无'}
最新价: ¥{score.latest_price or 0:.2f}
评级: {score.rating}
综合评分: {score.total_score:.2f}

评分明细:
- 技术面: {score.tech_score:+d}/+11
- 基本面: {score.fund_score:+d}/+6
- 消息面: {score.sentiment_score:+d}/+2
- 资金面: {score.capital_score:+d}/+2

估值: PE={score.pe or 'N/A'}, PB={score.pb or 'N/A'}
量比: {score.volume_ratio or 'N/A'}

{kline_summary}

看多理由: {', '.join(score.reasons) if score.reasons else '无'}
风险提示: {', '.join(score.warnings) if score.warnings else '无'}

请从技术面、基本面、资金面三个维度分析，给出投资建议和风险提示。"""

    # ── 完整报告生成 ────────────────────────────────────────

    def generate_full_report(
        self,
        scores: List[StockScore],
        market_df: pd.DataFrame = None,
        strategy_name: str = "",
    ) -> str:
        """生成完整的自然语言分析报告。

        包含：市场总评 + 个股点评 + 投资建议 + 风险提示
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║        A股智能分析报告（量化 + AI 解读）                ║",
            f"║  生成时间: {now}                         ║",
            "╠══════════════════════════════════════════════════════════╣",
            "",
        ]

        if strategy_name:
            lines.append(f"策略引擎: {strategy_name}")

        # 1. 市场总评
        lines.append(self.generate_market_summary(scores, market_df))
        lines.append("")

        # 2. 重点关注（买入信号）
        buy_scores = [s for s in scores if s.signal in ("BUY", "STRONG_BUY")]
        if buy_scores:
            lines.append("━" * 50)
            lines.append("🎯 重点关注（买入/强烈买入信号）")
            lines.append("━" * 50)
            for s in buy_scores[:5]:
                lines.append(self.generate_stock_commentary(s))

        # 3. Top 名单
        lines.append("")
        lines.append("━" * 50)
        lines.append("📊 综合评分 TOP 10")
        lines.append("━" * 50)
        for i, s in enumerate(scores[:10], 1):
            lines.append(
                f"  {i:>2}. {s.symbol} {s.name:<8s}  "
                f"总分:{s.total_score:.2f}  "
                f"评级:{s.rating}  "
                f"信号:{s.signal}"
            )

        # 4. 风险提示
        all_warnings = []
        for s in scores:
            all_warnings.extend(s.warnings)

        if all_warnings:
            lines.append("")
            lines.append("━" * 50)
            lines.append("⚠️ 风险汇总")
            lines.append("━" * 50)
            seen = set()
            for w in all_warnings[:10]:
                if w not in seen:
                    seen.add(w)
                    lines.append(f"  • {w}")

        lines.append("")
        lines.append("━" * 50)
        lines.append("📋 免责声明")
        lines.append("━" * 50)
        lines.append("本报告由量化算法 + AI 分析自动生成，仅供参考，不构成投资建议。")
        lines.append("股市有风险，投资需谨慎。历史表现不代表未来收益。")
        lines.append("")
        lines.append("╚══════════════════════════════════════════════════════════╝")

        return "\n".join(lines)