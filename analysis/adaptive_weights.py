"""自适应评分权重模块。

根据当前市场状态动态调整各维度权重：
- 牛市趋势 → 技术面权重提升（强者恒强）
- 熊市趋势 → 基本面权重提升（防守为主）
- 震荡市   → 资金面权重提升（量价关系重要）
- 恐慌     → 消息面权重提升（情绪驱动）
"""

from typing import Dict

from agent.market_state import MarketRegime


# 默认权重（基准）
DEFAULT_WEIGHTS = {
    "technical": 0.50,    # 技术面
    "fundamental": 0.25,  # 基本面
    "sentiment": 0.15,    # 消息面
    "capital": 0.10,      # 资金面
}

# 各状态下的权重调整
REGIME_WEIGHTS = {
    MarketRegime.BULL_TREND: {
        "technical": 0.55,
        "fundamental": 0.20,
        "sentiment": 0.10,
        "capital": 0.15,
        "reason": "牛市趋势 — 技术面权重 +5%，更关注趋势延续"
    },
    MarketRegime.BEAR_TREND: {
        "technical": 0.40,
        "fundamental": 0.35,
        "sentiment": 0.10,
        "capital": 0.15,
        "reason": "熊市趋势 — 基本面权重 +10%，防御为主"
    },
    MarketRegime.SIDEWAYS: {
        "technical": 0.45,
        "fundamental": 0.25,
        "sentiment": 0.10,
        "capital": 0.20,
        "reason": "横盘震荡 — 资金面权重 +10%，关注量价突破"
    },
    MarketRegime.HIGH_VOL: {
        "technical": 0.30,
        "fundamental": 0.30,
        "sentiment": 0.25,
        "capital": 0.15,
        "reason": "高波动 — 消息面权重 +10%，情绪驱动明显"
    },
    MarketRegime.PANIC: {
        "technical": 0.20,
        "fundamental": 0.35,
        "sentiment": 0.30,
        "capital": 0.15,
        "reason": "恐慌暴跌 — 基本面+消息面优先，等待情绪稳定"
    },
    MarketRegime.RECOVERY: {
        "technical": 0.45,
        "fundamental": 0.20,
        "sentiment": 0.15,
        "capital": 0.20,
        "reason": "超跌反弹 — 技术面+资金面并重，捕捉修复机会"
    },
}


class AdaptiveWeights:
    """自适应权重管理器。

    根据市场状态动态调整评分权重。
    """

    def __init__(self):
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.current_reason = "默认权重"

    def adjust(self, regime: MarketRegime) -> Dict[str, float]:
        """根据市场状态调整权重。

        Returns:
            调整后的权重字典 + reason
        """
        if regime in REGIME_WEIGHTS:
            rw = REGIME_WEIGHTS[regime]
            self.current_weights = {
                "technical": rw["technical"],
                "fundamental": rw["fundamental"],
                "sentiment": rw["sentiment"],
                "capital": rw["capital"],
            }
            self.current_reason = rw["reason"]
        else:
            self.current_weights = DEFAULT_WEIGHTS.copy()
            self.current_reason = "默认权重（未知市场状态）"

        return self.get_weights()

    def get_weights(self) -> Dict:
        """获取当前权重。"""
        return {
            **self.current_weights,
            "reason": self.current_reason,
        }

    def apply_weights(self, scores: list) -> list:
        """对评分列表应用自适应权重，重新计算总分。

        注意：这是一个近似调整，因为评分模块内部已用过默认权重。
        这里对已有评分做加权修正。
        """
        w = self.current_weights
        default_w = DEFAULT_WEIGHTS

        for s in scores:
            # 用新权重重新加权
            tech_part = getattr(s, 'tech_score', 0) / 11.0 * w["technical"]
            fund_part = getattr(s, 'fund_score', 0) / 6.0 * w["fundamental"]
            sent_part = getattr(s, 'sentiment_score', 0) / 2.0 * w["sentiment"]
            cap_part = getattr(s, 'capital_score', 0) / 2.0 * w["capital"]

            # 归一化（权重之和总是1.0）
            s.total_score = round(tech_part + fund_part + sent_part + cap_part, 2)

            # 重新评级
            if s.total_score >= 0.7:
                s.rating = "A"
            elif s.total_score >= 0.4:
                s.rating = "B"
            elif s.total_score >= 0.1:
                s.rating = "C"
            else:
                s.rating = "D"

        return scores