"""Agent 配置管理。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """交易Agent配置。"""

    # 策略配置
    strategy: str = "trend"  # momentum / value / trend

    # 风控配置
    max_positions: int = 5  # 最大持仓数
    max_position_pct: float = 0.20  # 单只最大仓位 20%
    stop_loss_pct: float = -8.0  # 止损线 -8%
    take_profit_pct: float = 20.0  # 止盈线 +20%

    # 扫描配置
    scan_top_n: int = 50  # 扫描技术分前 N 只

    # LLM 配置（可选，用于 AI 辅助决策）
    llm_enabled: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    # 数据配置
    cache_days: int = 365
    realtime_enabled: bool = True