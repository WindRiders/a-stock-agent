"""Agent 配置管理。"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """交易Agent配置。"""

    # 策略配置
    strategy: str = "trend"  # momentum / value / trend / grid / mean_reversion

    # 风控配置
    max_positions: int = 5  # 最大持仓数
    max_position_pct: float = 0.20  # 单只最大仓位 20%
    stop_loss_pct: float = -8.0  # 止损线 -8%
    take_profit_pct: float = 20.0  # 止盈线 +20%

    # 扫描配置
    scan_top_n: int = 50  # 扫描技术分前 N 只

    # LLM 配置（用于 AI 辅助决策）
    llm_enabled: bool = False
    llm_provider: Optional[str] = None  # 如 "openrouter", "openai", "deepseek"
    llm_model: Optional[str] = None  # 如 "deepseek/deepseek-chat"
    llm_base_url: Optional[str] = None  # API endpoint
    llm_api_key: Optional[str] = None  # API key（可留空从环境变量读取）

    # 数据配置
    cache_days: int = 365
    realtime_enabled: bool = True

    def get_llm_config(self) -> dict:
        """获取LLM配置，自动从环境变量补充缺失值。"""
        api_key = self.llm_api_key
        if not api_key:
            # 尝试常见环境变量
            for env_var in [
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY", 
                "DEEPSEEK_API_KEY",
                "LLM_API_KEY",
            ]:
                api_key = os.environ.get(env_var)
                if api_key:
                    break

        base_url = self.llm_base_url
        if not base_url:
            base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": self.llm_model or "gpt-4o-mini",
            "enabled": self.llm_enabled and bool(api_key),
        }