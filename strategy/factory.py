"""策略工厂。

根据名称获取策略实例。
"""

from .base import BaseStrategy
from .momentum import MomentumStrategy
from .value import ValueStrategy
from .trend import TrendFollowingStrategy


class StrategyFactory:
    """策略工厂。"""

    _strategies = {
        "momentum": MomentumStrategy,
        "value": ValueStrategy,
        "trend": TrendFollowingStrategy,
    }

    @classmethod
    def get(cls, name: str) -> BaseStrategy:
        """获取策略实例。"""
        strategy_class = cls._strategies.get(name.lower())
        if strategy_class is None:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"未知策略 '{name}'，可用策略: {available}")
        return strategy_class()

    @classmethod
    def list_strategies(cls) -> dict:
        """列出所有可用策略及其描述。"""
        return {k: v.description for k, v in cls._strategies.items()}