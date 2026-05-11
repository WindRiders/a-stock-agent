"""A股量化交易 Agent。

用法示例:
    from agent import TradingAgent, AgentConfig

    agent = TradingAgent()
    scores = agent.scan_market(top_n=20)
    report = agent.generate_report()
    print(report)
"""

__version__ = "1.0.0"