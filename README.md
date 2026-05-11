# A股量化交易 Agent

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen)](tests/)

命令行 A 股量化分析工具，集成技术分析、基本面评估、策略信号生成和回测引擎。

```
╔══════════════════════════════════════════════════════╗
║           A 股 投 资 分 析 报 告                     ║
║  生成时间: 2025-05-12 10:30                          ║
║  策略: 趋势跟踪：均线多头+量价配合的趋势策略         ║
╠══════════════════════════════════════════════════════╣
...
```

## 功能

- **市场扫描** — 全市场技术面筛选 + 多维度评分排序
- **个股分析** — 深度技术分析 + 基本面评估 + 消息面情感分析
- **策略信号** — 3 种内置策略（趋势跟踪 / 动量 / 价值）
- **回测引擎** — 完整 A 股交易模拟（T+1、涨跌停、佣金、印花税）
- **投资报告** — 一键生成带评级的分析报告

## 技术栈

| 层级 | 模块 | 技术 |
|------|------|------|
| 数据 | 行情 / 基本面 / 新闻 | akshare + baostock 双源 |
| 分析 | 技术面 | MA / MACD / RSI / KDJ / BOLL / 成交量 |
| 分析 | 基本面 | PE / PB / ROE / EPS / 资产负债率 |
| 分析 | 消息面 | 中文情感分析（关键词匹配） |
| 评分 | 多维度融合 | 技术50% + 基本25% + 消息15% + 资金10% |
| 策略 | 3 种策略 | 趋势跟踪 / 动量 / 价值 |
| 回测 | A 股交易模拟 | T+1 / 涨跌停 / 佣金万2.5 / 印花税千1 |

## 安装

```bash
pip install git+https://github.com/WindRiders/a-stock-agent.git
```

或者本地开发：

```bash
git clone https://github.com/WindRiders/a-stock-agent.git
cd a-stock-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 使用

```bash
# 全市场扫描（取前20只）
a-stock-agent scan --top 20

# 分析单只股票
a-stock-agent analyze 000001

# 生成交易信号
a-stock-agent signals --top 20 --strategy momentum

# 生成投资报告
a-stock-agent report --top 20

# 查看可用策略
a-stock-agent strategy list

# 切换策略
a-stock-agent strategy set value
```

或直接运行 Python：

```bash
python cli.py scan --top 20
python cli.py analyze 000001
```

## 评分体系

| 维度 | 权重 | 范围 | 说明 |
|------|------|------|------|
| 技术面 | 50% | -11 ~ +11 | 趋势 + MACD + RSI + KDJ + BOLL + 成交量 |
| 基本面 | 25% | -6 ~ +6 | 估值 + 盈利能力 + 成长性 |
| 消息面 | 15% | -2 ~ +2 | 新闻舆情情感分析 |
| 资金面 | 10% | -2 ~ +2 | 成交量变化、换手率 |

评级：A (≥0.7) / B (≥0.4) / C (≥0.1) / D (<0.1)

## 策略

| 策略 | 名称 | 核心逻辑 |
|------|------|----------|
| `trend` | 趋势跟踪 | 综合评分 ≥ 0.55 买入 |
| `momentum` | 动量 | 技术面强势 + 放量买入 |
| `value` | 价值 | 低估值 + 高 ROE 买入 |

## 回测

模拟真实 A 股交易环境：

- T+1 交易制度（当日买入次日可卖）
- 涨停无法买入，跌停无法卖出
- 佣金：万分之 2.5（最低 5 元）
- 印花税：千分之 1（仅卖出收取）
- 滑点：万分之一

绩效指标：总收益率、年化收益率、最大回撤、夏普比率、胜率

## 项目结构

```
a-stock-agent/
├── agent/              # 交易 Agent 核心
│   ├── core.py         # TradingAgent 主类
│   └── config.py       # 配置管理
├── data/               # 数据层
│   ├── market.py       # 行情数据（akshare + baostock）
│   ├── fundamental.py  # 基本面数据
│   └── news.py         # 新闻舆情
├── analysis/           # 分析层
│   ├── technical.py    # 技术分析（MA/MACD/RSI/KDJ/BOLL）
│   ├── fundamental.py  # 基本面评估
│   └── scoring.py      # 综合评分引擎
├── strategy/           # 策略层
│   ├── base.py         # 策略基类
│   ├── momentum.py     # 动量策略
│   ├── value.py        # 价值策略
│   ├── trend.py        # 趋势跟踪策略
│   └── factory.py      # 策略工厂
├── backtest/           # 回测引擎
│   ├── engine.py       # A 股回测引擎
│   └── metrics.py      # 绩效指标
├── tests/              # 测试
├── cli.py              # 命令行入口
└── pyproject.toml      # 项目配置
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 安装开发依赖
pip install -e ".[dev]"
```

## 风险提示

本工具仅供学习和研究，不构成投资建议。股市有风险，投资需谨慎。历史表现不代表未来收益。

## License

MIT License