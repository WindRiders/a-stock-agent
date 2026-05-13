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
[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)](tests/)
[![CI](https://github.com/WindRiders/a-stock-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/WindRiders/a-stock-agent/actions)

命令行 A 股量化分析工具，集成技术分析、基本面评估、策略信号生成、
回测引擎、**风险管理**、**市场状态检测**和 **AI 智能解读**。

## 功能

- **市场扫描** — 全市场技术面筛选 + 多维度评分排序
- **个股分析** — 深度技术分析 + 基本面评估 + 消息面情感分析
- **策略信号** — 5 种内置策略（趋势跟踪 / 动量 / 价值 / 网格 / 均值回归）
- **回测引擎** — 完整 A 股交易模拟 + 参数优化 + Walk-Forward 分析
- **市场状态检测** — 7 种市场状态自动识别 + 策略自动推荐
- **风险管理** — 仓位计算、组合风险度（VaR/CVaR/Beta/夏普）、止损止盈
- **LLM 智能分析** — 通用 OpenAI 兼容 API，支持真正的 AI 深度解读
- **投资报告** — 一键生成量化+AI增强分析报告

## 技术栈

| 层级 | 模块 | 技术 |
|------|------|------|
| 数据 | 行情 / 基本面 / 新闻 | akshare + baostock 双源 |
| 分析 | 技术面 | MA / MACD / RSI / KDJ / BOLL / 成交量 |
| 分析 | 基本面 | PE / PB / ROE / EPS / 资产负债率 |
| 分析 | 消息面 | 中文情感分析 + LLM 深度解读 |
| 评分 | 多维度融合 | 技术50% + 基本25% + 消息15% + 资金10% |
| 策略 | 5 种策略 | 趋势跟踪 / 动量 / 价值 / 网格 / 均值回归 |
| 回测 | A 股交易模拟 | T+1 / 涨跌停 / 佣金万2.5 / 印花税千1 |
| 回测 | 参数优化 | 网格搜索 + Walk-Forward 分析 |
| 风控 | 仓位 + 风险 | 仓位计算 / VaR / CVaR / Beta / 止损止盈 |
| 报告 | AI 增强 | 自然语言市场解读 + 个股点评 |

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

# AI 增强报告
a-stock-agent ai-report --top 20

# 仓位建议（假设10万资金）
a-stock-agent position 100000

# 风控分析
a-stock-agent risk 100000

# 市场状态检测 + 策略推荐
a-stock-agent market

# 自动切换为推荐策略
a-stock-agent market --auto

# 单股历史回测
a-stock-agent backtest --symbol 000001 --capital 100000

# 查看可用策略
a-stock-agent strategy list

# 切换策略
a-stock-agent strategy set grid
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

| 策略 | 名称 | 核心逻辑 | 适用环境 |
|------|------|----------|----------|
| `trend` | 趋势跟踪 | 综合评分 ≥ 0.55 买入 | 趋势行情 |
| `momentum` | 动量 | 技术面强势 + 放量买入 | 强势市场 |
| `value` | 价值 | 低估值 + 高 ROE 买入 | 价值发现 |
| `grid` | 网格交易 | 震荡区间内低买高卖 | 横盘震荡 |
| `mean_reversion` | 均值回归 | 超卖反弹 + 超买回调 | 急跌反弹 |

## 回测

模拟真实 A 股交易环境：

- T+1 交易制度（当日买入次日可卖）
- 涨停无法买入，跌停无法卖出
- 佣金：万分之 2.5（最低 5 元）
- 印花税：千分之 1（仅卖出收取）
- 滑点：万分之一

**增强回测功能：**
- 多资产组合回测
- 参数网格搜索优化
- Walk-Forward 样本外验证

绩效指标：总收益率、年化收益率、最大回撤、夏普比率、胜率、Calmar 比率、Sortino 比率

## 风险管理

- 仓位计算：基于评分权重 + 风控限制的仓位分配
- 风险指标：VaR (95%) / CVaR / Beta / 年化波动率 / 最大回撤
- 止损止盈：固定止损 -8%、固定止盈 +25%、移动止损 5%
- 分散化评分：基于持仓相关性的分散化度量

## 项目结构

```
a-stock-agent/
├── agent/              # 交易 Agent 核心
│   ├── core.py         # TradingAgent 主类
│   ├── config.py       # 配置管理
│   └── llm.py          # AI 智能分析（市场解读 + 个股点评）
├── data/               # 数据层
│   ├── market.py       # 行情数据（akshare + baostock）
│   ├── fundamental.py  # 基本面数据
│   └── news.py         # 新闻舆情 + 情感分析
├── analysis/           # 分析层
│   ├── technical.py    # 技术分析（MA/MACD/RSI/KDJ/BOLL）
│   ├── fundamental.py  # 基本面评估
│   └── scoring.py      # 综合评分引擎
├── strategy/           # 策略层（5种策略）
│   ├── base.py         # 策略基类
│   ├── momentum.py     # 动量策略
│   ├── value.py        # 价值策略
│   ├── trend.py        # 趋势跟踪策略
│   ├── grid.py         # 网格交易策略
│   ├── mean_reversion.py # 均值回归策略
│   └── factory.py      # 策略工厂
├── backtest/           # 回测引擎
│   ├── engine.py       # A 股回测引擎
│   ├── metrics.py      # 绩效指标
│   └── enhanced.py     # 增强回测（组合/优化/Walk-Forward）
├── risk/               # 风险管理
│   └── __init__.py     # 仓位管理 + 组合风险 + 止损止盈
├── tests/              # 76 项测试
├── cli.py              # 命令行入口（typer + rich）
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