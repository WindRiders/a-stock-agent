# A-Stock Agent 🐂📈

A股量化交易 Agent —— 基于 Python 的全自动股票分析、评分与交易信号系统。

## 功能特性

- 🔍 **全市场扫描** — 自动扫描沪深全部 A 股，按多维度评分排序
- 📊 **技术分析** — MA/MACD/RSI/KDJ/布林带/成交量 6 大体系
- 💰 **基本面评估** — PE/PB/ROE/利润率/负债率/成长性
- 📰 **消息面分析** — 个股新闻情感分析，涨停板监测
- 🎯 **多策略支持** — 动量策略 / 价值策略 / 趋势跟踪策略
- 📈 **回测引擎** — T+1 规则、涨跌停限制、手续费/印花税模拟
- 🖥️ **CLI 交互** — Rich 美化的命令行界面

## 架构

```
a-stock-agent/
├── data/           # 数据获取层（akshare + baostock）
│   ├── market.py      行情数据（实时/历史K线/指数/龙虎榜）
│   ├── fundamental.py 基本面数据（财报/估值/机构持仓）
│   └── news.py        新闻舆情（个股新闻/涨停板/情感分析）
├── analysis/       # 分析层
│   ├── technical.py   技术分析（MA/MACD/RSI/KDJ/BOLL/量）
│   ├── fundamental.py 基本面评估（估值/质量/成长）
│   └── scoring.py     综合评分引擎（40%技术+25%基本+15%消息+10%资金）
├── strategy/       # 策略层
│   ├── base.py        策略基类
│   ├── momentum.py    动量策略
│   ├── value.py       价值策略
│   └── trend.py       趋势跟踪
├── backtest/       # 回测层
│   ├── engine.py      T+1 回测引擎
│   └── metrics.py     绩效指标（夏普/最大回撤/Calmar）
├── agent/          # Agent 核心
│   ├── core.py        交易Agent主逻辑
│   └── config.py      配置管理
└── cli.py          # 命令行入口
```

## 快速开始

### 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用

```bash
# 查看帮助
python cli.py --help

# 全市场扫描（Top 20）
python cli.py scan -n 20

# 分析单只股票
python cli.py analyze 000001

# 生成投资报告
python cli.py report -n 20 -s trend

# 查看交易信号
python cli.py signals -n 20 -s momentum

# 查看可用策略
python cli.py strategy list

# 切换策略
python cli.py strategy set value
```

### Python API

```python
from agent import TradingAgent, AgentConfig

# 创建 Agent
config = AgentConfig(strategy="trend", max_positions=5)
agent = TradingAgent(config)

# 全市场扫描
scores = agent.scan_market(top_n=20)

# 单股分析
detail = agent.detail_report("000001")

# 生成报告
report = agent.generate_report()
print(report)
```

## 评分体系

| 维度 | 指标 | 权重 |
|------|------|------|
| 技术面 | MA趋势/MACD/RSI/KDJ/布林带/成交量 | 50% |
| 基本面 | PE/PB估值 + ROE质量 + 成长性 | 25% |
| 消息面 | 新闻情感分析 | 15% |
| 资金面 | 成交量分析 | 10% |

## 回测规则

- **T+1 交易** — 当日买入次日可卖
- **涨停不可买，跌停不可卖**
- **佣金** — 万2.5（最低5元）
- **印花税** — 千1（仅卖出）
- **滑点** — 万1
- **仓位管理** — 单只股票不超过20%

## 数据源

- [akshare](https://github.com/akfamily/akshare) — 主力数据源
- [baostock](http://baostock.com/) — 备用数据源

## 免责声明

本工具仅供学习和研究使用，不构成任何投资建议。
股市有风险，投资需谨慎。历史回测表现不代表未来收益。

## License

MIT