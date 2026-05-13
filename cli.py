#!/usr/bin/env python3
"""A股量化交易 Agent - 命令行入口。

Usage:
    python cli.py scan                    # 全市场扫描
    python cli.py analyze 000001          # 分析单只股票
    python cli.py report                  # 生成投资报告
    python cli.py ai-report               # AI增强报告
    python cli.py signals                 # 生成交易信号
    python cli.py position 100000         # 仓位建议
    python cli.py risk 100000             # 风控分析
    python cli.py market                  # 市场状态检测
    python cli.py market --auto           # 检测并自动切换策略
    python cli.py backtest                # 历史回测
    python cli.py history                 # 扫描历史
    python cli.py history -s 000001       # 股票评分历史
    python cli.py portfolio list          # 持仓列表
    python cli.py portfolio buy -s 000001 --shares 1000 -p 10.5  # 记录买入
    python cli.py portfolio sell -s 000001 --shares 1000 -p 11.0 # 记录卖出
    python cli.py trades                  # 交易记录
    python cli.py strategy list           # 列出策略
    python cli.py strategy set momentum   # 切换策略
"""

import logging
import sys
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from agent.core import TradingAgent
from agent.config import AgentConfig
from strategy.factory import StrategyFactory

app = typer.Typer(
    name="a-stock-agent",
    help="A股量化交易 Agent - 技术分析 + 基本面评估 + 策略信号 + 回测",
    add_completion=False,
)

console = Console()
agent: Optional[TradingAgent] = None


def get_agent(strategy: str = "trend") -> TradingAgent:
    global agent
    if agent is None or agent.config.strategy != strategy:
        config = AgentConfig(strategy=strategy)
        agent = TradingAgent(config)
    return agent


# ── 扫描 ──────────────────────────────────────────────────

@app.command()
def scan(
    top_n: int = typer.Option(20, "--top", "-n", help="返回前N只股票"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="显示进度"),
):
    """全市场扫描，按评分排序。"""
    a = get_agent(strategy)
    with console.status("[bold green]正在扫描全市场...[/bold green]"):
        scores = a.scan_market(top_n=top_n, verbose=verbose)

    _print_scores_table(scores)


# ── 分析 ──────────────────────────────────────────────────

@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="股票代码，如 000001"),
):
    """深度分析单只股票。"""
    a = get_agent()
    with console.status("[bold green]正在分析...[/bold green]"):
        report = a.detail_report(symbol)

    console.print(report)


# ── 信号 ──────────────────────────────────────────────────

@app.command()
def signals(
    top_n: int = typer.Option(20, "--top", "-n", help="扫描前N只"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
    show_all: bool = typer.Option(False, "--all", help="显示所有信号（含HOLD）"),
):
    """生成交易信号。"""
    a = get_agent(strategy)

    with console.status("[bold green]正在分析并生成信号...[/bold green]"):
        a.scan_market(top_n=top_n, verbose=False)
        all_signals = a.generate_signals()

    if not show_all:
        all_signals = [s for s in all_signals if s.signal.value not in ("HOLD",)]

    _print_signals_table(all_signals)


# ── 报告 ──────────────────────────────────────────────────

@app.command()
def report(
    top_n: int = typer.Option(20, "--top", "-n", help="分析前N只"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
):
    """生成完整投资分析报告。"""
    a = get_agent(strategy)

    with console.status("[bold green]正在生成报告...[/bold green]"):
        a.scan_market(top_n=top_n, verbose=False)
        rep = a.generate_report()

    console.print(rep)


# ── AI 报告 ──────────────────────────────────────────────

@app.command()
def ai_report(
    top_n: int = typer.Option(20, "--top", "-n", help="分析前N只"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
):
    """生成AI增强版投资分析报告。"""
    a = get_agent(strategy)

    with console.status("[bold green]正在扫描并生成AI分析报告...[/bold green]"):
        a.scan_market(top_n=top_n, verbose=False)
        rep = a.generate_ai_report()

    console.print(rep)


# ── 仓位建议 ──────────────────────────────────────────────

@app.command()
def position(
    capital: float = typer.Argument(..., help="总资金（元），如 100000"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
):
    """基于评分生成仓位配置建议。"""
    a = get_agent(strategy)

    with console.status("[bold green]正在分析并计算仓位...[/bold green]"):
        a.scan_market(top_n=30, verbose=False)
        suggestions = a.get_position_suggestions(total_capital=capital)

    if not suggestions:
        console.print("[yellow]暂无符合条件的仓位建议[/yellow]")
        return

    table = Table(title=f"仓位配置建议（总资金: ¥{capital:,.0f}）")
    table.add_column("#", style="dim", width=4)
    table.add_column("代码", style="cyan")
    table.add_column("名称")
    table.add_column("评分", justify="right")
    table.add_column("建议仓位", justify="right")
    table.add_column("建议金额", justify="right")
    table.add_column("止损价", justify="right")
    table.add_column("止盈价", justify="right")
    table.add_column("理由")

    for i, p in enumerate(suggestions, 1):
        amount = capital * p.suggested_pct
        table.add_row(
            str(i),
            p.symbol,
            p.name[:8] if p.name else "-",
            f"{p.score:.2f}",
            f"{p.suggested_pct*100:.1f}%",
            f"¥{amount:,.0f}",
            f"¥{p.stop_loss_price:.2f}" if p.stop_loss_price else "-",
            f"¥{p.take_profit_price:.2f}" if p.take_profit_price else "-",
            p.reason,
        )

    console.print(table)


# ── 风控分析 ──────────────────────────────────────────────

@app.command()
def risk(
    capital: float = typer.Argument(100000, help="总资金（元），如 100000"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
):
    """生成风控分析报告。"""
    a = get_agent(strategy)

    with console.status("[bold green]正在分析风险...[/bold green]"):
        a.scan_market(top_n=30, verbose=False)
        rep = a.generate_risk_report(total_capital=capital)

    console.print(rep)


# ── 市场状态 ──────────────────────────────────────────────

@app.command()
def market(
    index: str = typer.Option("000300", "--index", "-i", help="指数代码（默认沪深300）"),
    auto: bool = typer.Option(False, "--auto", help="根据市场状态自动切换策略"),
):
    """检测市场状态并推荐策略。"""
    a = get_agent()
    a.config.auto_strategy = auto

    with console.status("[bold green]正在检测市场状态...[/bold green]"):
        state = a.detect_market_state(index_code=index)

    # 状态面板
    regime_colors = {
        "bull_trend": "bold green", "bear_trend": "bold red",
        "sideways": "yellow", "high_volatility": "bold yellow",
        "panic": "bold red", "recovery": "green",
    }
    risk_colors = {"high": "bold red", "medium": "yellow", "low": "green"}

    console.print(
        Panel.fit(
            f"[{regime_colors.get(state.regime.value, 'white')}]"
            f"当前市场状态: {state.regime_cn}[/]\n"
            f"趋势方向: {state.trend_direction} | "
            f"波动: {state.volatility_regime} | "
            f"30日最大回撤: {state.max_drawdown_30d*100:.1f}%\n\n"
            f"[bold]推荐策略: {state.recommended_strategy}[/bold] "
            f"(置信度 {state.strategy_confidence:.0%})\n"
            f"理由: {state.strategy_reason}\n\n"
            f"风险等级: [{risk_colors.get(state.risk_level, 'white')}]{state.risk_level}[/]",
            title="市场状态诊断",
            border_style="cyan",
        )
    )

    if state.warnings:
        for w in state.warnings:
            console.print(f"[yellow]⚠️ {w}[/yellow]")


# ── 回测 ──────────────────────────────────────────────────

@app.command()
def backtest(
    symbol: str = typer.Option("000001", "--symbol", "-s", help="回测单只股票"),
    capital: float = typer.Option(100000, "--capital", "-c", help="初始资金"),
    strategy: str = typer.Option("trend", "--strategy", help="策略名称"),
    days: int = typer.Option(252, "--days", "-d", help="回测天数"),
):
    """单股历史回测，验证策略在历史数据上的表现。"""
    a = get_agent(strategy)

    with console.status(f"[bold green]正在回测 {symbol}（{days}个交易日）...[/bold green]"):
        try:
            # 获取历史K线
            from datetime import datetime, timedelta
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            df = a.market_data.get_daily_kline(symbol, start_date=start, end_date=end)

            if df.empty or len(df) < 60:
                console.print(f"[red]数据不足（需要至少60个交易日）[/red]")
                return

            # 对历史每一天做技术分析，生成信号
            from analysis.technical import TechnicalAnalyzer
            ta_analyzer = TechnicalAnalyzer()

            signals = []
            for i in range(60, len(df)):
                window = df.iloc[: i + 1]
                tech = ta_analyzer.analyze(window, symbol)

                # 生成简化信号
                if tech.total_score >= 4:
                    sig = "BUY"
                elif tech.total_score <= -4:
                    sig = "SELL"
                else:
                    sig = "HOLD"

                signals.append({
                    "date": str(window["date"].iloc[-1])[:10],
                    "symbol": symbol,
                    "signal": sig,
                    "score": tech.total_score / 11.0,
                })

            signals_df = pd.DataFrame(signals)

            # 运行回测
            from backtest.engine import BacktestEngine
            engine = BacktestEngine(initial_capital=capital)
            result = engine.run(signals_df, {symbol: df})

        except Exception as e:
            console.print(f"[red]回测失败: {e}[/red]")
            return

    # 结果展示
    return_pct = result.total_return
    color = "green" if return_pct > 0 else "red"

    table = Table(title=f"回测结果 — {symbol} × {strategy} 策略")
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right")

    table.add_row("初始资金", f"¥{result.initial_capital:,.0f}")
    table.add_row("最终权益", f"¥{result.final_equity:,.2f}")
    table.add_row("总收益率", f"[{color}]{return_pct:+.2f}%[/{color}]")
    table.add_row("年化收益率", f"[{color}]{result.annual_return:+.2f}%[/{color}]")
    table.add_row("最大回撤", f"[red]{result.max_drawdown:.2f}%[/red]")
    table.add_row("夏普比率", f"{result.sharpe_ratio:.2f}")
    table.add_row("胜率", f"{result.win_rate:.1f}%")
    table.add_row("交易次数", f"{result.total_trades}")

    console.print(table)

    # 近期交易记录
    if result.trades:
        console.print("\n[bold]最近10笔交易[/bold]")
        trade_table = Table(show_header=True, box=None)
        trade_table.add_column("日期")
        trade_table.add_column("操作")
        trade_table.add_column("价格", justify="right")
        trade_table.add_column("数量", justify="right")
        trade_table.add_column("金额", justify="right")

        for t in result.trades[-10:]:
            action_color = "green" if t.action == "BUY" else "red"
            trade_table.add_row(
                str(t.date)[:10],
                f"[{action_color}]{t.action}[/{action_color}]",
                f"¥{t.price:.2f}",
                f"{t.shares}",
                f"¥{t.amount:,.0f}",
            )
        console.print(trade_table)

    console.print(f"\n[dim]回测说明：基于历史K线滑动窗口技术分析，每次信号使用截至当日数据[/dim]")


# ── 历史 ──────────────────────────────────────────────────

@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="显示条数"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="查询某只股票的历史"),
):
    """查看扫描历史记录。"""
    a = get_agent()

    if symbol:
        records = a.get_stock_history(symbol, limit)
        if not records:
            console.print(f"[yellow]无 {symbol} 的历史记录[/yellow]")
            return

        table = Table(title=f"{symbol} 评分历史")
        table.add_column("时间", style="dim")
        table.add_column("评分", justify="right")
        table.add_column("评级")
        table.add_column("信号")
        table.add_column("价格", justify="right")
        table.add_column("PE", justify="right")

        for r in records:
            signal_styles = {
                "STRONG_BUY": "[green]🔥[/green]", "BUY": "[green]📈[/green]",
                "HOLD": "[dim]⏸️[/dim]", "SELL": "[red]📉[/red]",
                "STRONG_SELL": "[red]💀[/red]",
            }
            price = f"¥{r['latest_price']:.2f}" if r.get("latest_price") else "-"
            pe = f"{r['pe']:.1f}" if r.get("pe") else "-"
            table.add_row(
                str(r["scanned_at"])[:16],
                f"{r['total_score']:.2f}",
                r["rating"] or "-",
                signal_styles.get(r["signal"], r["signal"] or "-"),
                price,
                pe,
            )
        console.print(table)
    else:
        scans = a.get_history(limit)
        if not scans:
            console.print("[yellow]暂无扫描历史[/yellow]")
            return

        table = Table(title="扫描历史")
        table.add_column("#", style="dim", width=4)
        table.add_column("时间", style="dim")
        table.add_column("策略")
        table.add_column("股票数", justify="right")
        table.add_column("市场状态")

        for i, s in enumerate(scans, 1):
            table.add_row(
                str(i),
                str(s["scanned_at"])[:16],
                s["strategy"] or "-",
                str(s["total_stocks"]),
                s.get("market_regime") or "-",
            )
        console.print(table)

        # 数据库统计
        stats = a.get_db_stats()
        if stats["total_scans"] > 0:
            console.print(
                f"\n[dim]数据库: {stats['db_path']} | "
                f"总扫描: {stats['total_scans']} | "
                f"持仓: {stats['active_positions']} | "
                f"交易: {stats['total_trades']}[/dim]"
            )


# ── 持仓 ──────────────────────────────────────────────────

@app.command()
def portfolio(
    action: str = typer.Argument("list", help="list / buy / sell / refresh"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="股票代码"),
    shares: int = typer.Option(None, "--shares", help="股数"),
    price: float = typer.Option(None, "--price", "-p", help="成交价"),
    name: str = typer.Option(None, "--name", help="股票名称"),
    reason: str = typer.Option("", "--reason", "-r", help="交易理由"),
):
    """管理持仓和交易记录。"""
    a = get_agent()

    if action == "list":
        # 先刷新价格
        try:
            a.refresh_prices()
        except Exception:
            pass

        positions = a.get_portfolio()
        summary = a.get_portfolio_summary()

        if not positions:
            console.print("[yellow]暂无活跃持仓[/yellow]")
        else:
            table = Table(title="持仓列表")
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            table.add_column("股数", justify="right")
            table.add_column("成本", justify="right")
            table.add_column("现价", justify="right")
            table.add_column("市值", justify="right")
            table.add_column("盈亏", justify="right")
            table.add_column("收益率", justify="right")

            for p in positions:
                profit_color = "green" if (p["profit"] or 0) >= 0 else "red"
                pnl_color = "green" if (p["profit_pct"] or 0) >= 0 else "red"
                table.add_row(
                    p["symbol"],
                    (p["name"] or "")[:8],
                    str(p["shares"]),
                    f"¥{p['avg_cost']:.2f}",
                    f"¥{p['current_price']:.2f}" if p["current_price"] else "-",
                    f"¥{p['market_value']:,.0f}" if p["market_value"] else "-",
                    f"[{profit_color}]¥{p['profit']:,.0f}[/{profit_color}]" if p["profit"] is not None else "-",
                    f"[{pnl_color}]{p['profit_pct']:+.1f}%[/{pnl_color}]" if p["profit_pct"] is not None else "-",
                )
            console.print(table)

        # 汇总
        console.print(
            f"\n[bold]持仓汇总[/bold]: "
            f"{summary['active_count']} 只 | "
            f"总市值: ¥{summary['total_market_value']:,.0f} | "
            f"浮动盈亏: [{'green' if summary['active_profit'] >= 0 else 'red'}]¥{summary['active_profit']:,.0f}[/]"
        )

    elif action == "buy":
        if not symbol or not shares or not price:
            console.print("[red]用法: portfolio buy --symbol 000001 --shares 1000 --price 10.5 [--name 平安银行] [--reason 原因][/red]")
            return
        a.record_buy(symbol, shares, price, name or "", reason)
        console.print(f"[green]✅ 记录买入: {symbol} {shares}股 @ ¥{price:.2f}[/green]")

    elif action == "sell":
        if not symbol or not shares or not price:
            console.print("[red]用法: portfolio sell --symbol 000001 --shares 1000 --price 11.0 [--reason 原因][/red]")
            return
        a.record_sell(symbol, shares, price, name or "", reason)
        console.print(f"[green]✅ 记录卖出: {symbol} {shares}股 @ ¥{price:.2f}[/green]")

    elif action == "refresh":
        with console.status("[bold green]正在刷新持仓价格...[/bold green]"):
            a.refresh_prices()
        console.print("[green]✅ 持仓价格已刷新[/green]")

    else:
        console.print("[red]用法: portfolio list|buy|sell|refresh[/red]")


# ── 交易记录 ──────────────────────────────────────────────

@app.command()
def trades(
    limit: int = typer.Option(20, "--limit", "-n", help="显示条数"),
):
    """查看交易记录。"""
    a = get_agent()
    records = a.get_trades(limit)

    if not records:
        console.print("[yellow]暂无交易记录[/yellow]")
        return

    table = Table(title="交易记录")
    table.add_column("日期", style="dim")
    table.add_column("代码", style="cyan")
    table.add_column("操作")
    table.add_column("股数", justify="right")
    table.add_column("价格", justify="right")
    table.add_column("金额", justify="right")
    table.add_column("费用", justify="right")
    table.add_column("理由")

    for t in records:
        action_color = "green" if t["action"] == "BUY" else "red"
        fees = (t.get("commission") or 0) + (t.get("stamp_tax") or 0)
        table.add_row(
            str(t["traded_at"])[:16],
            t["symbol"],
            f"[{action_color}]{t['action']}[/{action_color}]",
            str(t["shares"]),
            f"¥{t['price']:.2f}",
            f"¥{t['amount']:,.0f}",
            f"¥{fees:.2f}" if fees else "-",
            (t.get("reason") or "")[:20],
        )
    console.print(table)


# ── 策略管理 ──────────────────────────────────────────────

@app.command()
def strategy(
    action: str = typer.Argument("list", help="list / set / info"),
    name: str = typer.Argument(None, help="策略名称"),
):
    """管理交易策略。"""
    if action == "list":
        _print_strategies()
    elif action == "info" and name:
        _print_strategy_info(name)
    elif action == "set" and name:
        try:
            StrategyFactory.get(name)  # 验证策略存在
            global agent
            agent = None  # 切换策略时重建 agent
            console.print(f"[green]✅ 已切换到策略: {name}[/green]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
    else:
        console.print("[red]用法: strategy list|info <name>|set <name>[/red]")


# ── 输出辅助函数 ──────────────────────────────────────────

def _print_scores_table(scores):
    """打印评分表格。"""
    if not scores:
        console.print("[yellow]暂无有效评分数据[/yellow]")
        return

    table = Table(title="A股综合评分排名", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("代码", style="cyan")
    table.add_column("名称")
    table.add_column("总分", justify="right")
    table.add_column("评级")
    table.add_column("信号")
    table.add_column("技术分", justify="right")
    table.add_column("基本分", justify="right")
    table.add_column("最新价", justify="right")
    table.add_column("PE", justify="right")

    signal_styles = {
        "STRONG_BUY": "[bold green]🔥 强烈买入[/bold green]",
        "BUY": "[green]📈 买入[/green]",
        "HOLD": "[dim]⏸️ 持有[/dim]",
        "SELL": "[red]📉 卖出[/red]",
        "STRONG_SELL": "[bold red]💀 强烈卖出[/bold red]",
    }
    rating_styles = {
        "A": "[bold green]A[/bold green]",
        "B": "[green]B[/green]",
        "C": "[yellow]C[/yellow]",
        "D": "[red]D[/red]",
    }

    for i, s in enumerate(scores, 1):
        signal_str = signal_styles.get(s.signal, s.signal)
        rating_str = rating_styles.get(s.rating, s.rating)
        price_str = f"¥{s.latest_price:.2f}" if s.latest_price else "-"
        pe_str = f"{s.pe:.1f}" if s.pe and s.pe > 0 else "-"

        table.add_row(
            str(i),
            s.symbol,
            s.name[:8] if s.name else "-",
            f"{s.total_score:.2f}",
            rating_str,
            signal_str,
            f"{s.tech_score:+d}",
            f"{s.fund_score:+d}",
            price_str,
            pe_str,
        )

    console.print(table)


def _print_signals_table(signals):
    """打印信号表格。"""
    if not signals:
        console.print("[yellow]暂无交易信号[/yellow]")
        return

    table = Table(title="交易信号")
    table.add_column("#", style="dim", width=4)
    table.add_column("代码", style="cyan")
    table.add_column("名称")
    table.add_column("信号")
    table.add_column("评分", justify="right")
    table.add_column("置信度", justify="right")
    table.add_column("理由")

    signal_styles = {
        "STRONG_BUY": "[bold green]🔥 强烈买入[/bold green]",
        "BUY": "[green]📈 买入[/green]",
        "HOLD": "[dim]⏸️ 持有[/dim]",
        "SELL": "[red]📉 卖出[/red]",
        "STRONG_SELL": "[bold red]💀 强烈卖出[/bold red]",
    }

    for i, sig in enumerate(signals, 1):
        table.add_row(
            str(i),
            sig.symbol,
            sig.name[:8] if sig.name else "-",
            signal_styles.get(sig.signal.value, sig.signal.value),
            f"{sig.score:.2f}",
            f"{sig.confidence:.0%}",
            sig.reason,
        )

    console.print(table)


def _print_strategies():
    """打印策略列表。"""
    strategies = StrategyFactory.list_strategies()
    table = Table(title="可用交易策略")
    table.add_column("名称", style="cyan")
    table.add_column("描述")

    for name, desc in strategies.items():
        table.add_row(name, desc)

    console.print(table)


def _print_strategy_info(name: str):
    """打印策略详情。"""
    try:
        s = StrategyFactory.get(name)
        console.print(f"\n[bold cyan]{s.name}[/bold cyan]")
        console.print(f"  {s.description}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


# ── main ───────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()