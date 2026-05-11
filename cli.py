#!/usr/bin/env python3
"""A股量化交易 Agent - 命令行入口。

Usage:
    python cli.py scan                    # 全市场扫描
    python cli.py analyze 000001          # 分析单只股票
    python cli.py report                  # 生成投资报告
    python cli.py signals                 # 生成交易信号
    python cli.py backtest                # 运行回测
    python cli.py strategy list           # 列出策略
    python cli.py strategy set momentum   # 切换策略
"""

import logging
import sys
from typing import Optional

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