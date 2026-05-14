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
import os
import sys
from datetime import datetime
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


# ── 日报 ──────────────────────────────────────────────────

@app.command()
def daily(
    top_n: int = typer.Option(50, "--top", "-n", help="扫描数量"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="策略名称"),
    output: str = typer.Option("", "--output", "-o", help="输出文件路径"),
    silent: bool = typer.Option(False, "--silent", help="静默模式（适合cron）"),
    notify: bool = typer.Option(False, "--notify", help="推送报告到已配置的通知渠道"),
):
    """一键生成每日分析报告。适合定时任务。
    
    自动完成：市场状态检测 → 自适应权重 → 全市场扫描 → 资金面分析 → 报告 → 可选推送
    """
    from analysis.adaptive_weights import AdaptiveWeights
    from analysis.capital_flow import CapitalFlowAnalyzer

    a = get_agent(strategy)

    # 输出目录
    if not output:
        daily_dir = os.path.expanduser("~/.a-stock-agent/daily")
        os.makedirs(daily_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d_%H%M")
        output = os.path.join(daily_dir, f"report_{today}.md")

    status_text = "[bold green]每日分析中...[/bold green]"
    if silent:
        status_text = None

    if status_text:
        with console.status(status_text):
            state = a.detect_market_state()
            weights = AdaptiveWeights().adjust(state.regime)
            scores = a.scan_market(top_n=top_n, verbose=not silent)

            # 记录市场状态到扫描
            if a._current_scan_id:
                a.store._get_conn().execute(
                    "UPDATE scans SET market_regime=?, market_risk=? WHERE id=?",
                    (state.regime_cn, state.risk_level, a._current_scan_id),
                )
                a.store._get_conn().commit()

            # 自适应权重调整评分
            weights_obj = AdaptiveWeights()
            weights_obj.adjust(state.regime)
            scores = weights_obj.apply_weights(scores)

            # 资金面附加
            cf = CapitalFlowAnalyzer()
            flow = cf.analyze()
            for s in scores:
                if flow["total_score"] > 0:
                    s.reasons.append(f"北向资金{flow['north_trend']}")
                elif flow["total_score"] < 0:
                    s.warnings.append(f"北向资金{flow['north_trend']}")

            report_lines = _build_daily_report(scores, state, a.config.strategy, weights_obj)
    else:
        state = a.detect_market_state()
        scores = a.scan_market(top_n=top_n, verbose=False)

        if a._current_scan_id:
            conn = a.store._get_conn()
            conn.execute(
                "UPDATE scans SET market_regime=?, market_risk=? WHERE id=?",
                (state.regime_cn, state.risk_level, a._current_scan_id),
            )
            conn.commit()

        weights_obj = AdaptiveWeights()
        weights_obj.adjust(state.regime)
        scores = weights_obj.apply_weights(scores)

        cf = CapitalFlowAnalyzer()
        flow = cf.analyze()
        for s in scores:
            if flow["total_score"] > 0:
                s.reasons.append(f"北向资金{flow['north_trend']}")
            elif flow["total_score"] < 0:
                s.warnings.append(f"北向资金{flow['north_trend']}")

        report_lines = _build_daily_report(scores, state, a.config.strategy, weights_obj)

    # 写入文件
    report_text = "\n".join(report_lines)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_text)

    buy_count = sum(1 for s in scores if s.signal in ("BUY", "STRONG_BUY"))
    sell_count = sum(1 for s in scores if s.signal in ("SELL", "STRONG_SELL"))

    if not silent:
        console.print(f"\n[green]✅ 每日报告已保存: {output}[/green]")
        console.print(
            f"[dim]状态: {state.regime_cn} | "
            f"策略: {state.recommended_strategy} | "
            f"风险: {state.risk_level} | "
            f"扫描: {len(scores)}只 | "
            f"买入: {buy_count} | 卖出: {sell_count}[/dim]"
        )

    # 静默模式下输出路径到stdout（方便cron脚本捕获）
    if silent:
        print(output)

    # 通知推送
    if notify:
        try:
            from agent.notifier import Notifier
            n = Notifier()
            channels = n.configured_channels
            if channels:
                summary = (
                    f"市场状态: {state.regime_cn}\n"
                    f"风险: {state.risk_level}\n"
                    f"推荐策略: {state.recommended_strategy}\n"
                    f"买入信号: {buy_count} | 卖出: {sell_count}\n"
                    f"完整报告: {output}"
                )
                results = n.broadcast(summary, title="A股每日分析")
                pushed = [ch for ch, ok in results.items() if ok]
                if pushed and not silent:
                    console.print(f"[green]📨 已推送到: {', '.join(pushed)}[/green]")
            elif not silent:
                console.print("[yellow]⚠️ 未配置通知渠道，跳过推送[/yellow]")
                console.print("[dim]设置环境变量: TELEGRAM_BOT_TOKEN / DISCORD_WEBHOOK_URL / WECOM_WEBHOOK_URL[/dim]")
        except Exception as e:
            logger.warning("通知推送失败: %s", e)


def _build_daily_report(scores, state, strategy_name, weights=None):
    """构建每日报告内容（Markdown格式）。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# A股每日分析报告",
        f"",
        f"**生成时间**: {now}  ",
        f"**策略引擎**: {strategy_name}  ",
        f"",
        f"## 市场状态",
        f"",
        f"- **当前状态**: {state.regime_cn}",
        f"- **趋势方向**: {state.trend_direction}",
        f"- **波动水平**: {state.volatility_regime}（年化 {state.volatility*100:.1f}%）",
        f"- **风险等级**: {state.risk_level}",
        f"- **推荐策略**: `{state.recommended_strategy}`（置信度 {state.strategy_confidence:.0%}）",
        f"- **30日最大回撤**: {state.max_drawdown_30d*100:.1f}%",
        f"",
    ]

    if state.warnings:
        lines.append("### 风险提示")
        for w in state.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.append("## 买入信号 TOP 10")
    lines.append("")
    lines.append("| # | 代码 | 名称 | 评分 | 评级 | 信号 | PE | 最新价 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    buy_scores = [s for s in scores if s.signal in ("BUY", "STRONG_BUY")][:10]
    if not buy_scores:
        buy_scores = sorted(scores, key=lambda x: x.total_score, reverse=True)[:10]

    for i, s in enumerate(buy_scores, 1):
        pe_str = f"{s.pe:.1f}" if s.pe and s.pe > 0 else "-"
        price_str = f"¥{s.latest_price:.2f}" if s.latest_price else "-"
        signal_icon = "🔥" if s.signal == "STRONG_BUY" else ("📈" if s.signal == "BUY" else "⏸️")
        lines.append(
            f"| {i} | {s.symbol} | {(s.name or '')[:8]} | {s.total_score:.2f} | {s.rating} | {signal_icon} | {pe_str} | {price_str} |"
        )

    lines.append("")
    lines.append("## 风险汇总")
    lines.append("")
    all_warnings = []
    for s in scores:
        all_warnings.extend(s.warnings)
    seen = set()
    for w in all_warnings[:10]:
        if w not in seen:
            seen.add(w)
            lines.append(f"- • {w}")

    if not [w for w in all_warnings if w]:
        lines.append("暂无重大风险信号。")

    lines.append("")
    lines.append("---")
    lines.append(f"*本报告由 a-stock-agent 自动生成，仅供参考，不构成投资建议。*")

    return lines


# ── 导出 ──────────────────────────────────────────────────

@app.command()
def export(
    scan_id: int = typer.Option(None, "--scan", help="导出指定扫描ID（默认最新）"),
    fmt: str = typer.Option("csv", "--format", "-f", help="csv / json / md"),
    output: str = typer.Option("", "--output", "-o", help="输出文件路径"),
):
    """导出扫描结果为 CSV/JSON/Markdown。"""
    a = get_agent()

    if scan_id is None:
        history = a.get_history(1)
        if not history:
            console.print("[red]无扫描记录[/red]")
            return
        scan_id = history[0]["id"]

    detail = a.store.get_scan_detail(scan_id)
    if not detail:
        console.print(f"[red]扫描 #{scan_id} 无数据[/red]")
        return

    if not output:
        output = os.path.expanduser(f"~/a-stock-export-{scan_id}.{fmt}")

    if fmt == "csv":
        import csv
        with open(output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=detail[0].keys())
            writer.writeheader()
            writer.writerows(detail)

    elif fmt == "json":
        import json
        with open(output, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2, default=str)

    elif fmt == "md":
        lines = ["# 扫描结果导出", "", f"扫描ID: {scan_id}", "", "| 代码 | 名称 | 评分 | 评级 | 信号 | 技术 | 基本 | PE |", "|---|---|---|---|---|---|---|---|---|"]
        for d in detail:
            lines.append(f"| {d['symbol']} | {(d.get('name') or '')[:8]} | {d['total_score']:.2f} | {d['rating']} | {d['signal']} | {d.get('tech_score', 0):+d} | {d.get('fund_score', 0):+d} | {d.get('pe') or '-'} |")
        with open(output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    else:
        console.print(f"[red]不支持的格式: {fmt}（支持 csv/json/md）[/red]")
        return

    console.print(f"[green]✅ 已导出 {len(detail)} 条记录 → {output}[/green]")


# ── 准确率 ──────────────────────────────────────────────

@app.command()
def accuracy():
    """计算历史交易信号准确率。"""
    a = get_agent()

    with console.status("[bold green]正在计算信号准确率...[/bold green]"):
        result = a.calc_accuracy()

    if result["total_signals"] == 0:
        console.print("[yellow]暂无足够的买入信号历史数据[/yellow]")
        console.print("[dim]提示：多运行几次 scan 积累数据后再查看[/dim]")
        return

    # 概览面板
    acc_color = "green" if result["accuracy"] >= 60 else ("yellow" if result["accuracy"] >= 40 else "red")
    pf_color = "green" if result.get("profit_factor", 0) >= 1.5 else "yellow"

    panel = Panel.fit(
        f"信号总数: {result['total_signals']}\n"
        f"正确: [green]{result['correct']}[/green] | 错误: [red]{result['wrong']}[/red]\n"
        f"准确率: [{acc_color}]{result['accuracy']:.1f}%[/{acc_color}]\n"
        f"平均收益: [{'green' if result['avg_return'] > 0 else 'red'}]{result['avg_return']:+.2f}%[/]\n"
        f"盈亏比: [{pf_color}]{result.get('profit_factor', 0):.2f}[/{pf_color}]",
        title="信号准确率分析",
        border_style="cyan",
    )
    console.print(panel)

    # 明细表
    if result.get("details"):
        console.print("\n[bold]近期信号明细（前20条）[/bold]")
        detail_table = Table(show_header=True)
        detail_table.add_column("日期")
        detail_table.add_column("代码", style="cyan")
        detail_table.add_column("入场价", justify="right")
        detail_table.add_column("出场价", justify="right")
        detail_table.add_column("收益", justify="right")
        detail_table.add_column("结果")

        for d in result["details"][:20]:
            pnl_color = "green" if d["pnl_pct"] > 0 else "red"
            detail_table.add_row(
                d["entry_date"],
                d["symbol"],
                f"¥{d['entry_price']:.2f}",
                f"¥{d['exit_price']:.2f}",
                f"[{pnl_color}]{d['pnl_pct']:+.2f}%[/{pnl_color}]",
                "✅" if d["correct"] else "❌",
            )
        console.print(detail_table)


# ── 通知测试 ──────────────────────────────────────────────

@app.command()
def notify():
    """测试通知推送配置。"""
    from agent.notifier import Notifier
    n = Notifier()
    channels = n.configured_channels

    if not channels:
        console.print("[yellow]未配置任何通知渠道[/yellow]")
        console.print("设置以下环境变量之一：")
        console.print("  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
        console.print("  DISCORD_WEBHOOK_URL")
        console.print("  WECOM_WEBHOOK_URL")
        return

    console.print(f"[green]已配置渠道: {', '.join(channels)}[/green]")
    results = n.broadcast("🧪 这是来自 a-stock-agent 的测试消息。通知系统配置正常。", title="测试通知")

    for ch, ok in results.items():
        if ok:
            console.print(f"  ✅ {ch}")
        else:
            console.print(f"  ❌ {ch}")


# ── 多周期 ──────────────────────────────────────────────

@app.command()
def multi(
    symbol: str = typer.Argument(..., help="股票代码"),
):
    """多周期技术分析（日线+周线+月线共振）。"""
    from analysis.multi_timeframe import MultiTimeframeAnalyzer

    a = get_agent()
    mta = MultiTimeframeAnalyzer()

    with console.status(f"[bold green]正在多周期分析 {symbol}...[/bold green]"):
        try:
            df = a.market_data.get_daily_kline(symbol, start_date="20200101")
            result = mta.analyze(df, symbol)
        except Exception as e:
            console.print(f"[red]分析失败: {e}[/red]")
            return

    console.print(mta.generate_report(result))


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


# ── 配置管理 ──────────────────────────────────────────────

@app.command()
def config(
    action: str = typer.Argument("show", help="show / save / reset / backups / restore"),
    strategy: str = typer.Option(None, "--strategy", "-s", help="设置策略"),
    max_positions: int = typer.Option(None, "--max-positions", help="最大持仓数"),
    stop_loss: float = typer.Option(None, "--stop-loss", help="止损比例 %，如 -8"),
    take_profit: float = typer.Option(None, "--take-profit", help="止盈比例 %，如 20"),
    llm_enabled: bool = typer.Option(None, "--llm/--no-llm", help="启用/禁用 LLM"),
    llm_model: str = typer.Option(None, "--llm-model", help="LLM 模型名"),
    backup_name: str = typer.Option(None, "--backup", help="备份文件名（恢复用）"),
):
    """管理配置文件 ~/.a-stock-agent/config.yaml"""
    from agent.config_file import ConfigManager

    mgr = ConfigManager()

    if action == "show":
        if mgr.exists():
            console.print(f"[dim]配置文件: {mgr.get_path()}[/dim]\n")
            console.print(mgr.show())
        else:
            console.print("[yellow]配置文件不存在[/yellow]")
            console.print("运行 'config save' 创建默认配置")

    elif action == "save":
        cfg = mgr.load()
        if strategy:
            cfg.strategy = strategy
        if max_positions is not None:
            cfg.max_positions = max_positions
        if stop_loss is not None:
            cfg.stop_loss_pct = stop_loss
        if take_profit is not None:
            cfg.take_profit_pct = take_profit
        if llm_enabled is not None:
            cfg.llm_enabled = llm_enabled
        if llm_model:
            cfg.llm_model = llm_model
        mgr.save(cfg)
        console.print(f"[green]✅ 配置已保存: {mgr.get_path()}[/green]")

    elif action == "reset":
        mgr.reset()
        console.print("[green]✅ 配置已重置为默认值[/green]")

    elif action == "backups":
        backups = mgr.list_backups()
        if not backups:
            console.print("[yellow]暂无备份[/yellow]")
            return
        for b in backups:
            console.print(f"  {b['filename']}  ({b['size']} bytes)")

    elif action == "restore":
        if not backup_name:
            console.print("[red]请指定 --backup 文件名[/red]")
            return
        try:
            cfg = mgr.restore(backup_name)
            console.print(f"[green]✅ 已从备份恢复: {backup_name}[/green]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")

    else:
        console.print("[red]用法: config show|save|reset|backups|restore[/red]")


# ── 模拟盘 ──────────────────────────────────────────────

@app.command()
def paper(
    action: str = typer.Argument("summary", help="summary / buy / sell / reset / snapshot"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="股票代码"),
    price: float = typer.Option(None, "--price", "-p", help="成交价"),
    shares: int = typer.Option(None, "--shares", help="股数"),
    amount: float = typer.Option(None, "--amount", "-a", help="买入金额"),
    reason: str = typer.Option("", "--reason", "-r", help="交易理由"),
    initial: float = typer.Option(100000, "--initial", help="初始资金"),
):
    """模拟盘/纸面交易。数据持久化到 ~/.a-stock-agent/paper.db"""
    from paper_trading import PaperTrader

    db = os.path.expanduser("~/.a-stock-agent/paper.db")
    trader = PaperTrader(initial_capital=initial, db_path=db)

    if action == "summary":
        s = trader.summary()
        panel = Panel.fit(
            f"初始资金: ¥{s.initial_capital:,.0f}\n"
            f"当前权益: ¥{s.equity:,.0f}\n"
            f"总收益: [{'green' if s.total_return >= 0 else 'red'}]{s.total_return_pct:+.2f}%[/]\n"
            f"现金: ¥{s.cash:,.0f}\n"
            f"交易次数: {s.total_trades}\n"
            f"胜率: {s.win_rate:.1f}%\n"
            f"夏普: {s.sharpe_ratio:.2f}\n"
            f"最大回撤: {s.max_drawdown:.2f}%\n"
            f"持仓数: {s.active_positions}",
            title="模拟盘汇总",
            border_style="cyan",
        )
        console.print(panel)

        positions = trader.positions_list()
        if positions:
            console.print("\n[bold]当前持仓[/bold]")
            for p in positions:
                color = "green" if p["unrealized_pnl"] >= 0 else "red"
                console.print(
                    f"  {p['symbol']} {p['name']:<6s} "
                    f"{p['shares']}股 "
                    f"成本¥{p['avg_cost']:.2f} → 现价¥{p['current_price']:.2f} "
                    f"[{color}]{p['unrealized_pnl']:+,.0f} ({p['unrealized_pnl_pct']:+.1f}%)[/]"
                )

    elif action == "buy":
        if not symbol or not price:
            console.print("[red]用法: paper buy --symbol 000001 --price 10.5 --shares 1000 [--reason 原因][/red]")
            return
        trade = trader.execute(symbol, "BUY", price, shares or 0, reason=reason, amount=amount or 0)
        if trade:
            trader.take_snapshot()
            console.print(f"[green]✅ 模拟买入: {symbol} {trade.shares}股 @ ¥{price:.2f}[/green]")
        else:
            console.print("[red]买入失败（资金不足/股数不足）[/red]")

    elif action == "sell":
        if not symbol or not price:
            console.print("[red]用法: paper sell --symbol 000001 --price 11.0 [--shares 1000] [--reason 原因][/red]")
            return
        trade = trader.execute(symbol, "SELL", price, shares or 0, reason=reason)
        if trade:
            trader.take_snapshot()
            console.print(
                f"[green]✅ 模拟卖出: {symbol} {trade.shares}股 @ ¥{price:.2f} "
                f"| 盈亏: [{'green' if trade.realized_pnl>=0 else 'red'}]{trade.realized_pnl:+,.0f}[/]"
            )
        else:
            console.print("[red]卖出失败（无持仓/股数不足）[/red]")

    elif action == "reset":
        trader.reset()
        console.print("[green]✅ 模拟盘已重置[/green]")

    elif action == "snapshot":
        snap = trader.take_snapshot()
        console.print(f"[green]✅ 快照已记录: {snap.date} | 权益: ¥{snap.equity:,.0f}[/green]")

    else:
        console.print("[red]用法: paper summary|buy|sell|reset|snapshot[/red]")


# ── 组合回测 ──────────────────────────────────────────────

@app.command()
def pf_backtest(
    symbols: str = typer.Option("000001,000002,000858", "--symbols", help="逗号分隔股票代码"),
    allocation: str = typer.Option("equal_weight", "--alloc", help="分配策略: equal_weight / score_weighted / risk_parity / kelly"),
    rebalance: str = typer.Option("weekly", "--rebalance", help="再平衡周期: daily / weekly / monthly"),
    initial_capital: float = typer.Option(100000, "--capital", "-c", help="初始资金"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="交易策略"),
):
    """多资产组合回测。"""
    from backtest.portfolio import PortfolioBacktest, format_portfolio_result

    sym_list = [s.strip() for s in symbols.split(",")]
    a = get_agent(strategy)

    console.print(f"[dim]正在加载 {len(sym_list)} 只股票数据...[/dim]")

    signals_by_sym = {}
    price_data = {}
    scores = {}

    with console.status("[bold green]组合回测中...[/bold green]"):
        for sym in sym_list:
            try:
                kline = a.market_data.get_daily_kline(sym)
                if kline.empty:
                    console.print(f"[yellow]⚠️ {sym} 无K线数据[/yellow]")
                    continue
                price_data[sym] = kline

                # 生成回测信号
                st = StrategyFactory.get(strategy)
                signals_df = st.generate_bt_signals(kline)
                if not signals_df.empty:
                    signals_by_sym[sym] = signals_df

                # 评分
                score = a.analyze(sym)
                scores[sym] = score.total_score
            except Exception as e:
                console.print(f"[yellow]⚠️ {sym} 失败: {e}[/yellow]")

    if not signals_by_sym:
        console.print("[red]无有效数据[/red]")
        return

    engine = PortfolioBacktest(initial_capital=initial_capital)
    result = engine.run(
        signals_by_sym, price_data,
        allocation=allocation,
        rebalance=rebalance,
        scores=scores,
    )

    console.print(format_portfolio_result(result))

    # 权益曲线
    if not result.equity_curve.empty:
        console.print("\n[bold]权益曲线（最近20日）[/bold]")
        for _, row in result.equity_curve.tail(20).iterrows():
            console.print(
                f"  {row['date']} | "
                f"权益: ¥{row['equity']:,.0f} | "
                f"回撤: {row['drawdown']*100:.1f}% | "
                f"持仓: {int(row['positions'])}"
            )


# ── Web 面板 ──────────────────────────────────────────────

@app.command()
def web(
    port: int = typer.Option(8501, "--port", "-p", help="监听端口"),
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
):
    """启动 Web 可视化面板（Streamlit）。"""
    import subprocess

    web_app = os.path.join(os.path.dirname(__file__), "web", "app.py")

    console.print(f"[bold green]🚀 启动 Web 面板...[/bold green]")
    console.print(f"[dim]地址: http://{host}:{port}[/dim]")
    console.print("[dim]按 Ctrl+C 停止[/dim]")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        web_app,
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
    ])


# ── 自选股 ──────────────────────────────────────────────

@app.command()
def watchlist(
    action: str = typer.Argument("list", help="list / add / remove / scan / alerts / tags"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="股票代码"),
    name: str = typer.Option(None, "--name", help="股票名称"),
    tags: str = typer.Option("", "--tags", help="逗号分隔标签"),
    alert_rule: str = typer.Option("", "--alert", help="告警规则，如 pct_change_gt=5"),
    strategy: str = typer.Option("trend", "--strategy", help="扫描策略"),
):
    """管理自选股监视列表。"""
    from watchlist import Watchlist

    wl = Watchlist()

    if action == "list":
        items = wl.list()
        if not items:
            console.print("[yellow]自选股列表为空[/yellow]")
            console.print("[dim]使用 watchlist add --symbol 000001 --name 平安银行 添加[/dim]")
            return

        table = Table(title="自选股列表")
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("标签")
        table.add_column("告警条件")
        table.add_column("添加时间", style="dim")

        for item in items:
            alert_str = ", ".join(f"{k}={v}" for k, v in item.alerts.items()) or "-"
            table.add_row(
                item.symbol,
                item.name or "-",
                ", ".join(item.tags) or "-",
                alert_str,
                item.added_at[:16] if item.added_at else "-",
            )
        console.print(table)

        stats = wl.stats()
        console.print(f"\n[dim]共 {stats['total']} 只 | {stats['with_alerts']} 只设告警 | 文件: {stats['path']}[/dim]")

    elif action == "add":
        if not symbol:
            console.print("[red]请指定 --symbol 股票代码[/red]")
            return

        alerts = {}
        if alert_rule:
            for part in alert_rule.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        alerts[k.strip()] = float(v.strip())
                    except ValueError:
                        alerts[k.strip()] = v.strip()

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        wl.add(symbol, name=name or "", tags=tag_list, alerts=alerts)
        console.print(f"[green]✅ 已添加自选股: {symbol}{' ' + name if name else ''}[/green]")

    elif action == "remove":
        if not symbol:
            console.print("[red]请指定 --symbol 股票代码[/red]")
            return
        wl.remove(symbol)
        console.print(f"[green]✅ 已移除: {symbol}[/green]")

    elif action == "scan":
        items = wl.list()
        if not items:
            console.print("[yellow]自选股列表为空[/yellow]")
            return

        a = get_agent(strategy)
        with console.status(f"[bold green]扫描 {len(items)} 只自选股...[/bold green]"):
            results = wl.scan(a)
            alerts = wl.check_alerts(results)

        if not results:
            console.print("[yellow]扫描无结果[/yellow]")
            return

        table = Table(title="自选股扫描结果")
        table.add_column("代码", style="cyan")
        table.add_column("名称")
        table.add_column("评分", justify="right")
        table.add_column("评级")
        table.add_column("信号")
        table.add_column("价格", justify="right")
        table.add_column("涨跌", justify="right")

        for r in sorted(results, key=lambda x: x.score, reverse=True):
            pct_color = "green" if r.pct_change >= 0 else "red"
            table.add_row(
                r.symbol, r.name or "-",
                f"{r.score:.2f}", r.rating, r.signal,
                f"¥{r.latest_price:.2f}" if r.latest_price else "-",
                f"[{pct_color}]{r.pct_change:+.2f}%[/{pct_color}]",
            )
        console.print(table)

        # 告警
        if alerts:
            console.print(f"\n[bold red]⚠️ {len(alerts)} 条告警触发[/bold red]")
            for a in alerts:
                console.print(f"  [{a.severity}]{a.message}[/]")

    elif action == "alerts":
        if not symbol:
            console.print("[red]请指定 --symbol 股票代码[/red]")
            return
        item = wl.get(symbol)
        if not item:
            console.print(f"[yellow]{symbol} 不在自选股列表中[/yellow]")
            return
        if not item.alerts:
            console.print(f"[dim]{symbol} 无告警条件[/dim]")
        else:
            console.print(f"[bold]{symbol} {item.name}[/bold] 的告警条件:")
            for k, v in item.alerts.items():
                console.print(f"  {k}: {v}")
        console.print("[dim]设置告警: watchlist add --symbol {symbol} --alert pct_change_gt=5,pct_change_lt=-5[/dim]")

    elif action == "tags":
        if not symbol or not tags:
            console.print("[red]用法: watchlist tags --symbol 000001 --tags 科技,龙头[/red]")
            return
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        wl.tag(symbol, *tag_list)
        console.print(f"[green]✅ {symbol} 标签已更新: {', '.join(tag_list)}[/green]")

    else:
        console.print("[red]用法: watchlist list|add|remove|scan|alerts|tags[/red]")


# ── API 服务 ──────────────────────────────────────────────

@app.command()
def api(
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
):
    """启动 REST API 服务（FastAPI）。文档: http://localhost:8000/docs"""
    from api_server import serve
    console.print(f"[bold green]🚀 启动 API 服务...[/bold green]")
    console.print(f"[dim]地址: http://{host}:{port}[/dim]")
    console.print(f"[dim]文档: http://{host}:{port}/docs[/dim]")
    console.print("[dim]按 Ctrl+C 停止[/dim]")
    serve(host=host, port=port)


# ── main ───────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()