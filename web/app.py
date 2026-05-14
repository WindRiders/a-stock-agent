"""
A-Stock Agent Web 可视化面板。

基于 Streamlit + Plotly，提供：
- 市场总览（指数、涨跌分布）
- 评分排行（热力图、柱状图）
- 策略回测可视化（权益曲线、回撤）
- 持仓监控（饼图、盈亏瀑布图）
- 模拟盘追踪
- 信号日历

启动:
    streamlit run web/app.py

或通过 CLI:
    python cli.py web
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install streamlit plotly")
    sys.exit(1)

from agent.config import AgentConfig
from agent.core import TradingAgent
from strategy.factory import StrategyFactory


# ── 页面配置 ──────────────────────────────────────────────

st.set_page_config(
    page_title="A-Stock Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 样式 ──────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; }
    .signal-buy { color: #2ecc71; font-weight: 700; }
    .signal-sell { color: #e74c3c; font-weight: 700; }
    .rating-A { color: #2ecc71; }
    .rating-B { color: #27ae60; }
    .rating-C { color: #f39c12; }
    .rating-D { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ── 缓存 ──────────────────────────────────────────────────

@st.cache_resource
def get_agent(strategy: str = "trend") -> TradingAgent:
    return TradingAgent(AgentConfig(strategy=strategy))


@st.cache_data(ttl=300)
def cached_scan(agent, top_n: int = 30):
    return agent.scan_market(top_n=top_n, verbose=False)


@st.cache_data(ttl=300)
def cached_market_state(agent):
    return agent.detect_market_state()


# ── 侧边栏 ────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## ⚙️ 设置")

    strategy = st.sidebar.selectbox(
        "策略",
        StrategyFactory.list_strategies(),
        index=StrategyFactory.list_strategies().index("trend"),
    )

    top_n = st.sidebar.slider("扫描数量", 10, 100, 30, step=5)
    use_llm = st.sidebar.checkbox("启用 LLM 分析", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 导航")
    page = st.sidebar.radio(
        "页面",
        ["市场总览", "评分排行", "回测分析", "持仓监控", "模拟盘", "历史信号"],
    )

    return strategy, top_n, use_llm, page


# ── 市场总览 ──────────────────────────────────────────────

def render_market_overview(agent: TradingAgent):
    st.markdown('<p class="main-header">📈 市场总览</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    try:
        state = cached_market_state(agent)
        col1.metric("市场状态", state.style_name or "—")
        col2.metric("推荐策略", state.recommended_strategy or "—")
        col3.metric("置信度", f"{state.strategy_confidence:.0%}" if state.strategy_confidence else "—")
        col4.metric("风险等级", state.risk_level or "—")

        if state.warnings:
            for w in state.warnings:
                st.warning(w)
    except Exception as e:
        st.warning(f"市场检测暂不可用: {e}")

    # 实时行情
    st.markdown("---")
    st.subheader("🔥 实时行情速览")
    try:
        quotes = agent.market_data.get_realtime_quotes()
        if not quotes.empty:
            quotes = quotes.head(20)
            fig = go.Figure(data=[
                go.Table(
                    header=dict(
                        values=["代码", "名称", "现价", "涨跌幅%", "量比", "换手率%"],
                        fill_color="#1f77b4",
                        font=dict(color="white", size=12),
                        align="center",
                    ),
                    cells=dict(
                        values=[
                            quotes["symbol"].tolist(),
                            quotes.get("name", [""] * len(quotes)).tolist(),
                            [f"¥{p:.2f}" for p in quotes.get("price", [0])],
                            [f"{p:+.2f}" for p in quotes.get("pct_change", [0])],
                            [f"{v:.2f}" for v in quotes.get("volume_ratio", [0])],
                            [f"{t:.2f}" for t in quotes.get("turnover_rate", [0])],
                        ],
                        fill_color="#f8f9fa",
                        align="center",
                        font=dict(size=11),
                    ),
                )
            ])
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无行情数据")
    except Exception as e:
        st.info(f"行情数据加载中: {e}")


# ── 评分排行 ──────────────────────────────────────────────

def render_score_ranking(agent: TradingAgent, top_n: int):
    st.markdown('<p class="main-header">🏆 评分排行</p>', unsafe_allow_html=True)

    with st.spinner("正在扫描市场..."):
        scores = cached_scan(agent, top_n)

    if not scores:
        st.warning("暂无扫描结果")
        return

    # 指标卡片
    avg_score = np.mean([s.total_score for s in scores])
    buy_count = sum(1 for s in scores if s.signal in ("BUY", "STRONG_BUY"))
    a_count = sum(1 for s in scores if s.rating == "A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("有效评分", len(scores))
    col2.metric("平均得分", f"{avg_score:.2f}")
    col3.metric("买入信号", buy_count)
    col4.metric("A级股票", a_count)

    # 柱状图
    symbols = [s.symbol for s in scores[:15]]
    names = [s.name or s.symbol for s in scores[:15]]
    total_scores = [s.total_score for s in scores[:15]]

    colors = [
        "#2ecc71" if s.rating == "A" else
        "#27ae60" if s.rating == "B" else
        "#f39c12" if s.rating == "C" else
        "#e74c3c"
        for s in scores[:15]
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=total_scores,
            y=[f"{s} {n}" for s, n in zip(symbols, names)],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.2f}" for s in total_scores],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Top 15 评分排行",
        xaxis_title="综合评分",
        height=450,
        margin=dict(l=0, r=0, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 详细列表
    st.markdown("---")
    st.subheader("📋 评分明细")

    data = []
    for s in scores[:top_n]:
        data.append({
            "代码": s.symbol,
            "名称": s.name or "",
            "总分": f"{s.total_score:.2f}",
            "评级": s.rating,
            "信号": s.signal,
            "技术": s.tech_score,
            "基本": s.fund_score,
            "消息": s.sentiment_score,
            "资金": getattr(s, "capital_score", 0),
            "理由": "；".join(s.reasons[:3]) if s.reasons else "",
            "风险": "；".join(s.warnings[:2]) if s.warnings else "",
        })

    df = pd.DataFrame(data)

    def highlight_rating(val):
        colors = {"A": "color: #2ecc71", "B": "color: #27ae60",
                   "C": "color: #f39c12", "D": "color: #e74c3c"}
        return colors.get(val, "")

    styled = df.style.map(highlight_rating, subset=["评级"])
    st.dataframe(styled, use_container_width=True, height=500)


# ── 回测分析 ──────────────────────────────────────────────

def render_backtest(agent: TradingAgent):
    st.markdown('<p class="main-header">⏮️ 回测分析</p>', unsafe_allow_html=True)

    symbol = st.text_input("股票代码", "000001", max_chars=6)

    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("初始资金", 10000, 10000000, 100000, step=10000)
    with col2:
        strategy_name = st.selectbox("策略", StrategyFactory.list_strategies(), index=0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("▶️ 运行回测", type="primary", use_container_width=True)

    if not run_bt:
        st.info("输入股票代码，点击「运行回测」")
        return

    with st.spinner(f"正在回测 {symbol} ..."):
        try:
            # 生成回测信号
            score = agent.analyze(symbol)
            kline = agent.market_data.get_daily_kline(symbol)

            if kline.empty:
                st.error(f"无法获取 {symbol} 的K线数据")
                return

            # 基于策略生成信号
            strategy = StrategyFactory.get(strategy_name)
            strategy_df = strategy.generate_bt_signals(kline)

            if strategy_df.empty:
                st.warning("无法生成回测信号")
                return

            result = agent.backtest(strategy_df, initial_capital=initial_capital)
        except Exception as e:
            st.error(f"回测失败: {e}")
            return

    # 指标卡
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总收益", f"{result.total_return:+.2f}%")
    col2.metric("年化收益", f"{result.annual_return:+.2f}%")
    col3.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    col4.metric("最大回撤", f"{result.max_drawdown:.2f}%")
    col5.metric("胜率", f"{result.win_rate:.1f}%")

    # 权益曲线
    if not result.equity_curve.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
        )

        equity = result.equity_curve

        fig.add_trace(
            go.Scatter(
                x=equity["date"], y=equity["equity"],
                mode="lines", name="权益",
                fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.1)",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=equity["date"], y=equity.get("drawdown", [0]) * 100,
                mode="lines", name="回撤%",
                fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.15)",
                line=dict(color="#e74c3c", width=1),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            title=f"{symbol} 回测权益曲线",
            height=500,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=10),
        )
        fig.update_yaxes(title_text="权益 (¥)", row=1, col=1)
        fig.update_yaxes(title_text="回撤 %", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # 交易记录
    if result.trades:
        st.markdown("---")
        st.subheader("📝 交易记录")
        trades_data = [
            {"日期": t.date, "代码": t.symbol, "操作": t.action,
             "价格": f"¥{t.price:.2f}", "股数": t.shares,
             "金额": f"¥{t.amount:,.0f}", "佣金": f"¥{t.commission:.2f}"}
            for t in result.trades
        ]
        st.dataframe(pd.DataFrame(trades_data), use_container_width=True, height=300)


# ── 持仓监控 ──────────────────────────────────────────────

def render_portfolio(agent: TradingAgent):
    st.markdown('<p class="main-header">💼 持仓监控</p>', unsafe_allow_html=True)

    try:
        agent.refresh_prices()
    except Exception:
        pass

    positions = agent.get_portfolio()
    summary = agent.get_portfolio_summary()

    # 汇总卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("持仓数", summary.get("active_count", 0))
    col2.metric("总市值", f"¥{summary.get('total_market_value', 0):,.0f}")
    col3.metric("总成本", f"¥{summary.get('total_cost', 0):,.0f}")
    col4.metric("浮动盈亏",
                f"¥{summary.get('active_profit', 0):+,.0f}",
                delta=f"{summary.get('active_profit_pct', 0):+.1f}%")

    if not positions:
        st.info("暂无持仓")
        return

    # 饼图
    labels = [p["symbol"] + (" " + p["name"] if p.get("name") else "") for p in positions]
    values = [p.get("market_value", 0) or 0 for p in positions]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                  hole=0.4, textinfo="label+percent")])
    fig.update_layout(title="持仓分布", height=350, margin=dict(l=0, r=0, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 盈亏瀑布图
    st.markdown("---")
    pnl_data = []
    for p in positions:
        pnl_data.append({
            "标的": p["symbol"],
            "浮动盈亏": p.get("profit", 0) or 0,
            "收益率%": p.get("profit_pct", 0) or 0,
        })

    pnl_df = pd.DataFrame(pnl_data)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pnl_df["浮动盈亏"]]

    fig2 = go.Figure(data=[
        go.Bar(
            x=pnl_df["标的"],
            y=pnl_df["浮动盈亏"],
            marker_color=colors,
            text=[f"¥{v:,.0f}" for v in pnl_df["浮动盈亏"]],
            textposition="outside",
        )
    ])
    fig2.update_layout(
        title="持仓盈亏",
        yaxis_title="浮动盈亏 (¥)",
        height=350,
        margin=dict(l=0, r=0, t=30, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── 模拟盘 ─────────────────────────────────────────────────

def render_paper_trading(agent: TradingAgent):
    st.markdown('<p class="main-header">🎮 模拟盘</p>', unsafe_allow_html=True)

    from paper_trading import PaperTrader

    # 初始化/加载模拟盘
    if "paper_trader" not in st.session_state:
        st.session_state.paper_trader = PaperTrader(initial_capital=100000)

    trader = st.session_state.paper_trader

    summary = trader.summary()

    # 指标
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总权益", f"¥{summary.equity:,.0f}")
    col2.metric("总收益", f"{summary.total_return_pct:+.2f}%")
    col3.metric("现金", f"¥{summary.cash:,.0f}")
    col4.metric("胜率", f"{summary.win_rate:.1f}%")
    col5.metric("夏普", f"{summary.sharpe_ratio:.2f}")

    # 权益曲线
    if summary.equity_curve:
        dates = [d["date"] for d in summary.equity_curve]
        equities = [d["equity"] for d in summary.equity_curve]
        fig = go.Figure(data=[
            go.Scatter(x=dates, y=equities, mode="lines+markers",
                       name="权益", line=dict(color="#1f77b4", width=2))
        ])
        fig.update_layout(title="模拟盘权益曲线", height=300,
                          margin=dict(l=0, r=0, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # 手动交易
    st.markdown("---")
    st.subheader("✏️ 模拟交易")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_sym = st.text_input("代码", key="sim_sym")
    with col2:
        sim_action = st.selectbox("操作", ["BUY", "SELL"], key="sim_action")
    with col3:
        sim_price = st.number_input("价格", 0.01, 10000.0, 10.0, key="sim_price")
    with col4:
        sim_shares = st.number_input("股数", 100, 100000, 1000, step=100, key="sim_shares")

    sim_reason = st.text_input("理由", "", key="sim_reason")

    if st.button("✅ 执行模拟交易", type="primary"):
        trade = trader.execute(sim_sym, sim_action, sim_price, sim_shares, reason=sim_reason)
        if trade:
            st.success(f"{sim_action} {sim_sym} {sim_shares}股 @ ¥{sim_price:.2f}")
            trader.take_snapshot()
        else:
            st.error("交易执行失败（资金不足/股数不足/无持仓）")

    # 持仓
    positions = trader.positions_list()
    if positions:
        st.markdown("---")
        st.subheader("📋 模拟持仓")
        st.dataframe(pd.DataFrame(positions), use_container_width=True)

    # 最近交易
    recent = trader.recent_trades(10)
    if recent:
        st.markdown("---")
        st.subheader("📝 最近交易")
        st.dataframe(pd.DataFrame(recent), use_container_width=True)

    # 重置
    if st.button("🔄 重置模拟盘"):
        trader.reset()
        st.success("模拟盘已重置")
        st.rerun()


# ── 历史信号 ──────────────────────────────────────────────

def render_history(agent: TradingAgent):
    st.markdown('<p class="main-header">📜 历史信号</p>', unsafe_allow_html=True)

    scans = agent.get_history(50)
    if not scans:
        st.info("暂无历史记录")
        return

    st.dataframe(pd.DataFrame(scans), use_container_width=True)

    # 信号准确率
    from data.store import DataStore
    store = DataStore()
    accuracy = store.calc_signal_accuracy()

    if accuracy:
        st.markdown("---")
        st.subheader("📊 信号准确率")
        col1, col2, col3 = st.columns(3)
        col1.metric("总信号", accuracy.get("total_signals", 0))
        col2.metric("盈利信号", accuracy.get("profitable_signals", 0))
        col3.metric("准确率", f"{accuracy.get('accuracy', 0):.1f}%")


# ── 主入口 ────────────────────────────────────────────────

def main():
    strategy, top_n, use_llm, page = render_sidebar()

    agent = get_agent(strategy)

    if page == "市场总览":
        render_market_overview(agent)
    elif page == "评分排行":
        render_score_ranking(agent, top_n)
    elif page == "回测分析":
        render_backtest(agent)
    elif page == "持仓监控":
        render_portfolio(agent)
    elif page == "模拟盘":
        render_paper_trading(agent)
    elif page == "历史信号":
        render_history(agent)

    # 页脚
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #999; font-size: 0.8rem;'>"
        f"A-Stock Agent v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()