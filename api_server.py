"""
A-Stock Agent REST API 服务。

基于 FastAPI，暴露所有核心功能为 REST 接口。
启动: python cli.py api --port 8000
文档: http://localhost:8000/docs

端点:
  GET  /health                   — 健康检查
  GET  /api/v1/market/state      — 市场状态
  POST /api/v1/scan              — 全市场扫描
  GET  /api/v1/stock/{symbol}    — 单股分析
  GET  /api/v1/signals           — 交易信号
  POST /api/v1/backtest/{symbol} — 回测
  GET  /api/v1/portfolio         — 持仓
  GET  /api/v1/trades            — 交易记录
  GET  /api/v1/history           — 扫描历史
  GET  /api/v1/accuracy          — 信号准确率
  POST /api/v1/paper/{action}    — 模拟盘操作
  GET  /api/v1/watchlist         — 自选股列表
  POST /api/v1/watchlist/scan    — 扫描自选股
  POST /api/v1/config            — 配置操作
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("缺少 fastapi 依赖，请运行: pip install fastapi uvicorn")
    sys.exit(1)

from agent.config import AgentConfig
from agent.config_file import ConfigManager
from agent.core import TradingAgent
from strategy.factory import StrategyFactory
from watchlist import Watchlist
from paper_trading import PaperTrader

# ── 应用初始化 ────────────────────────────────────────────

app = FastAPI(
    title="A-Stock Agent API",
    description="A股量化交易智能体 REST API — 技术分析 + 基本面评估 + 策略信号 + 回测引擎",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
_agent: Optional[TradingAgent] = None
_watchlist: Optional[Watchlist] = None
_paper_trader: Optional[PaperTrader] = None


def get_agent(strategy: str = "trend") -> TradingAgent:
    global _agent
    if _agent is None or _agent.config.strategy != strategy:
        _agent = TradingAgent(AgentConfig(strategy=strategy))
    return _agent


def get_watchlist() -> Watchlist:
    global _watchlist
    if _watchlist is None:
        _watchlist = Watchlist()
    return _watchlist


def get_paper_trader() -> PaperTrader:
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader(
            initial_capital=100000,
            db_path=os.path.expanduser("~/.a-stock-agent/paper.db"),
        )
    return _paper_trader


# ── 请求/响应模型 ─────────────────────────────────────────

class ScanRequest(BaseModel):
    top_n: int = Field(20, ge=1, le=200, description="返回前N只股票")
    strategy: str = Field("trend", description="策略名称")


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="股票代码")
    initial_capital: float = Field(100000, description="初始资金")
    strategy: str = Field("trend", description="策略名称")


class PaperTradeRequest(BaseModel):
    action: str = Field(..., description="BUY 或 SELL")
    symbol: str = Field(..., description="股票代码")
    price: float = Field(..., gt=0, description="成交价")
    shares: int = Field(0, ge=0, description="股数")
    amount: float = Field(0, ge=0, description="买入金额（与shares二选一）")
    reason: str = Field("", description="交易理由")


class WatchlistAddRequest(BaseModel):
    symbol: str = Field(..., description="股票代码")
    name: str = Field("", description="股票名称")
    tags: List[str] = Field(default_factory=list, description="标签")
    notes: str = Field("", description="备注")
    alerts: Dict[str, float] = Field(default_factory=dict, description="告警条件")


class ConfigUpdateRequest(BaseModel):
    strategy: Optional[str] = None
    max_positions: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    llm_enabled: Optional[bool] = None
    llm_model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str


# ── 端点 ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """健康检查。"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


# ── 市场状态 ──────────────────────────────────────────────

@app.get("/api/v1/market/state")
def market_state(strategy: str = "trend"):
    """获取当前市场状态。"""
    agent = get_agent(strategy)
    try:
        state = agent.detect_market_state()
        return {
            "regime": state.regime_cn,
            "trend": state.trend_direction,
            "volatility": state.volatility,
            "volatility_regime": state.volatility_regime,
            "risk_level": state.risk_level,
            "recommended_strategy": state.recommended_strategy,
            "confidence": state.strategy_confidence,
            "warnings": state.warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 全市场扫描 ────────────────────────────────────────────

@app.post("/api/v1/scan")
def scan_market(req: ScanRequest):
    """全市场扫描，返回评分排行。"""
    agent = get_agent(req.strategy)
    try:
        scores = agent.scan_market(top_n=req.top_n, verbose=False)
        return {
            "count": len(scores),
            "strategy": req.strategy,
            "results": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "total_score": s.total_score,
                    "rating": s.rating,
                    "signal": s.signal,
                    "tech_score": s.tech_score,
                    "fund_score": s.fund_score,
                    "sentiment_score": s.sentiment_score,
                    "capital_score": getattr(s, "capital_score", 0),
                    "latest_price": s.latest_price,
                    "pe": s.pe,
                    "pb": s.pb,
                    "reasons": s.reasons[:5],
                    "warnings": s.warnings[:3],
                }
                for s in scores
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 单股分析 ──────────────────────────────────────────────

@app.get("/api/v1/stock/{symbol}")
def analyze_stock(symbol: str, strategy: str = "trend"):
    """深度分析单只股票。"""
    agent = get_agent(strategy)
    try:
        score = agent.analyze(symbol)
        return {
            "symbol": score.symbol,
            "name": score.name,
            "total_score": score.total_score,
            "rating": score.rating,
            "signal": score.signal,
            "tech_score": score.tech_score,
            "fund_score": score.fund_score,
            "sentiment_score": score.sentiment_score,
            "capital_score": getattr(score, "capital_score", 0),
            "latest_price": score.latest_price,
            "pe": score.pe,
            "pb": score.pb,
            "reasons": score.reasons,
            "warnings": score.warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 交易信号 ──────────────────────────────────────────────

@app.get("/api/v1/signals")
def signals(strategy: str = "trend"):
    """获取当前交易信号。"""
    agent = get_agent(strategy)
    try:
        sigs = agent.generate_signals()
        return {
            "count": len(sigs),
            "signals": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "signal": s.signal.value,
                    "score": s.score,
                    "confidence": s.confidence,
                    "reason": s.reason,
                }
                for s in sigs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 回测 ──────────────────────────────────────────────────

@app.post("/api/v1/backtest")
def run_backtest(req: BacktestRequest):
    """运行单股回测。"""
    agent = get_agent(req.strategy)
    try:
        kline = agent.market_data.get_daily_kline(req.symbol)
        if kline.empty:
            raise HTTPException(status_code=404, detail=f"无法获取 {req.symbol} K线数据")

        strategy = StrategyFactory.get(req.strategy)
        signals_df = strategy.generate_bt_signals(kline)

        if signals_df.empty:
            return {"error": "无法生成回测信号", "symbol": req.symbol}

        result = agent.backtest(signals_df, initial_capital=req.initial_capital)

        return {
            "symbol": req.symbol,
            "strategy": req.strategy,
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "equity_curve": result.equity_curve.to_dict("records") if not result.equity_curve.empty else [],
            "trades": [
                {"date": t.date, "action": t.action, "price": t.price,
                 "shares": t.shares, "amount": t.amount}
                for t in result.trades[:20]
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 持仓 ──────────────────────────────────────────────────

@app.get("/api/v1/portfolio")
def portfolio_refresh(strategy: str = "trend"):
    """获取持仓列表（含刷新价格）。"""
    agent = get_agent(strategy)
    try:
        agent.refresh_prices()
    except Exception:
        pass

    positions = agent.get_portfolio()
    summary = agent.get_portfolio_summary()

    return {
        "summary": {
            "active_count": summary.get("active_count", 0),
            "total_market_value": summary.get("total_market_value", 0),
            "total_cost": summary.get("total_cost", 0),
            "active_profit": summary.get("active_profit", 0),
            "active_profit_pct": summary.get("active_profit_pct", 0),
        },
        "positions": [
            {
                "symbol": p["symbol"],
                "name": p.get("name", ""),
                "shares": p.get("shares", 0),
                "avg_cost": p.get("avg_cost", 0),
                "current_price": p.get("current_price", 0),
                "market_value": p.get("market_value", 0),
                "profit": p.get("profit", 0),
                "profit_pct": p.get("profit_pct", 0),
            }
            for p in positions
        ],
    }


# ── 交易记录 ──────────────────────────────────────────────

@app.get("/api/v1/trades")
def trades(limit: int = Query(20, ge=1, le=500)):
    """获取交易记录。"""
    agent = get_agent()
    records = agent.get_trades(limit)
    return {
        "count": len(records),
        "trades": [
            {
                "id": t.get("id"),
                "date": str(t.get("traded_at", "")),
                "symbol": t.get("symbol"),
                "action": t.get("action"),
                "price": t.get("price"),
                "shares": t.get("shares"),
                "amount": t.get("amount"),
                "commission": t.get("commission", 0),
                "stamp_tax": t.get("stamp_tax", 0),
                "reason": t.get("reason", ""),
            }
            for t in records
        ],
    }


# ── 历史 ──────────────────────────────────────────────────

@app.get("/api/v1/history")
def history(limit: int = Query(20, ge=1, le=100)):
    """扫描历史。"""
    agent = get_agent()
    scans = agent.get_history(limit)
    return {"count": len(scans), "scans": scans}


# ── 准确率 ────────────────────────────────────────────────

@app.get("/api/v1/accuracy")
def accuracy():
    """信号准确率统计。"""
    agent = get_agent()
    try:
        result = agent.calc_accuracy()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 模拟盘 ────────────────────────────────────────────────

@app.get("/api/v1/paper/summary")
def paper_summary():
    """模拟盘汇总。"""
    trader = get_paper_trader()
    s = trader.summary()
    return {
        "initial_capital": s.initial_capital,
        "equity": s.equity,
        "total_return_pct": s.total_return_pct,
        "cash": s.cash,
        "total_trades": s.total_trades,
        "win_rate": s.win_rate,
        "sharpe_ratio": s.sharpe_ratio,
        "max_drawdown": s.max_drawdown,
        "active_positions": s.active_positions,
        "positions_value": s.positions_value,
        "positions": trader.positions_list(),
        "recent_trades": trader.recent_trades(10),
    }


@app.post("/api/v1/paper/trade")
def paper_trade(req: PaperTradeRequest):
    """执行模拟交易。"""
    trader = get_paper_trader()
    trade = trader.execute(req.symbol, req.action, req.price,
                           req.shares, reason=req.reason, amount=req.amount)
    if trade:
        trader.take_snapshot()
        return {
            "success": True,
            "trade": {
                "id": trade.id,
                "date": trade.date,
                "symbol": trade.symbol,
                "action": trade.action,
                "price": trade.price,
                "shares": trade.shares,
                "amount": trade.amount,
                "commission": trade.commission,
                "stamp_tax": trade.stamp_tax,
                "realized_pnl": trade.realized_pnl,
            },
        }
    else:
        raise HTTPException(status_code=400, detail="交易失败（资金不足/无持仓/股数不足）")


@app.post("/api/v1/paper/reset")
def paper_reset():
    """重置模拟盘。"""
    trader = get_paper_trader()
    trader.reset()
    return {"success": True, "message": "模拟盘已重置"}


# ── 自选股 ────────────────────────────────────────────────

@app.get("/api/v1/watchlist")
def watchlist_list():
    """列出所有自选股。"""
    wl = get_watchlist()
    items = [
        {
            "symbol": item.symbol,
            "name": item.name,
            "added_at": item.added_at,
            "tags": item.tags,
            "notes": item.notes,
            "alerts": item.alerts,
        }
        for item in wl.list()
    ]
    stats = wl.stats()
    return {"count": len(items), "items": items, "stats": stats}


@app.post("/api/v1/watchlist/add")
def watchlist_add(req: WatchlistAddRequest):
    """添加自选股。"""
    wl = get_watchlist()
    wl.add(req.symbol, name=req.name, tags=req.tags, notes=req.notes, alerts=req.alerts)
    return {"success": True, "symbol": req.symbol}


@app.delete("/api/v1/watchlist/{symbol}")
def watchlist_remove(symbol: str):
    """移除自选股。"""
    wl = get_watchlist()
    wl.remove(symbol)
    return {"success": True, "symbol": symbol}


@app.post("/api/v1/watchlist/scan")
def watchlist_scan(strategy: str = "trend"):
    """扫描自选股列表。"""
    wl = get_watchlist()
    agent = get_agent(strategy)

    results = wl.scan(agent)
    alerts = wl.check_alerts(results)

    return {
        "results": [
            {
                "symbol": r.symbol,
                "name": r.name,
                "score": r.score,
                "rating": r.rating,
                "signal": r.signal,
                "latest_price": r.latest_price,
                "pct_change": r.pct_change,
                "volume_ratio": r.volume_ratio,
                "pe": r.pe,
            }
            for r in results
        ],
        "alerts": [
            {
                "symbol": a.symbol,
                "name": a.name,
                "type": a.alert_type,
                "message": a.message,
                "severity": a.severity,
            }
            for a in alerts
        ],
    }


# ── 配置 ──────────────────────────────────────────────────

@app.get("/api/v1/config")
def config_get():
    """获取当前配置。"""
    mgr = ConfigManager()
    cfg = mgr.load()
    return {
        "strategy": cfg.strategy,
        "risk": {
            "max_positions": cfg.max_positions,
            "max_position_pct": cfg.max_position_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
        },
        "scan": {"top_n": cfg.scan_top_n},
        "llm": {
            "enabled": cfg.llm_enabled,
            "provider": cfg.llm_provider,
            "model": cfg.llm_model,
            "base_url": cfg.llm_base_url,
        },
        "data": {
            "cache_days": cfg.cache_days,
            "realtime_enabled": cfg.realtime_enabled,
        },
    }


@app.post("/api/v1/config")
def config_update(req: ConfigUpdateRequest):
    """更新配置。"""
    mgr = ConfigManager()
    cfg = mgr.load()

    if req.strategy is not None:
        cfg.strategy = req.strategy
    if req.max_positions is not None:
        cfg.max_positions = req.max_positions
    if req.stop_loss_pct is not None:
        cfg.stop_loss_pct = req.stop_loss_pct
    if req.take_profit_pct is not None:
        cfg.take_profit_pct = req.take_profit_pct
    if req.llm_enabled is not None:
        cfg.llm_enabled = req.llm_enabled
    if req.llm_model is not None:
        cfg.llm_model = req.llm_model

    mgr.save(cfg)

    global _agent
    _agent = None  # 重建 agent 以应用新配置

    return {"success": True}


# ── 策略列表 ──────────────────────────────────────────────

@app.get("/api/v1/strategies")
def list_strategies():
    """列出可用策略。"""
    return {"strategies": StrategyFactory.list_strategies()}


# ── 启动入口 ──────────────────────────────────────────────

def serve(host: str = "0.0.0.0", port: int = 8000):
    """启动 API 服务器。"""
    import uvicorn
    uvicorn.run("api_server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    serve()