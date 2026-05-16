"""
仪表盘 HTTP 服务 — 零外部依赖，Python stdlib 即可启动。

启动: python web/server.py [--port 8888]
CLI:  python cli.py dashboard-serve
"""
import http.server
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Ensure project root in path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def get_api_state():
    """返回当前自主交易状态。"""
    try:
        from agent.trader import AutoTrader
        trader = AutoTrader(mode="paper")
        status = trader.status()
        summary = trader.paper_trader.summary()

        # 持仓
        active_positions = []
        for sym, pos in trader.paper_trader.positions.items():
            active_positions.append({
                "symbol": sym,
                "name": pos.name,
                "shares": pos.shares,
                "avg_cost": round(pos.avg_cost, 3),
                "current_price": round(pos.current_price, 3),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pos.unrealized_pnl_pct, 4),
            })

        # 最近决策
        last_decisions = []
        if trader._decisions:
            for d in trader._decisions:
                last_decisions.append({
                    "symbol": d.symbol,
                    "name": d.name,
                    "signal": d.signal,
                    "score": round(d.score, 2),
                    "action": d.action,
                    "amount": round(d.amount, 2),
                    "reason": d.reason,
                })

        # 权益曲线
        equity_curve = summary.equity_curve[-30:] if summary.equity_curve else []

        return {
            "mode": status["mode"],
            "strategy": status["strategy"],
            "market_state": status.get("market_state", ""),
            "equity": round(status["equity"], 2),
            "cash": round(status["cash"], 2),
            "total_return_pct": round(status["total_return_pct"], 2),
            "total_trades": status["total_trades"],
            "win_rate": round(status["win_rate"], 1),
            "sharpe": round(status["sharpe"], 2),
            "max_drawdown": round(status["max_drawdown"], 2),
            "positions": status["positions"],
            "positions_value": round(summary.positions_value, 2),
            "active_positions": active_positions,
            "last_decisions": last_decisions,
            "last_date": trader.session.date if trader.session else "",
            "equity_curve": equity_curve,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {
            "error": str(e),
            "mode": "paper",
            "equity": 0, "cash": 0, "total_return_pct": 0,
            "total_trades": 0, "win_rate": 0, "sharpe": 0, "max_drawdown": 0,
            "positions": 0, "positions_value": 0,
            "active_positions": [], "last_decisions": [], "equity_curve": [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def get_api_reports():
    """列出历史交易报告。"""
    reports_dir = os.path.expanduser("~/.a-stock-agent/reports")
    reports = []
    log_lines = []

    if os.path.isdir(reports_dir):
        # 报告列表
        for f in sorted(glob.glob(os.path.join(reports_dir, "trade_*.md")), reverse=True):
            fname = os.path.basename(f)
            date = fname.replace("trade_", "").replace(".md", "")
            reports.append({
                "date": date,
                "path": f"/reports/{fname}",
                "size": os.path.getsize(f),
            })

        # 日志（最后 100 行）
        log_file = os.path.join(reports_dir, "cron.log")
        if os.path.exists(log_file):
            try:
                with open(log_file) as lf:
                    lines = lf.readlines()
                    log_lines = [l.rstrip() for l in lines[-100:]]
            except Exception:
                pass

    return {"reports": reports, "log": log_lines}


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """处理仪表盘 HTTP 请求。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(PROJECT_DIR, "web"), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # API endpoints
        if parsed.path == "/api/state":
            self._json_response(get_api_state())
            return

        if parsed.path == "/api/reports":
            self._json_response(get_api_reports())
            return

        # Report files
        if parsed.path.startswith("/reports/"):
            fname = os.path.basename(parsed.path)
            report_path = os.path.expanduser(f"~/.a-stock-agent/reports/{fname}")
            if os.path.exists(report_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.end_headers()
                with open(report_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Report not found")
            return

        # Default: serve dashboard.html
        if parsed.path == "/" or parsed.path == "":
            self.path = "/dashboard.html"

        super().do_GET()

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # 静默日志


def main():
    import argparse
    parser = argparse.ArgumentParser(description="A股交易仪表盘 HTTP 服务")
    parser.add_argument("--port", type=int, default=8888, help="端口号 (默认 8888)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")
    args = parser.parse_args()

    server = http.server.HTTPServer((args.host, args.port), DashboardHandler)
    print(f"\n  📊 A股交易仪表盘")
    print(f"  ─────────────────")
    print(f"  地址: http://localhost:{args.port}")
    print(f"  API:  http://localhost:{args.port}/api/state")
    print(f"  API:  http://localhost:{args.port}/api/reports")
    print(f"  按 Ctrl+C 停止\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
        server.shutdown()


if __name__ == "__main__":
    main()