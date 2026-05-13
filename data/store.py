"""数据持久化模块。

使用 SQLite 存储扫描历史、信号、持仓和交易记录。
零外部依赖（Python 内置 sqlite3）。
"""

import json
import logging
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认数据库路径
DEFAULT_DB_PATH = os.path.expanduser("~/.a-stock-agent/history.db")


class DataStore:
    """SQLite 数据存储。

    表结构：
    - scans: 扫描记录（时间/策略/股票数）
    - scan_items: 扫描结果明细（股票/评分/信号/评级/价格）
    - signals: 生成的交易信号
    - portfolio: 持仓记录
    - trades: 交易记录（买入/卖出）
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """初始化数据库表。"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanned_at TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    total_stocks INTEGER,
                    scan_duration_sec REAL,
                    market_regime TEXT,
                    market_risk TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS scan_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    total_score REAL,
                    rating TEXT,
                    signal TEXT,
                    tech_score INTEGER,
                    fund_score INTEGER,
                    sentiment_score INTEGER,
                    capital_score INTEGER,
                    latest_price REAL,
                    pe REAL,
                    pb REAL,
                    volume_ratio REAL,
                    reasons TEXT,
                    warnings TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    scan_id INTEGER,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    signal TEXT NOT NULL,
                    score REAL,
                    confidence REAL,
                    reason TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    shares INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL,
                    market_value REAL,
                    profit REAL,
                    profit_pct REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    entered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    action TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    stamp_tax REAL DEFAULT 0,
                    traded_at TEXT NOT NULL,
                    reason TEXT,
                    signal_id INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_scan_items_scan_id ON scan_items(scan_id);
                CREATE INDEX IF NOT EXISTS idx_scan_items_symbol ON scan_items(symbol);
                CREATE INDEX IF NOT EXISTS idx_scan_items_signal ON scan_items(signal);
                CREATE INDEX IF NOT EXISTS idx_scans_scanned_at ON scans(scanned_at);
                CREATE INDEX IF NOT EXISTS idx_signals_signal ON signals(signal);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            """)

    # ── 扫描记录 ────────────────────────────────────────────

    def save_scan(
        self,
        scores: List,
        strategy: str = "",
        market_regime: str = "",
        market_risk: str = "",
        duration_sec: float = 0,
    ) -> int:
        """保存一次扫描结果。返回 scan_id。"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO scans (scanned_at, strategy, total_stocks, scan_duration_sec,
                   market_regime, market_risk)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (now, strategy, len(scores), duration_sec, market_regime, market_risk),
            )
            scan_id = cur.lastrowid

            # 批量插入明细
            items = []
            for s in scores:
                items.append((
                    scan_id, s.symbol, getattr(s, 'name', ''),
                    s.total_score, getattr(s, 'rating', 'C'), s.signal,
                    getattr(s, 'tech_score', 0), getattr(s, 'fund_score', 0),
                    getattr(s, 'sentiment_score', 0), getattr(s, 'capital_score', 0),
                    getattr(s, 'latest_price', None),
                    getattr(s, 'pe', None), getattr(s, 'pb', None),
                    getattr(s, 'volume_ratio', None),
                    json.dumps(s.reasons, ensure_ascii=False) if getattr(s, 'reasons', None) else None,
                    json.dumps(s.warnings, ensure_ascii=False) if getattr(s, 'warnings', None) else None,
                ))

            conn.executemany(
                """INSERT INTO scan_items
                   (scan_id, symbol, name, total_score, rating, signal,
                    tech_score, fund_score, sentiment_score, capital_score,
                    latest_price, pe, pb, volume_ratio, reasons, warnings)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                items,
            )

            logger.info("保存扫描 #%d: %d 只股票, 策略=%s", scan_id, len(scores), strategy)
            return scan_id

    def get_scan_history(self, limit: int = 20) -> List[Dict]:
        """获取最近的扫描记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, scanned_at, strategy, total_stocks, scan_duration_sec,
                   market_regime, market_risk, notes
                   FROM scans ORDER BY scanned_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_scan_detail(self, scan_id: int) -> List[Dict]:
        """获取某次扫描的明细。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT symbol, name, total_score, rating, signal,
                   tech_score, fund_score, latest_price, pe, volume_ratio
                   FROM scan_items WHERE scan_id = ?
                   ORDER BY total_score DESC""",
                (scan_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── 股票历史 ────────────────────────────────────────────

    def get_stock_history(
        self, symbol: str, limit: int = 30
    ) -> List[Dict]:
        """获取某只股票的历史评分记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.scanned_at, si.total_score, si.rating, si.signal,
                   si.latest_price, si.pe, si.tech_score, si.fund_score
                   FROM scan_items si
                   JOIN scans s ON si.scan_id = s.id
                   WHERE si.symbol = ?
                   ORDER BY s.scanned_at DESC
                   LIMIT ?""",
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_score_trend(self, symbol: str, limit: int = 20) -> List[Tuple]:
        """获取评分趋势（时间→分数）。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.scanned_at, si.total_score
                   FROM scan_items si
                   JOIN scans s ON si.scan_id = s.id
                   WHERE si.symbol = ?
                   ORDER BY s.scanned_at ASC
                   LIMIT ?""",
                (symbol, limit),
            ).fetchall()
            return [(r["scanned_at"], r["total_score"]) for r in rows]

    # ── 信号 ────────────────────────────────────────────────

    def save_signals(self, signals: List, strategy: str = "", scan_id: int = None) -> int:
        """批量保存信号。"""
        now = datetime.now().isoformat()
        count = 0

        with self._get_conn() as conn:
            for sig in signals:
                conn.execute(
                    """INSERT INTO signals (created_at, scan_id, strategy, symbol, name,
                       signal, score, confidence, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now, scan_id, strategy,
                        sig.symbol, getattr(sig, 'name', ''),
                        sig.signal.value if hasattr(sig.signal, 'value') else str(sig.signal),
                        getattr(sig, 'score', 0), getattr(sig, 'confidence', 0),
                        getattr(sig, 'reason', ''),
                    ),
                )
                count += 1

        return count

    def get_recent_signals(self, limit: int = 50, signal_filter: str = None) -> List[Dict]:
        """获取最近的信号。"""
        with self._get_conn() as conn:
            if signal_filter:
                rows = conn.execute(
                    """SELECT * FROM signals
                       WHERE signal = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (signal_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── 持仓 ────────────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        shares: int,
        avg_cost: float,
        name: str = "",
        stop_loss: float = None,
        take_profit: float = None,
    ) -> int:
        """开仓/加仓。"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            # 检查是否已持有
            existing = conn.execute(
                "SELECT id, shares, avg_cost FROM portfolio WHERE symbol = ? AND status = 'active'",
                (symbol,),
            ).fetchone()

            if existing:
                # 加仓：合并计算均价
                old_shares = existing["shares"]
                old_cost = existing["avg_cost"]
                total_shares = old_shares + shares
                new_avg_cost = (old_cost * old_shares + avg_cost * shares) / total_shares

                conn.execute(
                    """UPDATE portfolio
                       SET shares = ?, avg_cost = ?, updated_at = ?
                       WHERE id = ?""",
                    (total_shares, round(new_avg_cost, 4), now, existing["id"]),
                )
                return existing["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO portfolio
                       (symbol, name, shares, avg_cost, stop_loss, take_profit,
                        entered_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, name, shares, round(avg_cost, 4), stop_loss, take_profit, now, now),
                )
                return cur.lastrowid

    def close_position(self, symbol: str, exit_price: float = None) -> bool:
        """平仓。"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            pos = conn.execute(
                "SELECT * FROM portfolio WHERE symbol = ? AND status = 'active'",
                (symbol,),
            ).fetchone()

            if not pos:
                return False

            exit_price = exit_price or pos["current_price"] or pos["avg_cost"]

            conn.execute(
                """UPDATE portfolio
                   SET status = 'closed', current_price = ?, updated_at = ?,
                       market_value = ?, profit = ?,
                       profit_pct = ?
                   WHERE id = ?""",
                (
                    exit_price, now,
                    exit_price * pos["shares"],
                    (exit_price - pos["avg_cost"]) * pos["shares"],
                    round((exit_price / pos["avg_cost"] - 1) * 100, 2),
                    pos["id"],
                ),
            )
            return True

    def update_positions_prices(self, price_map: Dict[str, float]):
        """批量更新持仓的市场价格。"""
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            for symbol, price in price_map.items():
                pos = conn.execute(
                    "SELECT id, shares, avg_cost FROM portfolio WHERE symbol = ? AND status = 'active'",
                    (symbol,),
                ).fetchone()
                if pos:
                    mv = price * pos["shares"]
                    profit = (price - pos["avg_cost"]) * pos["shares"]
                    pnl_pct = round((price / pos["avg_cost"] - 1) * 100, 2) if pos["avg_cost"] > 0 else 0
                    conn.execute(
                        """UPDATE portfolio
                           SET current_price = ?, market_value = ?,
                               profit = ?, profit_pct = ?, updated_at = ?
                           WHERE id = ?""",
                        (price, mv, profit, pnl_pct, now, pos["id"]),
                    )

    def get_positions(self, status: str = "active") -> List[Dict]:
        """获取持仓。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio WHERE status = ? ORDER BY entered_at DESC",
                (status,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_portfolio_summary(self) -> Dict:
        """获取持仓汇总。"""
        with self._get_conn() as conn:
            active = conn.execute(
                "SELECT COUNT(*) as count, SUM(market_value) as total_value, SUM(profit) as total_profit FROM portfolio WHERE status = 'active'"
            ).fetchone()

            closed = conn.execute(
                "SELECT COUNT(*) as count, SUM(profit) as total_profit, AVG(profit_pct) as avg_pnl FROM portfolio WHERE status = 'closed'"
            ).fetchone()

            return {
                "active_count": active["count"] or 0,
                "total_market_value": active["total_value"] or 0,
                "active_profit": active["total_profit"] or 0,
                "closed_count": closed["count"] or 0,
                "closed_profit": closed["total_profit"] or 0,
                "closed_avg_pnl": round(closed["avg_pnl"] or 0, 2),
            }

    # ── 交易记录 ────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        name: str = "",
        commission: float = 0,
        stamp_tax: float = 0,
        reason: str = "",
    ) -> int:
        """记录一笔交易。"""
        now = datetime.now().isoformat()
        amount = price * shares

        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (symbol, name, action, shares, price, amount, commission,
                    stamp_tax, traded_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, name, action, shares, price, amount, commission, stamp_tax, now, reason),
            )
            return cur.lastrowid

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """获取交易记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY traded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── 统计 ────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """获取数据库统计。"""
        with self._get_conn() as conn:
            scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            items = conn.execute("SELECT COUNT(*) FROM scan_items").fetchone()[0]
            sig_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM portfolio WHERE status='active'").fetchone()[0]
            trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

            last_scan = conn.execute(
                "SELECT scanned_at FROM scans ORDER BY scanned_at DESC LIMIT 1"
            ).fetchone()

            return {
                "total_scans": scans,
                "total_items": items,
                "total_signals": sig_count,
                "active_positions": active,
                "total_trades": trades,
                "last_scan": last_scan["scanned_at"] if last_scan else None,
                "db_path": self.db_path,
            }