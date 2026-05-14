"""自选股监视列表。

功能：
- 维护用户自选股列表（JSON 文件持久化）
- 增量更新：仅获取监视列表中的股票数据，而非全市场扫描
- 价格/技术告警：突破 MA、涨跌幅阈值、评分变化
- 配置每个股票的个性化告警条件
- 与模拟盘联动：监视列表中信号触发时自动模拟交易

用法:
    from watchlist import Watchlist

    wl = Watchlist()
    wl.add("000001", name="平安银行", alerts={"pct_change_gt": 3, "pct_change_lt": -3})
    wl.remove("000002")
    wl.scan()  # 仅扫描监视列表中的股票
    alerts = wl.check_alerts()  # 检查告警
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST_PATH = os.path.expanduser("~/.a-stock-agent/watchlist.json")


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class WatchItem:
    """单个自选股。"""

    symbol: str
    name: str = ""
    added_at: str = ""  # ISO格式
    tags: List[str] = field(default_factory=list)  # 标签: ["科技", "龙头", "高股息"]
    notes: str = ""  # 用户备注
    alerts: Dict[str, Any] = field(default_factory=dict)  # 告警条件

    # 示例 alerts:
    # {
    #     "pct_change_gt": 5,       # 涨幅超过 5%
    #     "pct_change_lt": -5,      # 跌幅超过 5%
    #     "price_gt": 50.0,         # 价格超过 50
    #     "price_lt": 5.0,          # 价格低于 5
    #     "volume_ratio_gt": 3.0,   # 量比超过 3
    #     "ma_cross": "golden",     # 金叉/死叉告警
    #     "rating_below": "B",      # 评级低于 B
    #     "signal": "STRONG_BUY",   # 信号触达
    # }


@dataclass
class WatchAlert:
    """告警记录。"""

    symbol: str
    name: str
    alert_type: str  # 告警类型
    message: str
    current_value: Any
    threshold: Any
    triggered_at: str
    severity: str = "info"  # info / warning / critical


@dataclass
class WatchScanResult:
    """监视列表扫描结果。"""

    symbol: str
    name: str
    score: float = 0.0
    rating: str = ""
    signal: str = ""
    latest_price: float = 0.0
    pct_change: float = 0.0
    volume_ratio: float = 0.0
    pe: Optional[float] = None
    scanned_at: str = ""


# ── 主类 ──────────────────────────────────────────────────

class Watchlist:
    """自选股监视列表。

    用法:
        wl = Watchlist()
        wl.add("000001", name="平安银行", alerts={"pct_change_gt": 3})
        wl.add("000002", name="万科A")
        results = wl.scan(agent)  # 仅扫描自选股
        alerts = wl.check_alerts(results)
    """

    def __init__(self, path: str = None):
        self.path = path or DEFAULT_WATCHLIST_PATH
        self.items: Dict[str, WatchItem] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────

    def add(
        self,
        symbol: str,
        name: str = "",
        tags: List[str] = None,
        notes: str = "",
        alerts: Dict[str, Any] = None,
    ):
        """添加自选股。"""
        symbol = symbol.upper()
        if symbol in self.items:
            # 更新已有
            item = self.items[symbol]
            if name:
                item.name = name
            if tags is not None:
                item.tags = tags
            if notes:
                item.notes = notes
            if alerts is not None:
                item.alerts = alerts
        else:
            self.items[symbol] = WatchItem(
                symbol=symbol,
                name=name,
                added_at=datetime.now().isoformat(),
                tags=tags or [],
                notes=notes,
                alerts=alerts or {},
            )
        self._save()

    def remove(self, symbol: str):
        """移除自选股。"""
        symbol = symbol.upper()
        if symbol in self.items:
            del self.items[symbol]
            self._save()

    def list(self) -> List[WatchItem]:
        """列出所有自选股。"""
        return list(self.items.values())

    def get(self, symbol: str) -> Optional[WatchItem]:
        """获取单个自选股。"""
        return self.items.get(symbol.upper())

    def clear(self):
        """清空自选股列表。"""
        self.items = {}
        self._save()

    # ── 扫描 ──────────────────────────────────────────────

    def scan(self, agent) -> List[WatchScanResult]:
        """扫描监视列表中的所有股票。

        仅获取自选股的数据，而非全市场扫描。
        比全市场扫描快 10-100 倍。

        Args:
            agent: TradingAgent 实例

        Returns:
            扫描结果列表
        """
        if not self.items:
            logger.warning("自选股列表为空")
            return []

        symbols = list(self.items.keys())
        results = []

        for symbol in symbols:
            item = self.items[symbol]
            try:
                score = agent.analyze(symbol)
                if score.name is None:
                    score.name = item.name

                # 获取实时行情
                price = 0.0
                pct_change = 0.0
                volume_ratio = 0.0
                try:
                    quotes = agent.market_data.get_realtime_quotes()
                    if not quotes.empty:
                        match = quotes[quotes["symbol"] == symbol]
                        if not match.empty:
                            price = float(match.iloc[0].get("price", 0))
                            pct_change = float(match.iloc[0].get("pct_change", 0))
                            volume_ratio = float(match.iloc[0].get("volume_ratio", 0))
                except Exception:
                    pass

                results.append(WatchScanResult(
                    symbol=symbol,
                    name=item.name,
                    score=score.total_score,
                    rating=score.rating,
                    signal=score.signal,
                    latest_price=price or (score.latest_price or 0),
                    pct_change=pct_change,
                    volume_ratio=volume_ratio,
                    pe=score.pe,
                    scanned_at=datetime.now().isoformat(),
                ))
            except Exception as e:
                logger.warning("扫描 %s 失败: %s", symbol, e)

        return results

    # ── 告警 ──────────────────────────────────────────────

    def check_alerts(self, scan_results: List[WatchScanResult]) -> List[WatchAlert]:
        """根据扫描结果检查告警条件。

        Args:
            scan_results: scan() 返回的结果列表

        Returns:
            触发的告警列表
        """
        alerts = []
        now = datetime.now().isoformat()

        for result in scan_results:
            item = self.items.get(result.symbol)
            if not item or not item.alerts:
                continue

            for alert_type, threshold in item.alerts.items():
                alert = self._evaluate_alert(alert_type, threshold, result, item, now)
                if alert:
                    alerts.append(alert)

        return alerts

    def _evaluate_alert(
        self,
        alert_type: str,
        threshold: Any,
        result: WatchScanResult,
        item: WatchItem,
        now: str,
    ) -> Optional[WatchAlert]:
        """评估单个告警条件。"""
        msg = ""
        current = None

        if alert_type == "pct_change_gt":
            current = result.pct_change
            if current >= threshold:
                msg = f"{item.name}({result.symbol}) 涨幅 {current:+.1f}% 超过阈值 {threshold}%"
        elif alert_type == "pct_change_lt":
            current = result.pct_change
            if current <= threshold:
                msg = f"{item.name}({result.symbol}) 跌幅 {current:+.1f}% 超过阈值 {threshold}%"
        elif alert_type == "price_gt":
            current = result.latest_price
            if current >= threshold:
                msg = f"{item.name}({result.symbol}) 价格 ¥{current:.2f} 超过 ¥{threshold}"
        elif alert_type == "price_lt":
            current = result.latest_price
            if current <= threshold:
                msg = f"{item.name}({result.symbol}) 价格 ¥{current:.2f} 跌破 ¥{threshold}"
        elif alert_type == "volume_ratio_gt":
            current = result.volume_ratio
            if current >= threshold:
                msg = f"{item.name}({result.symbol}) 量比 {current:.2f} 超过 {threshold}"
        elif alert_type == "rating_below":
            current = result.rating
            rating_order = {"A": 0, "B": 1, "C": 2, "D": 3}
            if rating_order.get(current, 0) >= rating_order.get(threshold, 0):
                msg = f"{item.name}({result.symbol}) 评级 {current} 低于 {threshold}"
        elif alert_type == "signal":
            current = result.signal
            if current == threshold:
                msg = f"{item.name}({result.symbol}) 触发信号: {threshold}"
        elif alert_type == "score_gt":
            current = result.score
            if current >= threshold:
                msg = f"{item.name}({result.symbol}) 评分 {current:.2f} 超过 {threshold}"
        elif alert_type == "score_lt":
            current = result.score
            if current <= threshold:
                msg = f"{item.name}({result.symbol}) 评分 {current:.2f} 低于 {threshold}"

        if msg:
            severity = "critical" if alert_type in ("price_lt", "pct_change_lt") else "warning"
            return WatchAlert(
                symbol=result.symbol,
                name=item.name,
                alert_type=alert_type,
                message=msg,
                current_value=current,
                threshold=threshold,
                triggered_at=now,
                severity=severity,
            )
        return None

    # ── 标签管理 ──────────────────────────────────────────

    def tag(self, symbol: str, *tags: str):
        """给自选股打标签。"""
        symbol = symbol.upper()
        if symbol in self.items:
            for t in tags:
                if t not in self.items[symbol].tags:
                    self.items[symbol].tags.append(t)
            self._save()

    def untag(self, symbol: str, *tags: str):
        """移除标签。"""
        symbol = symbol.upper()
        if symbol in self.items:
            self.items[symbol].tags = [t for t in self.items[symbol].tags if t not in tags]
            self._save()

    def by_tag(self, tag: str) -> List[WatchItem]:
        """按标签筛选自选股。"""
        return [item for item in self.items.values() if tag in item.tags]

    # ── 告警配置 ──────────────────────────────────────────

    def set_alert(self, symbol: str, **alerts):
        """设置告警条件。

        示例:
            wl.set_alert("000001", pct_change_gt=5, pct_change_lt=-5)
        """
        symbol = symbol.upper()
        if symbol in self.items:
            self.items[symbol].alerts.update(alerts)
            self._save()

    def clear_alerts(self, symbol: str):
        """清除告警条件。"""
        symbol = symbol.upper()
        if symbol in self.items:
            self.items[symbol].alerts = {}
            self._save()

    # ── 持久化 ────────────────────────────────────────────

    def _load(self):
        """从 JSON 文件加载。"""
        if not os.path.isfile(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item_data in data:
                item = WatchItem(
                    symbol=item_data["symbol"],
                    name=item_data.get("name", ""),
                    added_at=item_data.get("added_at", ""),
                    tags=item_data.get("tags", []),
                    notes=item_data.get("notes", ""),
                    alerts=item_data.get("alerts", {}),
                )
                self.items[item.symbol] = item
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("自选股文件损坏: %s", e)
            self.items = {}

    def _save(self):
        """保存到 JSON 文件。"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        data = []
        for item in self.items.values():
            data.append({
                "symbol": item.symbol,
                "name": item.name,
                "added_at": item.added_at,
                "tags": item.tags,
                "notes": item.notes,
                "alerts": item.alerts,
            })

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 统计 ──────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """统计信息。"""
        tags_count = {}
        for item in self.items.values():
            for t in item.tags:
                tags_count[t] = tags_count.get(t, 0) + 1

        return {
            "total": len(self.items),
            "with_alerts": sum(1 for item in self.items.values() if item.alerts),
            "tags": tags_count,
            "path": self.path,
        }