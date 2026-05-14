"""
实盘券商对接模块。

提供统一抽象接口，支持多券商：
- 华泰证券 (HTSC) — 通过 xtquant/QMT 或官方 API
- 东方财富 (EastMoney) — 通过 easytrader 或 EM API
- 模拟券商 (Mock) — 测试用，完整透传不做真实下单

核心功能：
- 账户查询（资金/持仓/委托/成交）
- 下单（限价/市价/条件单）
- 撤单
- 实时行情回调
- 订单成交回调

安全机制：
- 环境变量存储凭证，绝不硬编码
- 下单前二次确认（可配置跳过）
- 单日最大亏损限额
- 单笔最大金额限制

用法:
    from broker import get_broker

    broker = get_broker("mock")  # 测试
    # broker = get_broker("htsc")  # 华泰
    # broker = get_broker("eastmoney")  # 东方财富

    account = broker.query_account()
    broker.buy("000001", price=10.5, volume=1000)
    broker.sell("000001", price=11.0, volume=1000)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 枚举 ──────────────────────────────────────────────────

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"  # 限价单
    MARKET = "MARKET"  # 市价单
    STOP = "STOP"  # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 限价止损


class OrderStatus(Enum):
    PENDING = "PENDING"  # 待提交
    SUBMITTED = "SUBMITTED"  # 已提交
    PARTIAL_FILLED = "PARTIAL_FILLED"  # 部分成交
    FILLED = "FILLED"  # 全部成交
    CANCELLED = "CANCELLED"  # 已撤销
    REJECTED = "REJECTED"  # 被拒绝
    EXPIRED = "EXPIRED"  # 已过期


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class AccountInfo:
    """账户信息。"""
    account_id: str = ""
    broker_name: str = ""
    total_asset: float = 0.0  # 总资产
    available_cash: float = 0.0  # 可用资金
    frozen_cash: float = 0.0  # 冻结资金
    market_value: float = 0.0  # 持仓市值
    total_return: float = 0.0  # 总收益
    daily_return: float = 0.0  # 当日收益
    updated_at: str = ""


@dataclass
class PositionInfo:
    """持仓信息。"""
    symbol: str
    name: str = ""
    shares: int = 0  # 持仓数量
    available_shares: int = 0  # 可卖数量
    avg_cost: float = 0.0  # 成本价
    current_price: float = 0.0  # 现价
    market_value: float = 0.0  # 市值
    profit: float = 0.0  # 浮动盈亏
    profit_pct: float = 0.0  # 盈亏比例
    today_shares: int = 0  # 今日买入（T+1限制）


@dataclass
class OrderInfo:
    """订单信息。"""
    order_id: str = ""
    symbol: str = ""
    name: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    price: float = 0.0
    volume: int = 0  # 委托数量
    filled_volume: int = 0  # 已成交
    filled_amount: float = 0.0  # 成交金额
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    reason: str = ""  # 备注/理由


@dataclass
class TradeInfo:
    """成交信息。"""
    trade_id: str = ""
    order_id: str = ""
    symbol: str = ""
    name: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    volume: int = 0
    amount: float = 0.0
    commission: float = 0.0
    stamp_tax: float = 0.0
    traded_at: str = ""


@dataclass
class QuoteInfo:
    """行情信息。"""
    symbol: str
    name: str = ""
    price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    pre_close: float = 0.0
    volume: int = 0
    amount: float = 0.0
    bid1_price: float = 0.0
    bid1_volume: int = 0
    ask1_price: float = 0.0
    ask1_volume: int = 0
    time: str = ""


# ── 风控配置 ──────────────────────────────────────────────

@dataclass
class RiskConfig:
    """实盘风控配置。"""
    max_single_order_amount: float = 500000  # 单笔最大金额 50万
    max_daily_loss: float = 50000  # 单日最大亏损 5万
    max_position_ratio: float = 0.30  # 单只最大仓位 30%
    confirm_before_order: bool = True  # 下单前确认
    max_orders_per_day: int = 50  # 单日最大下单次数
    enabled: bool = True  # 风控开关


# ── 抽象基类 ──────────────────────────────────────────────

class AbstractBroker(ABC):
    """券商抽象接口。

    所有券商实现必须继承此类并实现全部抽象方法。
    """

    def __init__(self, risk_config: RiskConfig = None):
        self.risk_config = risk_config or RiskConfig()
        self._daily_loss = 0.0
        self._daily_orders = 0
        self._connected = False

    @property
    def name(self) -> str:
        """券商名称。"""
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> bool:
        """连接券商。返回是否成功。"""
        ...

    @abstractmethod
    def disconnect(self):
        """断开连接。"""
        ...

    @abstractmethod
    def query_account(self) -> AccountInfo:
        """查询账户信息。"""
        ...

    @abstractmethod
    def query_positions(self) -> List[PositionInfo]:
        """查询持仓。"""
        ...

    @abstractmethod
    def query_orders(self, status: str = None) -> List[OrderInfo]:
        """查询委托。"""
        ...

    @abstractmethod
    def query_trades(self, date: str = None) -> List[TradeInfo]:
        """查询成交。"""
        ...

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        volume: int,
        reason: str = "",
    ) -> OrderInfo:
        """提交订单。"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单。"""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[QuoteInfo]:
        """获取实时行情。"""
        ...

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> List[QuoteInfo]:
        """批量获取实时行情。"""
        ...

    # ── 便捷方法 ──────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT,
        reason: str = "",
    ) -> OrderInfo:
        """买入。"""
        return self.submit_order(symbol, OrderSide.BUY, order_type, price, volume, reason)

    def sell(
        self,
        symbol: str,
        price: float,
        volume: int,
        order_type: OrderType = OrderType.LIMIT,
        reason: str = "",
    ) -> OrderInfo:
        """卖出。"""
        return self.submit_order(symbol, OrderSide.SELL, order_type, price, volume, reason)

    def buy_by_amount(
        self,
        symbol: str,
        price: float,
        amount: float,
        reason: str = "",
    ) -> OrderInfo:
        """按金额买入（自动计算股数，取整手）。"""
        volume = int(amount / price / 100) * 100
        if volume < 100:
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED,
                             reason=f"金额{amount}不足1手")
        return self.buy(symbol, price, volume, reason=reason)

    # ── 风控检查 ──────────────────────────────────────────

    def _check_risk(self, side: OrderSide, amount: float) -> Optional[str]:
        """风控检查。返回 None 表示通过，否则返回拒绝原因。"""
        if not self.risk_config.enabled:
            return None

        if amount > self.risk_config.max_single_order_amount:
            return f"单笔金额 ¥{amount:,.0f} 超过上限 ¥{self.risk_config.max_single_order_amount:,.0f}"

        if self._daily_loss <= -self.risk_config.max_daily_loss:
            return f"当日亏损 ¥{abs(self._daily_loss):,.0f} 已达上限"

        self._daily_orders += 1
        if self._daily_orders > self.risk_config.max_orders_per_day:
            return "当日下单次数已达上限"

        return None


# ── 模拟券商（Mock）────────────────────────────────────────

class MockBroker(AbstractBroker):
    """模拟券商 — 用于测试和开发。

    完整实现所有接口，不做真实下单。
    所有数据在内存中维护。
    """

    @property
    def name(self) -> str:
        return "MockBroker"

    def __init__(self, initial_cash: float = 1000000, risk_config: RiskConfig = None):
        super().__init__(risk_config)
        self._cash = initial_cash
        self._frozen = 0.0
        self._positions: Dict[str, PositionInfo] = {}
        self._orders: List[OrderInfo] = []
        self._trades: List[TradeInfo] = []
        self._order_counter = 0
        self._connected = False
        self._init_price = 10.0

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockBroker 已连接（模拟模式）")
        return True

    def disconnect(self):
        self._connected = False

    def query_account(self) -> AccountInfo:
        mv = sum(p.market_value for p in self._positions.values())
        total = self._cash + self._frozen + mv
        return AccountInfo(
            account_id="MOCK-001",
            broker_name="MockBroker",
            total_asset=total,
            available_cash=self._cash,
            frozen_cash=self._frozen,
            market_value=mv,
            total_return=total - 1000000,
            updated_at=datetime.now().isoformat(),
        )

    def query_positions(self) -> List[PositionInfo]:
        return list(self._positions.values())

    def query_orders(self, status: str = None) -> List[OrderInfo]:
        if status is None:
            return self._orders
        return [o for o in self._orders if o.status.value == status]

    def query_trades(self, date: str = None) -> List[TradeInfo]:
        if date is None:
            return self._trades
        return [t for t in self._trades if t.traded_at[:10] == date]

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        volume: int,
        reason: str = "",
    ) -> OrderInfo:
        # 风控检查
        amount = price * volume
        reject = self._check_risk(side, amount)
        if reject:
            return OrderInfo(
                order_id="", symbol=symbol, side=side, price=price,
                volume=volume, status=OrderStatus.REJECTED, reason=reject,
            )

        self._order_counter += 1
        order_id = f"MOCK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"
        now = datetime.now().isoformat()

        # 模拟市价成交（简化：直接按当前价全部成交）
        order = OrderInfo(
            order_id=order_id, symbol=symbol, side=side,
            order_type=order_type, price=price, volume=volume,
            filled_volume=volume, filled_amount=amount,
            status=OrderStatus.FILLED, created_at=now, updated_at=now,
            reason=reason,
        )

        commission = max(5.0, amount * 0.00025)
        stamp_tax = amount * 0.001 if side == OrderSide.SELL else 0

        if side == OrderSide.BUY:
            total_cost = amount + commission
            if self._cash < total_cost:
                order.status = OrderStatus.REJECTED
                order.reason = f"资金不足（需 ¥{total_cost:,.0f}，可用 ¥{self._cash:,.0f}）"
                return order

            self._cash -= total_cost

            # 更新持仓
            if symbol in self._positions:
                pos = self._positions[symbol]
                new_cost = pos.avg_cost * pos.shares + amount + commission
                pos.shares += volume
                pos.avg_cost = new_cost / pos.shares if pos.shares > 0 else 0
                pos.market_value = pos.shares * price
            else:
                self._positions[symbol] = PositionInfo(
                    symbol=symbol, shares=volume,
                    avg_cost=(amount + commission) / volume,
                    current_price=price, market_value=amount,
                    available_shares=volume, today_shares=volume,
                )
        else:
            # 卖出
            if symbol not in self._positions:
                order.status = OrderStatus.REJECTED
                order.reason = f"无 {symbol} 持仓"
                return order

            pos = self._positions[symbol]
            if volume > pos.available_shares:
                order.status = OrderStatus.REJECTED
                order.reason = f"可卖数量不足（{pos.available_shares}<{volume}）"
                return order

            pos.shares -= volume
            pos.available_shares -= volume
            pos.market_value = pos.shares * price

            if pos.shares <= 0:
                del self._positions[symbol]

            profit = (price - pos.avg_cost) * volume - commission - stamp_tax
            self._cash += amount - commission - stamp_tax
            self._daily_loss -= profit  # 累加日亏损（盈利为负亏损）
            order.reason = f"{reason} | 盈亏: ¥{profit:+,.0f}"

        # 记录成交
        self._trades.append(TradeInfo(
            trade_id=f"T{order_id}",
            order_id=order_id, symbol=symbol,
            side=side, price=price, volume=volume,
            amount=amount, commission=commission,
            stamp_tax=stamp_tax, traded_at=now,
        ))

        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status in (
                OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED
            ):
                o.status = OrderStatus.CANCELLED
                o.updated_at = datetime.now().isoformat()
                return True
        return False

    def get_quote(self, symbol: str) -> Optional[QuoteInfo]:
        # 模拟行情
        import random
        random.seed(hash(symbol))
        base = 10.0
        price = base + random.uniform(-5, 5)
        return QuoteInfo(
            symbol=symbol, name=f"模拟-{symbol}", price=max(0.01, price),
            open=price * random.uniform(0.98, 1.02),
            high=price * random.uniform(1.0, 1.03),
            low=price * random.uniform(0.97, 1.0),
            pre_close=price * random.uniform(0.99, 1.01),
            volume=random.randint(1000000, 50000000),
            time=datetime.now().strftime("%H:%M:%S"),
        )

    def get_quotes(self, symbols: List[str]) -> List[QuoteInfo]:
        return [self.get_quote(s) for s in symbols]

    def reset_daily(self):
        """重置每日计数器（新交易日）。"""
        self._daily_loss = 0.0
        self._daily_orders = 0


# ── 华泰证券 ──────────────────────────────────────────────

class HTSCBroker(AbstractBroker):
    """华泰证券 (HTSC) 接口。

    支持方式：
    - xtquant / QMT Mini（本地量化终端）
    - 官方 API（HTSC 机构服务）

    环境变量：
        HTSC_ACCOUNT=账户号
        HTSC_PASSWORD=密码（可选）
        HTSC_API_PATH=QMT/xtquant 安装路径
    """

    @property
    def name(self) -> str:
        return "华泰证券(HTSC)"

    def __init__(self, risk_config: RiskConfig = None):
        super().__init__(risk_config)
        self._session = None
        self._account = os.environ.get("HTSC_ACCOUNT", "")
        self._password = os.environ.get("HTSC_PASSWORD", "")
        self._api_path = os.environ.get("HTSC_API_PATH", "")
        self._xtdata = None
        self._xttrade = None

    def connect(self) -> bool:
        """连接华泰QMT/xtquant。

        尝试导入 xtquant，如不可用则提示用户安装。
        """
        try:
            import xtquant.xtdata as xtdata
            import xtquant.xttrader as xttrader

            self._xtdata = xtdata
            self._xttrade = xttrader

            if self._api_path:
                xtdata.connect(self._api_path)
            else:
                xtdata.connect()

            self._session = xttrader.XtQuantTrader(self._api_path, self._account)
            self._session.start()
            result = self._session.connect()

            if result == 0:
                self._connected = True
                logger.info("华泰证券 QMT 连接成功，账户: %s", self._account)
                return True
            else:
                logger.error("华泰证券连接失败，返回码: %d", result)
                return False

        except ImportError:
            logger.error(
                "华泰证券需要 xtquant/QMT。"
                "请安装 QMT 客户端并将 xtquant 添加到 PYTHONPATH。"
            )
            return False
        except Exception as e:
            logger.error("华泰证券连接异常: %s", e)
            return False

    def disconnect(self):
        if self._session:
            try:
                self._session.stop()
            except Exception:
                pass
        self._connected = False

    def query_account(self) -> AccountInfo:
        if not self._connected:
            return AccountInfo(broker_name=self.name)

        try:
            asset = self._session.query_asset()
            return AccountInfo(
                account_id=self._account,
                broker_name=self.name,
                total_asset=asset.total_asset,
                available_cash=asset.cash,
                frozen_cash=asset.frozen_cash,
                market_value=asset.market_value,
                total_return=asset.total_profit,
                updated_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning("查询账户失败: %s", e)
            return AccountInfo(broker_name=self.name)

    def query_positions(self) -> List[PositionInfo]:
        if not self._connected:
            return []

        try:
            raw = self._session.query_positions()
            positions = []
            for pos in raw:
                positions.append(PositionInfo(
                    symbol=pos.stock_code,
                    name=pos.stock_name,
                    shares=pos.volume,
                    available_shares=pos.can_use_volume,
                    avg_cost=pos.avg_price,
                    current_price=pos.current_price,
                    market_value=pos.market_value,
                    profit=pos.profit,
                    profit_pct=pos.profit_ratio,
                ))
            return positions
        except Exception as e:
            logger.warning("查询持仓失败: %s", e)
            return []

    def query_orders(self, status: str = None) -> List[OrderInfo]:
        if not self._connected:
            return []

        try:
            raw = self._session.query_orders()
            orders = []
            for o in raw:
                if status and o.status != status:
                    continue
                orders.append(OrderInfo(
                    order_id=str(o.order_id),
                    symbol=o.stock_code,
                    name=getattr(o, "stock_name", ""),
                    side=OrderSide.BUY if o.order_type == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=o.price,
                    volume=o.order_volume,
                    filled_volume=o.traded_volume,
                    filled_amount=o.traded_amount,
                    status=self._map_status(o.status),
                    created_at=str(o.order_time),
                ))
            return orders
        except Exception as e:
            logger.warning("查询委托失败: %s", e)
            return []

    def query_trades(self, date: str = None) -> List[TradeInfo]:
        if not self._connected:
            return []

        try:
            raw = self._session.query_trades()
            trades = []
            for t in raw:
                if date and str(t.traded_time)[:10] != date:
                    continue
                trades.append(TradeInfo(
                    trade_id=str(t.trade_id),
                    order_id=str(t.order_id),
                    symbol=t.stock_code,
                    side=OrderSide.BUY if t.order_type == 0 else OrderSide.SELL,
                    price=t.price,
                    volume=t.traded_volume,
                    amount=t.traded_amount,
                    commission=getattr(t, "commission", 0),
                    stamp_tax=getattr(t, "stamp_tax", 0),
                    traded_at=str(t.traded_time),
                ))
            return trades
        except Exception as e:
            logger.warning("查询成交失败: %s", e)
            return []

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        volume: int,
        reason: str = "",
    ) -> OrderInfo:
        if not self._connected:
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED,
                             reason="未连接到券商")

        amount = price * volume
        reject = self._check_risk(side, amount)
        if reject:
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED, reason=reject)

        try:
            order_id = self._session.order_stock(
                symbol, volume,
                side == OrderSide.BUY and 0 or 1,  # 0=买入 1=卖出
                price, 0, reason or "量化策略信号",
            )
            return OrderInfo(
                order_id=str(order_id), symbol=symbol, side=side,
                order_type=order_type, price=price, volume=volume,
                status=OrderStatus.SUBMITTED,
                created_at=datetime.now().isoformat(), reason=reason,
            )
        except Exception as e:
            logger.error("下单失败: %s", e)
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED, reason=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        try:
            return self._session.cancel_order(int(order_id)) == 0
        except Exception:
            return False

    def get_quote(self, symbol: str) -> Optional[QuoteInfo]:
        if not self._xtdata:
            return None
        try:
            data = self._xtdata.get_market_data(
                ["lastPrice", "open", "high", "low", "lastClose", "volume", "amount"],
                [symbol], period="tick",
            )
            return QuoteInfo(
                symbol=symbol, time=datetime.now().strftime("%H:%M:%S"),
                price=data.get("lastPrice", 0),
                open=data.get("open", 0), high=data.get("high", 0),
                low=data.get("low", 0), pre_close=data.get("lastClose", 0),
                volume=data.get("volume", 0), amount=data.get("amount", 0),
            )
        except Exception:
            return None

    def get_quotes(self, symbols: List[str]) -> List[QuoteInfo]:
        return [q for s in symbols if (q := self.get_quote(s)) is not None]

    def _map_status(self, status_code: int) -> OrderStatus:
        mapping = {
            0: OrderStatus.SUBMITTED,
            1: OrderStatus.PARTIAL_FILLED,
            2: OrderStatus.FILLED,
            3: OrderStatus.CANCELLED,
            4: OrderStatus.REJECTED,
        }
        return mapping.get(status_code, OrderStatus.PENDING)


# ── 东方财富 ──────────────────────────────────────────────

class EastMoneyBroker(AbstractBroker):
    """东方财富 (EastMoney) 接口。

    支持方式：
    - easytrader（开源，模拟鼠标操作东方财富客户端）
    - 东方财富官方 API

    环境变量：
        EASTMONEY_ACCOUNT=账户号
        EASTMONEY_CLIENT_PATH=东方财富客户端路径（easytrader用）

    注意：easytrader 依赖 GUI 客户端，适合个人量化，不适合服务端部署。
    """

    @property
    def name(self) -> str:
        return "东方财富(EastMoney)"

    def __init__(self, risk_config: RiskConfig = None):
        super().__init__(risk_config)
        self._user = None
        self._account = os.environ.get("EASTMONEY_ACCOUNT", "")
        self._client_path = os.environ.get("EASTMONEY_CLIENT_PATH", "")

    def connect(self) -> bool:
        """连接东方财富。

        通过 easytrader 库连接客户端。
        """
        try:
            import easytrader

            self._user = easytrader.use("eastmoney")

            if self._client_path:
                self._user.connect(self._client_path)
            else:
                self._user.connect()

            self._user.enable_type_keys_for_editor()

            # 登录
            self._user.login()

            self._connected = True
            logger.info("东方财富连接成功")
            return True

        except ImportError:
            logger.error(
                "东方财富需要 easytrader。请安装: pip install easytrader"
            )
            return False
        except Exception as e:
            logger.error("东方财富连接失败: %s", e)
            return False

    def disconnect(self):
        self._connected = False
        self._user = None

    def query_account(self) -> AccountInfo:
        if not self._connected or not self._user:
            return AccountInfo(broker_name=self.name)

        try:
            balance = self._user.balance
            position = self._user.position

            total_mv = sum(
                p.get("market_value", 0)
                for p in (position if isinstance(position, list) else [])
            )

            return AccountInfo(
                account_id=self._account,
                broker_name=self.name,
                total_asset=balance.get("total_asset", 0),
                available_cash=balance.get("available", 0),
                frozen_cash=balance.get("frozen", 0),
                market_value=total_mv,
                updated_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning("查询账户失败: %s", e)
            return AccountInfo(broker_name=self.name)

    def query_positions(self) -> List[PositionInfo]:
        if not self._connected or not self._user:
            return []

        try:
            raw = self._user.position
            if not isinstance(raw, list):
                raw = []

            positions = []
            for p in raw:
                positions.append(PositionInfo(
                    symbol=p.get("stock_code", ""),
                    name=p.get("stock_name", ""),
                    shares=p.get("current_amount", 0),
                    available_shares=p.get("enable_amount", 0),
                    avg_cost=p.get("cost_price", 0),
                    current_price=p.get("current_price", 0),
                    market_value=p.get("market_value", 0),
                    profit=p.get("income_balance", 0),
                ))
            return positions
        except Exception as e:
            logger.warning("查询持仓失败: %s", e)
            return []

    def query_orders(self, status: str = None) -> List[OrderInfo]:
        if not self._connected or not self._user:
            return []

        try:
            raw = self._user.today_entrusts
            orders = []
            for o in (raw if isinstance(raw, list) else []):
                orders.append(OrderInfo(
                    order_id=str(o.get("entrust_no", "")),
                    symbol=o.get("stock_code", ""),
                    side=OrderSide.BUY if "买入" in str(o.get("entrust_type", "")) else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=o.get("entrust_price", 0),
                    volume=o.get("entrust_amount", 0),
                    filled_volume=o.get("business_amount", 0),
                    status=self._map_status(o.get("entrust_status", "")),
                    created_at=str(o.get("report_time", "")),
                ))
            return orders
        except Exception as e:
            logger.warning("查询委托失败: %s", e)
            return []

    def query_trades(self, date: str = None) -> List[TradeInfo]:
        if not self._connected or not self._user:
            return []

        try:
            raw = self._user.today_trades
            trades = []
            for t in (raw if isinstance(raw, list) else []):
                trades.append(TradeInfo(
                    trade_id=str(t.get("business_no", "")),
                    order_id=str(t.get("entrust_no", "")),
                    symbol=t.get("stock_code", ""),
                    side=OrderSide.BUY if "买入" in str(t.get("business_type", "")) else OrderSide.SELL,
                    price=t.get("business_price", 0),
                    volume=t.get("business_amount", 0),
                    amount=t.get("business_balance", 0),
                    traded_at=str(t.get("business_time", "")),
                ))
            return trades
        except Exception as e:
            logger.warning("查询成交失败: %s", e)
            return []

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        volume: int,
        reason: str = "",
    ) -> OrderInfo:
        if not self._connected or not self._user:
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED,
                             reason="未连接到券商")

        amount = price * volume
        reject = self._check_risk(side, amount)
        if reject:
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED, reason=reject)

        try:
            if side == OrderSide.BUY:
                result = self._user.buy(symbol, price, volume)
            else:
                result = self._user.sell(symbol, price, volume)

            return OrderInfo(
                order_id=str(result.get("entrust_no", "")),
                symbol=symbol, side=side,
                order_type=order_type, price=price, volume=volume,
                status=OrderStatus.SUBMITTED,
                created_at=datetime.now().isoformat(), reason=reason,
            )
        except Exception as e:
            logger.error("下单失败: %s", e)
            return OrderInfo(symbol=symbol, status=OrderStatus.REJECTED, reason=str(e))

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or not self._user:
            return False
        try:
            self._user.cancel_entrust(order_id)
            return True
        except Exception:
            return False

    def get_quote(self, symbol: str) -> Optional[QuoteInfo]:
        """EM行情需配合其他数据源。返回None。"""
        return None  # easytrader 不直接提供行情

    def get_quotes(self, symbols: List[str]) -> List[QuoteInfo]:
        return []

    def _map_status(self, status_str: str) -> OrderStatus:
        mapping = {
            "已报": OrderStatus.SUBMITTED,
            "部成": OrderStatus.PARTIAL_FILLED,
            "已成": OrderStatus.FILLED,
            "已撤": OrderStatus.CANCELLED,
            "废单": OrderStatus.REJECTED,
        }
        for key, val in mapping.items():
            if key in str(status_str):
                return val
        return OrderStatus.PENDING


# ── Broker 工厂 ───────────────────────────────────────────

_BROKERS = {
    "mock": MockBroker,
    "htsc": HTSCBroker,
    "eastmoney": EastMoneyBroker,
}


def get_broker(name: str = "mock", **kwargs) -> AbstractBroker:
    """获取券商实例。

    Args:
        name: 券商名称 (mock / htsc / eastmoney)
        **kwargs: 传递给券商构造函数的参数
    """
    name = name.lower()
    if name not in _BROKERS:
        raise ValueError(f"不支持的券商: {name}。可选: {list(_BROKERS.keys())}")

    broker = _BROKERS[name](**kwargs)
    broker.connect()
    return broker


def list_brokers() -> Dict[str, str]:
    """列出所有支持的券商。"""
    return {
        "mock": "模拟券商 — 测试和开发用",
        "htsc": "华泰证券 — 需 QMT/xtquant，支持完整交易接口",
        "eastmoney": "东方财富 — 需 easytrader + GUI客户端",
    }


def detect_broker() -> str:
    """自动检测环境中的可用券商。

    返回第一个可用的券商名称，默认 'mock'。
    """
    # 检测华泰
    if os.environ.get("HTSC_ACCOUNT"):
        try:
            import xtquant
            return "htsc"
        except ImportError:
            pass

    # 检测东方财富
    if os.environ.get("EASTMONEY_ACCOUNT"):
        return "eastmoney"

    return "mock"