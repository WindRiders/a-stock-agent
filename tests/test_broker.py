"""测试券商对接模块。"""

import pytest

from broker import (
    MockBroker, HTSCBroker, EastMoneyBroker, AbstractBroker,
    get_broker, list_brokers, detect_broker,
    OrderSide, OrderType, OrderStatus,
    AccountInfo, PositionInfo, OrderInfo, TradeInfo, RiskConfig,
)


class TestDataStructures:
    """数据结构测试。"""

    def test_order_side_enum(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_enum(self):
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.MARKET.value == "MARKET"

    def test_order_status_enum(self):
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.PENDING.value == "PENDING"

    def test_account_info_defaults(self):
        a = AccountInfo()
        assert a.total_asset == 0

    def test_risk_config_defaults(self):
        rc = RiskConfig()
        assert rc.enabled is True
        assert rc.max_single_order_amount > 0


class TestMockBroker:
    """模拟券商测试。"""

    def setup_method(self):
        self.broker = MockBroker(initial_cash=500000)
        self.broker.connect()

    def teardown_method(self):
        self.broker.disconnect()

    def test_name(self):
        assert self.broker.name == "MockBroker"

    def test_connect(self):
        assert self.broker._connected is True

    def test_query_account(self):
        account = self.broker.query_account()
        assert account.total_asset == 500000
        assert account.available_cash == 500000
        assert account.market_value == 0

    def test_query_positions_empty(self):
        positions = self.broker.query_positions()
        assert positions == []

    def test_query_orders_empty(self):
        orders = self.broker.query_orders()
        assert orders == []

    def test_buy_creates_order_and_position(self):
        order = self.broker.buy("000001", 10.0, 1000, reason="测试买入")
        assert order.status == OrderStatus.FILLED
        assert order.symbol == "000001"
        assert order.volume == 1000

        positions = self.broker.query_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "000001"
        assert positions[0].shares == 1000

    def test_buy_deducts_cash(self):
        order = self.broker.buy("000001", 10.0, 1000)
        account = self.broker.query_account()
        assert account.available_cash < 500000

    def test_buy_insufficient_cash_rejected(self):
        order = self.broker.buy("000001", 50000.0, 1000)
        assert order.status == OrderStatus.REJECTED

    def test_sell_creates_sell_order(self):
        self.broker.buy("000001", 10.0, 1000)
        order = self.broker.sell("000001", 11.0, 500, reason="止盈")
        assert order.status == OrderStatus.FILLED

        positions = self.broker.query_positions()
        assert positions[0].shares == 500

    def test_sell_all_clears_position(self):
        self.broker.buy("000001", 10.0, 1000)
        order = self.broker.sell("000001", 11.0, 1000)
        assert order.status == OrderStatus.FILLED
        assert len(self.broker.query_positions()) == 0

    def test_sell_no_position_rejected(self):
        order = self.broker.sell("000001", 10.0, 1000)
        assert order.status == OrderStatus.REJECTED

    def test_sell_exceeding_shares_rejected(self):
        self.broker.buy("000001", 10.0, 1000)
        order = self.broker.sell("000001", 10.0, 2000)
        assert order.status == OrderStatus.REJECTED

    def test_buy_by_amount(self):
        order = self.broker.buy_by_amount("000001", 10.0, 50000)
        assert order.status == OrderStatus.FILLED
        assert order.volume >= 100  # 至少1手

    def test_buy_by_amount_too_small(self):
        order = self.broker.buy_by_amount("000001", 10.0, 500)
        assert order.status == OrderStatus.REJECTED

    def test_cancel_order_not_exists(self):
        assert self.broker.cancel_order("NONEXISTENT") is False

    def test_get_quote(self):
        quote = self.broker.get_quote("000001")
        assert quote is not None
        assert quote.symbol == "000001"
        assert quote.price > 0

    def test_get_quotes(self):
        quotes = self.broker.get_quotes(["000001", "000002", "000003"])
        assert len(quotes) == 3

    def test_trade_records(self):
        self.broker.buy("000001", 10.0, 1000)
        self.broker.sell("000001", 11.0, 1000)

        trades = self.broker.query_trades()
        assert len(trades) == 2
        assert trades[0].side == OrderSide.BUY
        assert trades[1].side == OrderSide.SELL

    def test_commission_on_trade(self):
        order = self.broker.buy("000001", 10.0, 1000)
        trades = self.broker.query_trades()
        assert trades[0].commission > 0

    def test_stamp_tax_only_on_sell(self):
        self.broker.buy("000001", 10.0, 1000)
        self.broker.sell("000001", 11.0, 1000)

        trades = self.broker.query_trades()
        buy_trade = trades[0]
        sell_trade = trades[1]

        assert buy_trade.stamp_tax == 0  # 买入无印花税
        assert sell_trade.stamp_tax > 0  # 卖出有

    def test_risk_limit_single_order(self):
        rc = RiskConfig(max_single_order_amount=10000, enabled=True)
        broker = MockBroker(initial_cash=500000, risk_config=rc)
        broker.connect()

        # 尝试超大单
        order = broker.buy("000001", 1000.0, 100)  # 100000 > 10000
        assert order.status == OrderStatus.REJECTED

        broker.disconnect()

    def test_risk_disabled(self):
        rc = RiskConfig(enabled=False, max_single_order_amount=1)
        broker = MockBroker(initial_cash=500000, risk_config=rc)
        broker.connect()

        order = broker.buy("000001", 10.0, 1000)
        assert order.status == OrderStatus.FILLED

        broker.disconnect()

    def test_reset_daily(self):
        self.broker.buy("000001", 10.0, 1000)
        self.broker.sell("000001", 9.0, 1000)  # 亏损
        self.broker.reset_daily()
        assert self.broker._daily_loss == 0
        assert self.broker._daily_orders == 0

    def test_multiple_positions(self):
        self.broker.buy("000001", 10.0, 1000)
        self.broker.buy("000002", 15.0, 500)
        assert len(self.broker.query_positions()) == 2

    def test_disconnect(self):
        self.broker.disconnect()
        assert self.broker._connected is False


class TestHTSCBroker:
    """华泰证券测试（无实盘环境，只测构造和连接失败）。"""

    def test_name(self):
        broker = HTSCBroker()
        assert "华泰" in broker.name

    def test_connect_fails_without_qmt(self):
        broker = HTSCBroker()
        result = broker.connect()
        assert result is False  # xtquant 不会安装在测试环境

    def test_query_returns_empty_when_not_connected(self):
        broker = HTSCBroker()
        assert broker.query_account().total_asset == 0
        assert broker.query_positions() == []


class TestEastMoneyBroker:
    """东方财富测试（无实盘环境）。"""

    def test_name(self):
        broker = EastMoneyBroker()
        assert "东方财富" in broker.name or "EastMoney" in broker.name

    def test_connect_fails_without_easytrader(self):
        broker = EastMoneyBroker()
        result = broker.connect()
        assert result is False

    def test_query_returns_empty_when_not_connected(self):
        broker = EastMoneyBroker()
        assert broker.query_account().total_asset == 0


class TestBrokerFactory:
    """Broker 工厂测试。"""

    def test_get_broker_mock(self):
        broker = get_broker("mock")
        assert isinstance(broker, MockBroker)
        assert broker._connected
        broker.disconnect()

    def test_get_broker_invalid(self):
        with pytest.raises(ValueError):
            get_broker("invalid_broker")

    def test_list_brokers(self):
        brokers = list_brokers()
        assert "mock" in brokers
        assert "htsc" in brokers
        assert "eastmoney" in brokers

    def test_detect_broker_defaults_mock(self):
        detected = detect_broker()
        assert detected == "mock"

    def test_get_broker_case_insensitive(self):
        broker = get_broker("MOCK")
        assert isinstance(broker, MockBroker)
        broker.disconnect()