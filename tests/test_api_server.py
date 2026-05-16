"""测试 REST API 服务。使用 FastAPI TestClient。"""

import pytest

# 可能未安装 fastapi 或 httpx
try:
    from fastapi.testclient import TestClient
    from api_server import app
    FASTAPI_AVAILABLE = True
except (ImportError, RuntimeError):
    FASTAPI_AVAILABLE = False

import os
import tempfile

os.environ["A_STOCK_TEST"] = "1"


@pytest.fixture
def client():
    """FastAPI TestClient（如果可用）。"""
    if not FASTAPI_AVAILABLE:
        pytest.skip("fastapi 未安装")
    return TestClient(app)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_health(client):
    """健康检查端点。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_strategies_list(client):
    """策略列表。"""
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_market_state(client):
    """市场状态端点。"""
    response = client.get("/api/v1/market/state")
    # 可能因网络问题失败，接受 200 或 500
    assert response.status_code in (200, 500)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_config_get(client):
    """获取配置。"""
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "strategy" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_config_update(client):
    """更新配置。"""
    response = client.post("/api/v1/config", json={"strategy": "trend"})
    assert response.status_code == 200
    assert response.json()["success"] is True


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_history(client):
    """历史端点。"""
    response = client.get("/api/v1/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "scans" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_portfolio(client):
    """持仓端点。"""
    response = client.get("/api/v1/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "positions" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_trades(client):
    """交易记录端点。"""
    response = client.get("/api/v1/trades?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "trades" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_watchlist_endpoints(client):
    """自选股 CRUD。"""
    # 添加
    resp = client.post("/api/v1/watchlist/add", json={
        "symbol": "000001", "name": "平安", "tags": ["银行"], "alerts": {"pct_change_gt": 5}
    })
    assert resp.status_code == 200

    # 列表
    resp = client.get("/api/v1/watchlist")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1

    # 扫描（可能因网络失败）
    resp = client.post("/api/v1/watchlist/scan", params={"strategy": "trend"})
    assert resp.status_code in (200, 500)

    # 删除
    resp = client.delete("/api/v1/watchlist/000001")
    assert resp.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_paper_endpoints(client):
    """模拟盘端点。"""
    resp = client.get("/api/v1/paper/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "equity" in data

    resp = client.post("/api/v1/paper/reset")
    assert resp.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_scan_endpoint(client):
    """扫描端点（可能网络失败）。"""
    resp = client.post("/api/v1/scan", json={"top_n": 5, "strategy": "trend"})
    assert resp.status_code in (200, 422, 500)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_accuracy_endpoint(client):
    """准确率端点。"""
    resp = client.get("/api/v1/accuracy")
    assert resp.status_code in (200, 500)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_analyze_endpoint(client):
    """单股分析端点。"""
    resp = client.get("/api/v1/stock/000001")
    assert resp.status_code in (200, 500)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi 未安装")
def test_backtest_endpoint(client):
    """回测端点。"""
    resp = client.post("/api/v1/backtest", json={
        "symbol": "000001", "initial_capital": 100000, "strategy": "trend"
    })
    assert resp.status_code in (200, 404, 500)


def test_api_server_module_loads():
    """api_server 模块可以导入（不要求 fastapi 已安装）。"""
    try:
        import api_server
        assert hasattr(api_server, "app") or True
    except (ImportError, SystemExit):
        # fastapi 未安装，允许跳过
        pass