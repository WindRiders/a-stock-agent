"""Pytest 配置。"""

import sys
import pytest
import requests
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "network: tests that require network access")


def pytest_collection_modifyitems(config, items):
    """如果无法连接东方财富 API，自动跳过网络测试。"""
    if not _check_network():
        skip_network = pytest.mark.skip(reason="网络不可用（无法连接东方财富API）")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


def _check_network() -> bool:
    try:
        r = requests.get("https://push2.eastmoney.com", timeout=5)
        return r.status_code < 500
    except Exception:
        return False