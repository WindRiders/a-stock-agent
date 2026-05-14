"""配置文件持久化模块。

支持 YAML 配置文件读写、默认值生成、CLI 配置管理。
配置文件位置：~/.a-stock-agent/config.yaml
"""

import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agent.config import AgentConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.a-stock-agent")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.yaml")
BACKUP_DIR = os.path.join(DEFAULT_CONFIG_DIR, "backups")


def _default_config() -> Dict[str, Any]:
    """生成默认配置字典。"""
    cfg = AgentConfig()
    return {
        "version": 1,
        "last_updated": datetime.now().isoformat(),
        "strategy": cfg.strategy,
        "risk": {
            "max_positions": cfg.max_positions,
            "max_position_pct": cfg.max_position_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
        },
        "scan": {
            "top_n": cfg.scan_top_n,
        },
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
        "notifications": {
            "enabled": False,
            "channels": {
                "telegram": {"enabled": False, "chat_id": None},
                "discord": {"enabled": False, "webhook_url": None},
                "wecom": {"enabled": False, "webhook_url": None},
                "webhook": {"enabled": False, "url": None},
            },
            "schedule": {
                "daily_report_time": "15:30",
                "only_trade_days": True,
            },
        },
    }


class ConfigManager:
    """配置文件管理器。

    Usage:
        mgr = ConfigManager()
        cfg = mgr.load()         # 加载配置 -> AgentConfig
        mgr.save(cfg)            # 保存配置
        mgr.reset()              # 重置为默认
        mgr.list_backups()       # 列出备份
        mgr.restore("20250101_120000")  # 恢复备份
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)

    def exists(self) -> bool:
        """检查配置文件是否存在。"""
        return os.path.isfile(self.config_path)

    def load(self) -> AgentConfig:
        """加载配置文件，不存在时创建默认配置。"""
        if not self.exists():
            logger.info("配置文件不存在，创建默认配置: %s", self.config_path)
            self._write_default()
            return AgentConfig()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not data:  # 空文件或无有效内容
                self._write_default()
                return AgentConfig()
        except Exception as e:
            logger.warning("配置文件损坏，备份并重置: %s", e)
            self._backup_corrupted()
            self._write_default()
            return AgentConfig()

        return self._dict_to_config(data)

    def save(self, config: AgentConfig, backup: bool = True):
        """保存配置到文件。

        Args:
            config: AgentConfig 对象
            backup: 是否在保存前备份
        """
        if backup and self.exists():
            self._backup()

        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "strategy": config.strategy,
            "risk": {
                "max_positions": config.max_positions,
                "max_position_pct": config.max_position_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
            },
            "scan": {
                "top_n": config.scan_top_n,
            },
            "llm": {
                "enabled": config.llm_enabled,
                "provider": config.llm_provider,
                "model": config.llm_model,
                "base_url": config.llm_base_url,
            },
            "data": {
                "cache_days": config.cache_days,
                "realtime_enabled": config.realtime_enabled,
            },
        }

        # 保留已有的通知配置
        existing = {}
        if self.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                pass

        if "notifications" in existing:
            data["notifications"] = existing["notifications"]

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, indent=2)

        logger.info("配置已保存: %s", self.config_path)

    def reset(self):
        """重置配置文件为默认值。"""
        self._backup()
        self._write_default()

    def show(self) -> str:
        """显示当前配置内容。"""
        if not self.exists():
            return "配置文件不存在"
        with open(self.config_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_path(self) -> str:
        return self.config_path

    # ── 备份管理 ──────────────────────────────────────────

    def list_backups(self) -> list:
        """列出所有备份。"""
        if not os.path.isdir(BACKUP_DIR):
            return []
        backups = []
        for f in sorted(os.listdir(BACKUP_DIR), reverse=True):
            path = os.path.join(BACKUP_DIR, f)
            size = os.path.getsize(path)
            backups.append({"filename": f, "path": path, "size": size})
        return backups

    def restore(self, filename: str) -> AgentConfig:
        """从备份恢复配置。"""
        backup_path = os.path.join(BACKUP_DIR, filename)
        if not os.path.isfile(backup_path):
            raise FileNotFoundError(f"备份文件不存在: {filename}")

        with open(backup_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 覆盖当前配置
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, indent=2)

        return self._dict_to_config(data)

    # ── 内部 ──────────────────────────────────────────────

    def _write_default(self):
        data = _default_config()
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, indent=2)

    def _backup(self):
        """备份当前配置。"""
        if not self.exists():
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_{ts}.yaml"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        shutil.copy2(self.config_path, backup_path)

        # 保留最近 20 个备份
        self._cleanup_backups(20)

    def _backup_corrupted(self):
        """备份损坏的配置文件。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"corrupted_{ts}.yaml"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        try:
            shutil.copy2(self.config_path, backup_path)
            logger.info("损坏配置已备份: %s", backup_path)
        except Exception:
            pass

    def _cleanup_backups(self, keep: int):
        """清理旧备份，只保留最近 N 个。"""
        if not os.path.isdir(BACKUP_DIR):
            return
        backups = sorted(
            [f for f in os.listdir(BACKUP_DIR) if f.startswith("config_")],
            reverse=True,
        )
        for old in backups[keep:]:
            try:
                os.remove(os.path.join(BACKUP_DIR, old))
            except Exception:
                pass

    def _dict_to_config(self, data: Dict[str, Any]) -> AgentConfig:
        """字典 → AgentConfig。安全读取，缺失字段使用默认值。"""
        risk = data.get("risk", {})
        scan = data.get("scan", {})
        llm = data.get("llm", {})
        dconf = data.get("data", {})

        return AgentConfig(
            strategy=data.get("strategy", "trend"),
            max_positions=int(risk.get("max_positions", 5)),
            max_position_pct=float(risk.get("max_position_pct", 0.20)),
            stop_loss_pct=float(risk.get("stop_loss_pct", -8.0)),
            take_profit_pct=float(risk.get("take_profit_pct", 20.0)),
            scan_top_n=int(scan.get("top_n", 50)),
            llm_enabled=bool(llm.get("enabled", False)),
            llm_provider=llm.get("provider") or None,
            llm_model=llm.get("model") or None,
            llm_base_url=llm.get("base_url") or None,
            cache_days=int(dconf.get("cache_days", 365)),
            realtime_enabled=bool(dconf.get("realtime_enabled", True)),
        )