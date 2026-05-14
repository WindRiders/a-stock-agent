"""测试配置文件持久化模块。"""

import os
import tempfile
import pytest

from agent.config_file import ConfigManager, _default_config
from agent.config import AgentConfig


class TestConfigManager:
    """配置文件管理器测试。"""

    def setup_method(self):
        """每个测试前创建临时配置文件。"""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        self.tmp.close()
        self.mgr = ConfigManager(config_path=self.tmp.name)

    def teardown_method(self):
        """每个测试后清理。"""
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_default_config_structure(self):
        """默认配置包含所有必需字段。"""
        data = _default_config()
        assert "version" in data
        assert "strategy" in data
        assert "risk" in data
        assert "scan" in data
        assert "llm" in data
        assert "data" in data
        assert "notifications" in data

    def test_load_creates_default(self):
        """不存在时自动创建默认配置。"""
        cfg = self.mgr.load()
        assert isinstance(cfg, AgentConfig)
        assert cfg.strategy == "trend"
        assert os.path.exists(self.tmp.name)

    def test_save_and_load_roundtrip(self):
        """保存再加载，配置一致。"""
        cfg = self.mgr.load()
        cfg.strategy = "momentum"
        cfg.max_positions = 10
        cfg.stop_loss_pct = -5.0
        cfg.llm_enabled = True
        cfg.llm_model = "gpt-4"

        self.mgr.save(cfg)

        loaded = self.mgr.load()
        assert loaded.strategy == "momentum"
        assert loaded.max_positions == 10
        assert loaded.stop_loss_pct == -5.0
        assert loaded.llm_enabled is True
        assert loaded.llm_model == "gpt-4"

    def test_save_creates_backup(self):
        """保存时自动创建备份。"""
        self.mgr.load()
        self.mgr.save(AgentConfig(strategy="grid"))

        backups = self.mgr.list_backups()
        assert len(backups) >= 1

    def test_reset(self):
        """重置恢复默认值。"""
        cfg = self.mgr.load()
        cfg.strategy = "momentum"
        self.mgr.save(cfg)

        self.mgr.reset()
        loaded = self.mgr.load()
        assert loaded.strategy == "trend"

    def test_restore(self):
        """从备份恢复。"""
        self.mgr.load()
        # 先保存一次创建基准备份
        self.mgr.save(AgentConfig(strategy="momentum"))
        # 再保存新策略，save() 会自动备份旧值
        self.mgr.save(AgentConfig(strategy="grid"))

        backups = self.mgr.list_backups()
        assert len(backups) >= 1

        # 恢复最新的备份（应该是 momentum）
        filename = backups[0]["filename"]
        self.mgr.restore(filename)

        loaded = self.mgr.load()
        # 恢复后应该是备份时的内容
        assert loaded.strategy == "momentum"

    def test_show_returns_content(self):
        """show 返回配置内容。"""
        self.mgr.load()
        content = self.mgr.show()
        assert "strategy" in content
        assert "risk" in content

    def test_corrupted_file_recovers(self):
        """损坏的配置文件自动恢复。"""
        with open(self.tmp.name, "w") as f:
            f.write(":::not valid yaml:::")

        cfg = self.mgr.load()
        assert isinstance(cfg, AgentConfig)
        assert cfg.strategy == "trend"

    def test_partial_config_fills_defaults(self):
        """不完整的配置文件用默认值填充。"""
        import yaml
        with open(self.tmp.name, "w") as f:
            yaml.dump({"strategy": "value", "risk": {"max_positions": 3}}, f)

        cfg = self.mgr.load()
        assert cfg.strategy == "value"
        assert cfg.max_positions == 3
        assert cfg.stop_loss_pct == -8.0  # 默认值