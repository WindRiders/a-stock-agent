"""通知推送模块。

支持通过 Webhook 将每日报告推送到：
- Telegram Bot
- Discord Webhook
- 企业微信机器人
- 自定义 Webhook

配置通过环境变量：
- TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
- DISCORD_WEBHOOK_URL
- WECOM_WEBHOOK_URL
- CUSTOM_WEBHOOK_URL
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    """通用通知推送器。"""

    def __init__(self):
        self.results = []

    # ── Telegram ────────────────────────────────────────────

    def send_telegram(self, text: str, parse_mode: str = "HTML") -> bool:
        """通过 Telegram Bot 发送消息。"""
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if not token or not chat_id:
            logger.debug("Telegram 未配置 (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
            return False

        import urllib.request

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = json.dumps({
            "chat_id": chat_id,
            "text": text[:4000],  # Telegram 限制 4096
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode()

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                ok = data.get("ok", False)
                if ok:
                    logger.info("Telegram 推送成功")
                else:
                    logger.warning("Telegram 推送失败: %s", data)
                return ok
        except Exception as e:
            logger.error("Telegram 推送异常: %s", e)
            return False

    # ── Discord ─────────────────────────────────────────────

    def send_discord(self, content: str, title: str = None) -> bool:
        """通过 Discord Webhook 发送消息。"""
        url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not url:
            logger.debug("Discord 未配置 (DISCORD_WEBHOOK_URL)")
            return False

        import urllib.request

        embed = {
            "title": title or "A股每日分析",
            "description": content[:2000],
            "color": 0x00AA00,
            "timestamp": datetime.now().isoformat(),
        }

        body = json.dumps({"embeds": [embed]}).encode()

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                ok = 200 <= resp.status < 300
                if ok:
                    logger.info("Discord 推送成功")
                return ok
        except Exception as e:
            logger.error("Discord 推送异常: %s", e)
            return False

    # ── 企业微信 ────────────────────────────────────────────

    def send_wecom(self, content: str, msg_type: str = "markdown") -> bool:
        """通过企业微信机器人发送消息。"""
        url = os.environ.get("WECOM_WEBHOOK_URL")
        if not url:
            logger.debug("企业微信未配置 (WECOM_WEBHOOK_URL)")
            return False

        import urllib.request

        body = json.dumps({
            "msgtype": msg_type,
            msg_type: {"content": content[:4000]},
        }).encode()

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                ok = 200 <= resp.status < 300
                if ok:
                    logger.info("企业微信推送成功")
                return ok
        except Exception as e:
            logger.error("企业微信推送异常: %s", e)
            return False

    # ── 自定义 Webhook ─────────────────────────────────────

    def send_custom_webhook(self, payload: dict) -> bool:
        """发送到自定义 Webhook。"""
        url = os.environ.get("CUSTOM_WEBHOOK_URL")
        if not url:
            return False

        import urllib.request

        body = json.dumps(payload, ensure_ascii=False, default=str).encode()

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return 200 <= resp.status < 300
        except Exception as e:
            logger.error("自定义Webhook推送异常: %s", e)
            return False

    # ── 群发 ────────────────────────────────────────────────

    def broadcast(
        self,
        message: str,
        title: str = "A股每日分析报告",
        targets: list = None,
    ) -> dict:
        """向所有已配置的渠道群发消息。

        Args:
            message: 消息内容
            title: 标题
            targets: 指定渠道列表 ["telegram", "discord", "wecom"]，None=全部

        Returns:
            {channel: success}
        """
        if targets is None:
            targets = ["telegram", "discord", "wecom"]

        results = {}

        if "telegram" in targets:
            # Telegram 用简化的HTML格式
            tg_msg = f"<b>{title}</b>\n\n{message[:3500]}"
            if len(message) > 3500:
                tg_msg += "\n\n<i>...内容过长已截断</i>"
            results["telegram"] = self.send_telegram(tg_msg)

        if "discord" in targets:
            results["discord"] = self.send_discord(message[:1800], title)

        if "wecom" in targets:
            # 企业微信 markdown
            md_msg = f"## {title}\n\n{message[:3800]}"
            results["wecom"] = self.send_wecom(md_msg)

        if "custom" in targets:
            results["custom"] = self.send_custom_webhook({
                "title": title,
                "content": message,
                "timestamp": datetime.now().isoformat(),
            })

        self.results = results
        return results

    @property
    def configured_channels(self) -> list:
        """列出已配置的渠道。"""
        channels = []
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            channels.append("telegram")
        if os.environ.get("DISCORD_WEBHOOK_URL"):
            channels.append("discord")
        if os.environ.get("WECOM_WEBHOOK_URL"):
            channels.append("wecom")
        if os.environ.get("CUSTOM_WEBHOOK_URL"):
            channels.append("custom")
        return channels

    @property
    def has_any_channel(self) -> bool:
        return len(self.configured_channels) > 0