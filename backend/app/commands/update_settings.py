"""Command: Update application settings."""

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UpdateSettingsCommand:
    key: str
    value: str
    user_id: str = "system"


class UpdateSettingsHandler:
    async def handle(self, cmd: UpdateSettingsCommand) -> dict:
        logger.info("cmd_update_settings", key=cmd.key, user_id=cmd.user_id)
        from app.services.data.cdc import cdc
        await cdc.on_config_changed(cmd.key, "", cmd.value)
        return {"key": cmd.key, "value": cmd.value, "updated": True}


update_settings_handler = UpdateSettingsHandler()
