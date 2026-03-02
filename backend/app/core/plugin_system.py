"""Plugin system for custom trading strategies and extensions.

Supports hot-reload of plugins without downtime.
"""

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class PluginInterface(ABC):
    """Base interface for all TradeMaster plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        ...

    @abstractmethod
    async def initialize(self, context: dict) -> None:
        """Initialize the plugin with application context."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up plugin resources."""
        ...


class StrategyPlugin(PluginInterface):
    """Base class for custom trading strategy plugins."""

    @abstractmethod
    async def on_candle(self, symbol: str, candle: dict) -> dict | None:
        """Called on each new candle. Return signal dict or None."""
        ...

    @abstractmethod
    async def on_trade(self, trade: dict) -> None:
        """Called when a trade is executed."""
        ...


@dataclass
class PluginInfo:
    name: str
    version: str
    path: str
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    error: str | None = None


class PluginManager:
    """Manage plugin lifecycle: discovery, loading, and hot-reload."""

    def __init__(self, plugins_dir: str = "plugins"):
        self._plugins_dir = Path(plugins_dir)
        self._plugins: dict[str, PluginInterface] = {}
        self._plugin_info: dict[str, PluginInfo] = {}

    async def discover_plugins(self) -> list[str]:
        """Discover available plugins in the plugins directory."""
        discovered = []
        if not self._plugins_dir.exists():
            return discovered

        for path in self._plugins_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            discovered.append(path.stem)

        logger.info("plugins_discovered", count=len(discovered), plugins=discovered)
        return discovered

    async def load_plugin(self, plugin_name: str, context: dict | None = None) -> bool:
        """Load a plugin by name."""
        plugin_path = self._plugins_dir / f"{plugin_name}.py"
        if not plugin_path.exists():
            logger.warning("plugin_not_found", name=plugin_name)
            return False

        try:
            spec = importlib.util.spec_from_file_location(plugin_name, str(plugin_path))
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Look for a class implementing PluginInterface
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, PluginInterface) and attr is not PluginInterface and attr is not StrategyPlugin:
                    plugin_class = attr
                    break

            if not plugin_class:
                logger.warning("plugin_no_interface", name=plugin_name)
                return False

            plugin = plugin_class()
            await plugin.initialize(context or {})
            self._plugins[plugin_name] = plugin
            self._plugin_info[plugin_name] = PluginInfo(
                name=plugin.name, version=plugin.version, path=str(plugin_path),
            )
            logger.info("plugin_loaded", name=plugin.name, version=plugin.version)
            return True
        except Exception as e:
            logger.error("plugin_load_failed", name=plugin_name, error=str(e))
            self._plugin_info[plugin_name] = PluginInfo(
                name=plugin_name, version="unknown", path=str(plugin_path), is_active=False, error=str(e),
            )
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            return False

        try:
            await plugin.shutdown()
        except Exception as e:
            logger.warning("plugin_shutdown_error", name=plugin_name, error=str(e))

        del self._plugins[plugin_name]
        if plugin_name in self._plugin_info:
            self._plugin_info[plugin_name].is_active = False

        if plugin_name in sys.modules:
            del sys.modules[plugin_name]

        logger.info("plugin_unloaded", name=plugin_name)
        return True

    async def reload_plugin(self, plugin_name: str, context: dict | None = None) -> bool:
        """Hot-reload a plugin without downtime."""
        await self.unload_plugin(plugin_name)
        return await self.load_plugin(plugin_name, context)

    def get_plugin(self, name: str) -> PluginInterface | None:
        return self._plugins.get(name)

    def get_strategy_plugins(self) -> list[StrategyPlugin]:
        return [p for p in self._plugins.values() if isinstance(p, StrategyPlugin)]

    def get_status(self) -> dict:
        return {
            "plugins_dir": str(self._plugins_dir),
            "loaded": len(self._plugins),
            "plugins": {
                name: {
                    "name": info.name,
                    "version": info.version,
                    "active": info.is_active,
                    "loaded_at": info.loaded_at.isoformat(),
                    "error": info.error,
                }
                for name, info in self._plugin_info.items()
            },
        }


plugin_manager = PluginManager()
