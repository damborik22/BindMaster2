"""Auto-discover and register installed design tools."""

from __future__ import annotations

import logging

from bm2.tools.base import ToolLauncher

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available design tools."""

    def __init__(self):
        self._launchers: dict[str, ToolLauncher] = {}

    def register(self, launcher: ToolLauncher) -> None:
        """Register a tool launcher."""
        self._launchers[launcher.name] = launcher

    def auto_discover(self) -> None:
        """Try to instantiate each known launcher.

        Only registers tools where check_installed() passes.
        """
        from bm2.tools.bindcraft import BindCraftLauncher
        from bm2.tools.boltzgen import BoltzGenLauncher
        from bm2.tools.mosaic import MosaicLauncher
        from bm2.tools.pxdesign import PXDesignLauncher
        from bm2.tools.rfdiffusion import RFdiffusionLauncher
        from bm2.tools.complexa import ComplexaLauncher

        for cls in [
            BindCraftLauncher,
            BoltzGenLauncher,
            MosaicLauncher,
            PXDesignLauncher,
            RFdiffusionLauncher,
            ComplexaLauncher,
        ]:
            try:
                launcher = cls()
                if launcher.check_installed():
                    self.register(launcher)
                    logger.info(
                        f"Tool registered: {launcher.name} ({launcher.env_spec})"
                    )
                else:
                    logger.debug(f"Tool not installed: {cls.__name__}")
            except Exception as e:
                logger.debug(f"Tool discovery failed for {cls.__name__}: {e}")

    def list_installed(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._launchers.keys())

    def get(self, name: str) -> ToolLauncher:
        """Get a launcher by name.

        Raises:
            KeyError: If tool is not registered.
        """
        if name not in self._launchers:
            raise KeyError(
                f"Tool not registered: {name}. "
                f"Installed: {self.list_installed()}"
            )
        return self._launchers[name]

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._launchers
