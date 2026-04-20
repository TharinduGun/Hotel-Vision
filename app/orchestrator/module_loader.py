"""
Module Loader
==============
Discovers and instantiates enabled analytics modules from config.
Maps each module class name to its Python import path.
"""

from __future__ import annotations

from typing import Type

from app.contracts.base_module import AnalyticsModule


# ── Module Registry ────────────────────────────────────────────────────
# Maps config module names to their Python module paths and class names.
# Add new modules here — no other code changes needed.
MODULE_REGISTRY: dict[str, str] = {
    "gun_detection":    "app.modules.gun_detection.module.GunDetectionModule",
    "cash_detection":   "app.modules.cash_detection.module.CashDetectionModule",
    "crowd_detection":  "app.modules.crowd_detection.module.CrowdDetectionModule",
    "staff_tracking":   "app.modules.staff_tracking.module.StaffTrackingModule",
    "parking":        "app.modules.parking.module.ParkingModule",
    # "inventory":      "app.modules.inventory.module.InventoryModule",
}


def _import_class(dotted_path: str) -> Type[AnalyticsModule]:
    """Dynamically import a class from a dotted path string."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    if not issubclass(cls, AnalyticsModule):
        raise TypeError(
            f"{dotted_path} does not implement AnalyticsModule interface"
        )
    return cls


def load_modules(config: dict) -> list[AnalyticsModule]:
    modules_cfg = config.get("modules", {})
    loaded: list[AnalyticsModule] = []

    for mod_name, mod_cfg in modules_cfg.items():
        if not mod_cfg.get("enabled", True):
            print(f"[ModuleLoader] {mod_name}: DISABLED (skipping)")
            continue

        if mod_name not in MODULE_REGISTRY:
            print(f"[ModuleLoader] WARNING: '{mod_name}' not in registry — skipping")
            continue

        dotted_path = MODULE_REGISTRY[mod_name]

        try:
            cls = _import_class(dotted_path)
            instance = cls()
            instance.initialize(mod_cfg)
            loaded.append(instance)
            print(f"[ModuleLoader] ✅ {mod_name}: loaded ({cls.__name__})")
        except Exception as e:
            print(f"[ModuleLoader] ❌ {mod_name}: FAILED to load — {e}")
            continue

    print(f"[ModuleLoader] {len(loaded)} module(s) loaded successfully")
    return loaded


def get_modules_for_camera(
    modules: list[AnalyticsModule],
    camera_id: str,
    config: dict,
) -> list[AnalyticsModule]:
    from app.config import get_camera_modules
    enabled_names = get_camera_modules(config, camera_id)

    return [
        m for m in modules
        if m.name in enabled_names or "*" in m.applicable_cameras()
    ]
