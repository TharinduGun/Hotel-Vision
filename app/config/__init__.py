"""
Config Loader
==============
Reads the central system_config.yaml and resolves
all paths relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Project root: the parent of app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "app" / "config" / "system_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and validate the system config YAML.

    Args:
        config_path: Path to YAML file. Uses default if None.

    Returns:
        Parsed config dict with paths resolved to absolute.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Resolve relative paths ─────────────────────────────────────
    _resolve_paths(cfg)

    return cfg


def _resolve_paths(cfg: dict) -> None:
    """Convert relative paths in the config to absolute (based on PROJECT_ROOT)."""
    # Camera sources
    for cam_id, cam_cfg in cfg.get("cameras", {}).items():
        source = cam_cfg.get("source", "")
        if source and not os.path.isabs(source):
            cam_cfg["source"] = str(PROJECT_ROOT / source)

    # Module model paths
    for mod_name, mod_cfg in cfg.get("modules", {}).items():
        model_path = mod_cfg.get("model_path", "")
        if model_path and not os.path.isabs(model_path):
            mod_cfg["model_path"] = str(PROJECT_ROOT / model_path)

    # Shared paths
    shared = cfg.get("shared", {})
    for key in ("person_model_path", "pose_model_path", "output_dir",
                "zones_config", "tracker_config"):
        val = shared.get(key, "")
        if val and not os.path.isabs(val):
            shared[key] = str(PROJECT_ROOT / val)


def get_device(cfg: dict) -> str:
    """Determine the compute device from config."""
    import torch
    device = cfg.get("shared", {}).get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_camera_modules(cfg: dict, camera_id: str) -> list[str]:
    """Get the list of enabled module names for a given camera."""
    cam_cfg = cfg.get("cameras", {}).get(camera_id, {})
    module_names = cam_cfg.get("modules", [])

    # Filter to only enabled modules
    enabled = []
    for name in module_names:
        mod_cfg = cfg.get("modules", {}).get(name, {})
        if mod_cfg.get("enabled", True):
            enabled.append(name)
    return enabled
