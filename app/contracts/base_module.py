"""
Analytics Module Contract
==========================
Every detection module (gun, cash, staff, etc.) MUST implement this
interface. The orchestrator calls these methods — no module should
open cameras or publish alerts directly.

Usage:
    class GunDetectionModule(AnalyticsModule):
        def initialize(self, config): ...
        def applicable_cameras(self) -> list[str]: ...
        def process_frame(self, frame, context) -> list[AnalyticsEvent]: ...
        def shutdown(self): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .event_schema import AnalyticsEvent


# ── Frame Context (passed to every module per frame) ───────────────────

@dataclass
class FrameContext:
    """
    Metadata the orchestrator provides alongside each video frame.

    Every module receives the same FrameContext so they can share
    person detections, ROI info, and timing without duplicating work.
    """
    camera_id: str
    frame_idx: int
    timestamp: float            # seconds since stream start
    fps: float
    frame_width: int
    frame_height: int

    # Shared person detection results (populated by the orchestrator)
    # { logical_id: { "bbox": [x1,y1,x2,y2], "cls": int } }
    person_tracks: dict[int, dict] = field(default_factory=dict)

    # ROI manager reference (optional — not all deployments have zones)
    roi_manager: Any = None

    # Roles (populated by role classifier if available)
    # { logical_id: "Cashier" | "Customer" }
    roles: dict[int, str] = field(default_factory=dict)


# ── Module Interface ──────────────────────────────────────────────────

class AnalyticsModule(ABC):
    """
    Abstract base class for all analytics modules.

    Rules:
      1. Modules MUST NOT open camera streams directly.
      2. Modules MUST NOT write alerts/events directly.
      3. Modules receive frames from the orchestrator and return events.
      4. All returned events use the common AnalyticsEvent schema.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module identifier, e.g. 'gun_detection'."""
        ...

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """
        One-time setup: load models, parse config, allocate resources.

        Args:
            config: Module-specific config dict from system_config.yaml.
        """
        ...

    @abstractmethod
    def applicable_cameras(self) -> list[str]:
        """
        Return camera IDs this module should process.

        Returns:
            List of camera_id strings, or ['*'] for all cameras.
        """
        ...

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        context: FrameContext,
    ) -> list[AnalyticsEvent]:
        """
        Analyse one video frame and return zero or more events.

        Args:
            frame: BGR numpy array (the raw video frame).
            context: FrameContext with shared detections & metadata.

        Returns:
            List of AnalyticsEvent objects (empty list if nothing detected).
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release models, close files, free GPU memory."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
