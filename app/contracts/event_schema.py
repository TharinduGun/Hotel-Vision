"""
Analytics Event Schema
=======================
One universal event structure emitted by every module.
The orchestrator collects these and routes them to the event publisher.

Every detection — gun, cash, staff anomaly — becomes an AnalyticsEvent
before it reaches the database, dashboard, or alert system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class Severity(str, Enum):
    """Alert severity levels (ascending urgency)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnalyticsEvent:
    """
    Common event structure all modules produce.

    Examples:
        AnalyticsEvent(
            module="gun_detection",
            camera_id="CAM-01",
            timestamp=12.4,
            event_type="weapon_detected",
            confidence=0.91,
            bbox=[120, 200, 180, 340],
            severity=Severity.CRITICAL,
        )

        AnalyticsEvent(
            module="cash_detection",
            camera_id="CAM-03",
            timestamp=45.1,
            event_type="cash_pickup",
            confidence=0.78,
            bbox=[400, 300, 480, 350],
            severity=Severity.MEDIUM,
            metadata={"person_id": 5, "zone": "Cashier 1"},
        )
    """

    # ── Required fields ────────────────────────────────────────────────
    module: str                     # "gun_detection", "cash_detection", etc.
    camera_id: str                  # Camera that produced this event
    timestamp: float                # Seconds since stream start
    event_type: str                 # Module-specific event name
    confidence: float               # Detection confidence [0.0 – 1.0]
    bbox: list[float]               # [x1, y1, x2, y2] bounding box
    severity: Severity              # Alert severity level

    # ── Optional fields ────────────────────────────────────────────────
    frame_idx: int = 0              # Frame number in the video
    person_id: int | None = None    # Associated person track ID (if applicable)
    snapshot_path: str | None = None  # Path to saved snapshot image
    clip_path: str | None = None    # Path to saved video clip
    iso_timestamp: str | None = None  # ISO 8601 wall-clock time

    # Flexible key-value bag for module-specific data
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Auto-populate ISO timestamp if not provided."""
        if self.iso_timestamp is None:
            self.iso_timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for CSV/JSON export."""
        return {
            "module": self.module,
            "camera_id": self.camera_id,
            "timestamp": round(self.timestamp, 3),
            "event_type": self.event_type,
            "confidence": round(self.confidence, 4),
            "bbox": f"{self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f}",
            "severity": self.severity.value,
            "frame_idx": self.frame_idx,
            "person_id": self.person_id,
            "snapshot_path": self.snapshot_path or "",
            "clip_path": self.clip_path or "",
            "iso_timestamp": self.iso_timestamp or "",
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }

    def __repr__(self) -> str:
        return (
            f"AnalyticsEvent({self.module}/{self.event_type}, "
            f"cam={self.camera_id}, conf={self.confidence:.2f}, "
            f"sev={self.severity.value})"
        )
