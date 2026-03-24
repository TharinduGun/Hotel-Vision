"""
Orchestrator Engine
====================
The main processing loop that ties cameras, modules, and shared
services together.

Flow:
  1. Load config → discover cameras + enabled modules
  2. For each camera:
     a. Open stream via CameraManager
     b. Run shared person detection (streaming mode)
     c. For each frame, build FrameContext and route to modules
     d. Collect events and publish
  3. Shutdown all services
"""

from __future__ import annotations

import os
import sys
import gc
from datetime import datetime

import cv2
import torch
import numpy as np
from tqdm import tqdm

from app.config import load_config, get_device
from app.contracts.base_module import FrameContext
from app.contracts.event_schema import AnalyticsEvent
from app.shared.camera_manager import CameraManager
from app.shared.person_detector import PersonDetector
from app.shared.event_publisher import EventPublisher
from app.shared.snapshot_saver import SnapshotSaver
from app.shared.roi_manager import ROIManager
from app.orchestrator.module_loader import load_modules, get_modules_for_camera


def flush_memory():
    """Release GPU/CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class Engine:
    """
    The orchestrator engine — runs the full analytics pipeline.

    Usage:
        engine = Engine(config_path="app/config/system_config.yaml")
        engine.run()
    """

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.device = get_device(self.config)

        self.camera_manager = CameraManager(self.config)
        self.person_detector: PersonDetector | None = None
        self.event_publisher: EventPublisher | None = None
        self.snapshot_saver: SnapshotSaver | None = None
        self.modules = []

    def run(self):
        """Execute the full analytics pipeline."""
        print("=" * 60)
        print("  HOTEL VISION — Modular Analytics Engine")
        print("=" * 60)
        print(f"  Device   : {self.device}")
        print(f"  Cameras  : {self.camera_manager.camera_ids}")
        print("=" * 60)

        # ── 1. Initialize shared services ──────────────────────────
        self.person_detector = PersonDetector(self.config)

        shared_cfg = self.config.get("shared", {})
        self.event_publisher = EventPublisher(
            output_dir=shared_cfg.get("output_dir", "output/logs"),
        )

        # ── 2. Load modules ────────────────────────────────────────
        self.modules = load_modules(self.config)

        if not self.modules:
            print("\n⚠️  No modules loaded. Nothing to process.")
            return

        # ── 3. Process each camera ─────────────────────────────────
        for cam_id, stream in self.camera_manager.streams():
            cam_modules = get_modules_for_camera(
                self.modules, cam_id, self.config
            )
            if not cam_modules:
                print(f"\n[Engine] {cam_id}: No modules enabled — skipping")
                continue

            print(f"\n{'─' * 50}")
            print(f"Processing camera: {cam_id} "
                  f"({', '.join(m.name for m in cam_modules)})")
            print(f"{'─' * 50}")

            self._process_camera(cam_id, stream, cam_modules)
            flush_memory()

        # ── 4. Shutdown ────────────────────────────────────────────
        self._shutdown()

    def _process_camera(self, cam_id, stream, cam_modules):
        """Process all frames from one camera through applicable modules."""
        with stream:
            cam_info = stream.info

            # Initialize snapshot saver for this camera
            self.snapshot_saver = SnapshotSaver(
                session_dir=self.event_publisher.session_dir,
                fps=cam_info.fps,
            )

            # Run person tracking in streaming mode
            results = self.person_detector.track(cam_info.source)

            # Output video writer
            output_path = self.event_publisher.session_dir / f"output_{cam_id}.mp4"
            out_writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                cam_info.fps,
                (cam_info.width, cam_info.height),
            )

            # Initialize ROI Manager for this camera's resolution
            zones_config = self.config.get("shared", {}).get("zones_config", "pycode/config/zones.json")
            roi_manager = ROIManager(
                config_path=zones_config,
                frame_size=(cam_info.width, cam_info.height)
            )

            # ── Frame loop ─────────────────────────────────────────
            for frame_idx, result in enumerate(
                tqdm(results, total=cam_info.total_frames, desc=f"{cam_id}")
            ):
                frame = result.orig_img
                timestamp = frame_idx / cam_info.fps

                # Enforce max frames limit
                if frame_idx >= cam_info.total_frames:
                    break

                # Parse person tracks from YOLO result
                person_tracks = PersonDetector.parse_result(result)

                # Build shared context
                context = FrameContext(
                    camera_id=cam_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    fps=cam_info.fps,
                    frame_width=cam_info.width,
                    frame_height=cam_info.height,
                    person_tracks=person_tracks,
                    roi_manager=roi_manager,
                    roles={},  # Modules handle their own roles for now
                )

                # Buffer frame for clip capture
                self.snapshot_saver.buffer_frame(frame)

                # ── Route to modules ───────────────────────────────
                all_events: list[AnalyticsEvent] = []

                for module in cam_modules:
                    try:
                        events = module.process_frame(frame, context)
                        all_events.extend(events)
                    except Exception as e:
                        print(f"  [Engine] ERROR in {module.name}: {e}")

                # ── Publish events ─────────────────────────────────
                for event in all_events:
                    # Save snapshot for high-severity events
                    if event.severity.value in ("high", "critical"):
                        snap_path = self.snapshot_saver.save_snapshot(
                            frame, event
                        )
                        event.snapshot_path = snap_path

                        # Start clip capture
                        clip_path = self.snapshot_saver.start_clip_capture(
                            event,
                            frame_size=(cam_info.width, cam_info.height),
                        )
                        event.clip_path = clip_path

                    self.event_publisher.publish(event)

                # Feed post-capture frames to active clips
                self.snapshot_saver.feed_post_frame(frame)

                # ── Annotate output video ──────────────────────────
                annotated = self._annotate_frame(
                    frame, person_tracks, all_events, roi_manager=roi_manager, roles=context.roles
                )
                out_writer.write(annotated)

            # ── Cleanup camera ─────────────────────────────────────
            out_writer.release()
            self.snapshot_saver.shutdown()
            print(f"[Engine] Output video: {output_path}")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        person_tracks: dict,
        events: list[AnalyticsEvent],
        roi_manager: ROIManager | None = None,
        roles: dict | None = None,
    ) -> np.ndarray:
        """Draw person boxes, zones, and event detections on the frame."""
        annotated = frame.copy()

        # Draw ROIs if available
        if roi_manager and roi_manager.has_zones:
            roi_manager.draw_zones(annotated, alpha=0.15)

        # Draw person bounding boxes
        for tid, tdata in person_tracks.items():
            if tdata["cls"] != 0:  # Only annotate persons
                continue
            bx = tdata["bbox"]
            x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
            
            # Default formatting
            color = (0, 255, 255)  # Yellow for unknown/persons
            label = f"ID: {tid}"

            # Role formatting
            if roles and tid in roles:
                role = roles[tid]
                if role == "Cashier":
                    color = (0, 255, 0)   # Green for staff
                    label = f"Cashier {tid}"
                elif role == "Customer":
                    color = (255, 0, 0)   # Blue for customer
                    label = f"Customer {tid}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            t_size = cv2.getTextSize(label, 0, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - t_size[1] - 3),
                (x1 + t_size[0], y1),
                color, -1
            )
            cv2.putText(
                annotated, label, (x1, y1 - 2),
                0, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA
            )

        # Draw event-specific annotations
        for event in events:
            x1, y1, x2, y2 = [int(v) for v in event.bbox]

            if event.severity.value == "critical":
                color = (0, 0, 255)  # Red for critical
                thickness = 3
            elif event.severity.value == "high":
                color = (0, 100, 255)  # Dark orange
                thickness = 2
            else:
                color = (0, 165, 255)  # Orange
                thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"{event.event_type} ({event.confidence:.0%})"
            t_size = cv2.getTextSize(label, 0, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (x1, y2), (x1 + t_size[0], y2 + t_size[1] + 4),
                color, -1
            )
            cv2.putText(
                annotated, label, (x1, y2 + t_size[1] + 2),
                0, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA
            )

        # Draw global alert banner for critical events
        critical = [e for e in events if e.severity.value == "critical"]
        if critical:
            banner = f"⚠ ALERT: {critical[0].event_type.upper()}"
            cv2.putText(
                annotated, banner, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )

        return annotated

    def _shutdown(self):
        """Shut down all services and modules."""
        print("\n" + "=" * 50)
        print("  SHUTTING DOWN")
        print("=" * 50)

        for module in self.modules:
            try:
                module.shutdown()
            except Exception as e:
                print(f"  [Engine] Error shutting down {module.name}: {e}")

        if self.person_detector:
            self.person_detector.shutdown()

        if self.event_publisher:
            self.event_publisher.shutdown()

        flush_memory()
        print("=" * 50)
        print("  ENGINE STOPPED")
        print("=" * 50)
