"""
Manages multiple camera pipelines, one thread per camera.
"""

import json
import threading
from pipeline.camera_pipeline import CameraPipeline
from events.event_builder import EventBuilder


class PipelineManager:

    def __init__(self, config_path="configs/cameras.json", enable_websocket=False):
        """
        Args:
            config_path: Path to cameras.json config file.
            enable_websocket: Whether to enable WebSocket event streaming.
        """
        with open(config_path) as f:
            self.cameras = json.load(f)["cameras"]

        self.event_builder = EventBuilder(enable_websocket=enable_websocket)
        self.threads = []

    def start(self):
        """Launch all camera pipelines in separate threads and wait for completion."""
        print(f"Starting {len(self.cameras)} camera pipeline(s)...")

        for cam_config in self.cameras:
            camera_id = cam_config["id"]
            source = cam_config["source"]

            pipeline = CameraPipeline(
                camera_id=camera_id,
                source=source,
                event_builder=self.event_builder,
            )

            thread = threading.Thread(
                target=pipeline.run,
                name=f"pipeline-{camera_id}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()

        print("All pipelines stopped.")