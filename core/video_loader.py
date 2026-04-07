import cv2
import time
import logging

logger = logging.getLogger(__name__)


class VideoLoader:

    # File extensions treated as local files — no retry on read failure
    LOCAL_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')

    def __init__(self, source, max_retries=10, retry_delay=2):

        self.source      = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Detect once at init whether this is a local file or a live stream
        # so read() knows which behaviour to use
        self.is_file = (
            isinstance(source, str) and source.endswith(self.LOCAL_EXTENSIONS)
        )

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")

        if self.is_file:
            logger.info(f"Opened local video file: {source}")
        else:
            logger.info(f"Opened live stream: {source}")


    def read(self):

        ret, frame = self.cap.read()

        if ret:
            return frame

        # ── Local file: a failed read simply means the video ended ───────────
        # Return None immediately so the main loop breaks cleanly.
        # DO NOT retry — reopening would restart from frame 0 silently.
        if self.is_file:
            logger.info("Video file finished.")
            return None

        # ── Live stream (RTSP etc): a failed read means a network drop ────────
        # Retry with reconnection. The sleep here is intentional for streams —
        # it gives the NVR/camera time to recover before reconnecting.
        for attempt in range(self.max_retries):
            logger.warning(
                f"Stream lost. Reconnecting... "
                f"(attempt {attempt + 1}/{self.max_retries})"
            )
            self.cap.release()
            time.sleep(self.retry_delay)
            self.cap = cv2.VideoCapture(self.source)
            ret, frame = self.cap.read()
            if ret:
                logger.info("Stream reconnected.")
                return frame

        logger.error("Could not reconnect after max retries. Stopping.")
        return None


    def release(self):
        self.cap.release()