"""
Simple OpenCV video capture wrapper.
"""

import cv2


class VideoLoader:

    def __init__(self, source):
        """
        Args:
            source: File path or RTSP URL.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()