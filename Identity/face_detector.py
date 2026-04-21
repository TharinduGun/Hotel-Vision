"""
Face detection using InsightFace.

Returns face objects with bounding boxes and embeddings that can be
passed directly to FaceRecognizer.
"""

from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(self, ctx_id=0):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id)

    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame: BGR numpy array.

        Returns:
            List of InsightFace face objects. Each has .bbox (x1,y1,x2,y2)
            and .embedding (512-d vector).
        """
        return self.app.get(frame)