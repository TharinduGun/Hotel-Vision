import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    Uses tiled/sliced inference to fix the core problem:
    YOLO merging closely-parked vehicles into a single detection.

    How it works:
    - The full frame is cut into overlapping tiles (e.g. 4 tiles with 20% overlap)
    - YOLO runs separately on each tile
    - Detections from all tiles are merged back into full-frame coordinates
    - A final NMS pass removes any remaining duplicates at tile boundaries

    This is why it fixes merging: two cars that look like one blob in the
    full 1080p frame are clearly two separate cars when YOLO sees a zoomed-in
    640x640 tile of just that corner of the lot.
    """

    CONF_DAY            = 0.25
    CONF_NIGHT          = 0.20
    LOW_LIGHT_THRESHOLD = 60
    VEHICLE_CLASSES     = [2, 3, 5, 7]  # car, motorcycle, bus, truck

   
    TILE_ROWS = 2
    TILE_COLS = 2
    TILE_OVERLAP = 0.2   # 20% overlap between tiles to catch vehicles at edges

    NMS_IOU_THRESHOLD = 0.3

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


    def _preprocess(self, frame):
        lab            = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b        = cv2.split(lab)
        l_enhanced     = self.clahe.apply(l)
        lab_merged     = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


    def _get_confidence(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.CONF_NIGHT if gray.mean() < self.LOW_LIGHT_THRESHOLD \
               else self.CONF_DAY


    def _get_tiles(self, frame):
        """
        Split frame into overlapping tiles.
        Returns list of (tile_image, x_offset, y_offset).
        x_offset and y_offset are used to map detections back to
        full-frame coordinates.
        """
        H, W  = frame.shape[:2]
        tiles = []

        step_y = int(H / self.TILE_ROWS)
        step_x = int(W / self.TILE_COLS)

        overlap_y = int(step_y * self.TILE_OVERLAP)
        overlap_x = int(step_x * self.TILE_OVERLAP)

        for row in range(self.TILE_ROWS):
            for col in range(self.TILE_COLS):

                # Tile boundaries with overlap
                y1 = max(0, row * step_y - overlap_y)
                y2 = min(H, y1 + step_y + overlap_y * 2)
                x1 = max(0, col * step_x - overlap_x)
                x2 = min(W, x1 + step_x + overlap_x * 2)

                tile = frame[y1:y2, x1:x2]
                tiles.append((tile, x1, y1))

        return tiles


    def _nms(self, detections):
        """
        Non-Maximum Suppression across all tile detections.
        Removes duplicate detections of the same vehicle that appeared
        in two overlapping tiles.
        """
        if not detections:
            return []

        boxes  = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

        # OpenCV NMS expects (x, y, w, h) format
        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.CONF_DAY,
            self.NMS_IOU_THRESHOLD
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]


    def detect(self, frame) -> list:

        enhanced = self._preprocess(frame)
        conf     = self._get_confidence(frame)
        tiles    = self._get_tiles(enhanced)

        all_detections = []

        for tile_img, x_offset, y_offset in tiles:

            results = self.model(tile_img, conf=conf, verbose=False)[0]

            if results.boxes is None:
                continue

            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in self.VEHICLE_CLASSES:
                    continue

                # Map tile-local coordinates back to full-frame coordinates
                tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                fx1 = tx1 + x_offset
                fy1 = ty1 + y_offset
                fx2 = tx2 + x_offset
                fy2 = ty2 + y_offset

                all_detections.append({
                    "bbox":       [fx1, fy1, fx2, fy2],
                    "confidence": float(box.conf[0]),
                    "class_id":   cls
                })

        # Final NMS pass to remove duplicates from overlapping tile regions
        return self._nms(all_detections)