class IouTracker:
    
    def __init__(self, iou_threshold: float = 0.15, max_age: int = 20):

        self.iou_threshold = iou_threshold
        self.max_age       = max_age

        self.tracks  = {}    # tid → {"bbox": [...], "missed": int}
        self.next_id = 0


    def _iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / float(areaA + areaB - inter + 1e-6)


    def update(self, detections: list) -> list:

        matched_ids = set()

        for det in detections:
            bbox     = det["bbox"]
            best_tid = None
            best_iou = 0

            for tid, track in self.tracks.items():
                iou = self._iou(bbox, track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_tid = tid

            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1

            self.tracks[best_tid] = {"bbox": bbox, "missed": 0}
            matched_ids.add(best_tid)
            det["track_id"] = best_tid

        # FIX H-02: age unmatched tracks instead of instantly deleting them
        stale = []
        for tid, track in self.tracks.items():
            if tid not in matched_ids:
                track["missed"] += 1
                if track["missed"] > self.max_age:
                    stale.append(tid)

        for tid in stale:
            del self.tracks[tid]

        return detections