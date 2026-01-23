import time
import math

class OcclusionHandler:
    def __init__(self, max_age_seconds=3.0):
        """
        Manages lost tracks to handle occlusion re-linking.
        
        Args:
            max_age_seconds (float): How long to keep a lost track in memory.
        """
        self.max_age = max_age_seconds
        # buffer = { track_id: {'bbox': [x1,y1,x2,y2], 'cls': int, 'time': float, 'img_feats': None} }
        self.buffer = {}

    def add_lost_track(self, track_id, bbox, cls_id, timestamp):
        """
        Register a track that has just disappeared.
        """
        self.buffer[track_id] = {
            'bbox': bbox,
            'cls': cls_id,
            'time': timestamp,
            'relinked_to': None # Trace if this was eventually merged
        }
        # print(f"[OcclusionHandler] Track {track_id} buffered as lost at {timestamp:.2f}")

    def cleanup(self, current_time):
        """
        Remove tracks that have been lost for too long.
        """
        keys_to_remove = []
        for tid, data in self.buffer.items():
            age = current_time - data['time']
            if age > self.max_age:
                keys_to_remove.append(tid)
        
        for k in keys_to_remove:
            del self.buffer[k]
            # print(f"[OcclusionHandler] Track {k} expired from buffer")

    def try_relink(self, new_track_bbox, new_cls_id, current_time, iou_thresh=0.3, dist_thresh=150.0):
        """
        Attempt to match a NEW track to a LOST track.
        Priority 1: High IoU (Overlapping)
        Priority 2: Low Distance (Proximity - good for edge re-entry)
        
        Returns: old_track_id if matched, else None.
        """
        # Candidates for IoU matching
        best_iou_match_id = None
        best_iou = -1.0
        
        # Candidates for Distance matching
        best_dist_match_id = None
        min_dist = float('inf')

        for old_id, data in self.buffer.items():
            # 1. Class Check
            if data['cls'] != new_cls_id:
                continue

            # 2. IoU Check
            iou = self._calculate_iou(new_track_bbox, data['bbox'])
            
            if iou > iou_thresh:
                if iou > best_iou:
                    best_iou = iou
                    best_iou_match_id = old_id
            
            # 3. Distance Check (Fallback if IoU is low/zero)
            else:
                dist = self._calculate_distance(new_track_bbox, data['bbox'])
                if dist < dist_thresh:
                    if dist < min_dist:
                        min_dist = dist
                        best_dist_match_id = old_id

        # Decision: Prefer IoU match, then Distance match
        if best_iou_match_id:
            del self.buffer[best_iou_match_id]
            return best_iou_match_id
        
        if best_dist_match_id:
            del self.buffer[best_dist_match_id]
            return best_dist_match_id
        
        return None

    def _calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _calculate_distance(self, boxA, boxB):
        # Calculate centers
        centerA_x = (boxA[0] + boxA[2]) / 2
        centerA_y = (boxA[1] + boxA[3]) / 2
        
        centerB_x = (boxB[0] + boxB[2]) / 2
        centerB_y = (boxB[1] + boxB[3]) / 2
        
        # Euclidean distance
        dist = math.sqrt((centerA_x - centerB_x)**2 + (centerA_y - centerB_y)**2)
        return dist
