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

    def try_relink(self, new_track_bbox, new_cls_id, current_time, iou_thresh=0.10, dist_thresh=150.0, frame_size=None):
        """
        Attempt to match a NEW track to a LOST track.
        Priority 1: High IoU (Overlapping)
        Priority 2: Low Distance (Proximity - good for edge re-entry)
        
        Args:
            frame_size: (width, height) tuple for edge detection.
        
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
                
                # --- User Requested: Dual-box adaptive threshold ---
                # Use max dimensions of BOTH the old lost box and the new candidate box
                old_w = data['bbox'][2] - data['bbox'][0]
                old_h = data['bbox'][3] - data['bbox'][1]
                new_w = new_track_bbox[2] - new_track_bbox[0]
                new_h = new_track_bbox[3] - new_track_bbox[1]

                scale = max(old_w, old_h, new_w, new_h)
                dynamic_thresh = scale * 2.5
                
                # Allow a hard upper floor (max of dynamic, config, or explicit 80px)
                # This ensures we don't pick a tiny threshold for small objects if dist_thresh is generous
                final_thresh = max(dynamic_thresh, dist_thresh, 80.0)
                
                # --- User Requested: Edge Boost ---
                if frame_size:
                    W, H = frame_size
                    edge_margin = 0.06  # 6% of frame
                    
                    # Check if OLD lost box was near edge
                    ob = data['bbox']
                    near_edge = (ob[0] < W*edge_margin or ob[2] > W*(1-edge_margin) or
                                 ob[1] < H*edge_margin or ob[3] > H*(1-edge_margin))
                    
                    if near_edge:
                        final_thresh *= 1.5
                
                if dist < final_thresh:
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
