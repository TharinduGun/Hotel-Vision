import math

class EventMerger:
    def __init__(self, max_time_gap=5.0, max_speed_mps=None):
        """
        Args:
            max_time_gap (float): Maximum allowed gap (seconds) between events to consider merging.
            max_speed_mps (float): Maximum implied speed (pixels/sec or meters/sec) to allow merge. 
                                   If None, ignores spatial check.
                                   Note: Since we are in pixel coordinates, this is effectively "pixels per second".
                                   A higher value is more permissive.
        """
        self.max_time_gap = max_time_gap
        # Default generous speed threshold if not specified (e.g. 500 pixels per second)
        self.max_velocity = max_speed_mps if max_speed_mps else 1000.0 

    def merge_events(self, events):
        """
        Merges a list of event dictionaries.
        """
        if not events:
            return []

        # Sort by Start Time
        sorted_events = sorted(events, key=lambda x: x['Start_Time_Sec'])
        
        merged_events = []

        for next_event in sorted_events:
            best_match_idx = -1
            min_dist = float('inf')
            
            # Try to find a match in existing merged_events
            # We look at recent events that finished before this one started
            for idx, existing_event in enumerate(merged_events):
                
                # 1. Class Check
                if existing_event['Class'] != next_event['Class']:
                    continue

                # 2. Time Logic
                # strict no-overlap: next_event must start AFTER existing ends
                # (Allowing a tiny tolerance of -0.5s for jitter, maybe, but technically different IDs shouldn't overlap)
                gap = next_event['Start_Time_Sec'] - existing_event['End_Time_Sec']
                
                # If they overlap significantly (gap < -0.1), they are likely distinct objects present simultaneously
                if gap < -0.1:
                    continue
                
                if gap > self.max_time_gap:
                    continue
                    
                # 3. Spatial Logic
                p1 = existing_event['End_Center']
                p2 = next_event['Start_Center']
                dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                time_delta = max(gap, 0.1)
                required_velocity = dist / time_delta
                
                if required_velocity <= self.max_velocity:
                    # Valid candidate. Is it the best (closest)?
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx

            if best_match_idx != -1:
                # MERGE into the best match
                target = merged_events[best_match_idx]
                target['End_Time_Sec'] = next_event['End_Time_Sec']
                target['End_Center'] = next_event['End_Center']
                target['Frame_Count'] += next_event['Frame_Count']
                # Start Time & Start Center remain from the Target (earlier event)
            else:
                # New distinct event
                merged_events.append(next_event)

        return merged_events

    def _should_merge(self, e1, e2):
        # Deprecated helper, logic moved inside loop for "best match" context
        pass
