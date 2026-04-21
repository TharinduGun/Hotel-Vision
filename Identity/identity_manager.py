"""
Caches the mapping from track IDs to employee identities.

Once a track is recognised, the identity is stored so face recognition
does not need to run again for that same track.
"""


class IdentityManager:

    def __init__(self):
        self.track_identity = {}

    def assign(self, track_id, identity):
        """Store identity for a track (only first assignment sticks)."""
        if track_id not in self.track_identity:
            self.track_identity[track_id] = identity

    def get(self, track_id):
        """Get stored identity for a track, or 'unknown'."""
        return self.track_identity.get(track_id, "unknown")

    def is_identified(self, track_id):
        """Check if this track already has a non-unknown identity."""
        return self.track_identity.get(track_id, "unknown") != "unknown"

    def get_all_identified(self):
        """Return set of all track IDs that have been identified as employees."""
        return {
            tid for tid, identity in self.track_identity.items()
            if identity != "unknown"
        }

    def clear(self):
        """Reset all identity mappings."""
        self.track_identity.clear()