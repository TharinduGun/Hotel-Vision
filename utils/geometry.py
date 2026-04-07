"""
Geometry utility functions.
"""

from shapely.geometry import Point, Polygon


def point_in_zone(point, polygon_points):
    """
    Check if a point falls inside a polygon.

    Args:
        point: Tuple (x, y).
        polygon_points: List of (x, y) tuples defining the polygon.

    Returns:
        True if the point is inside the polygon.
    """
    polygon = Polygon(polygon_points)
    return polygon.contains(Point(point))