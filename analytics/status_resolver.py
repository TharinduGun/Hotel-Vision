"""
Resolves employee status based on zone, idle state, and visibility.

Statuses:
    ON_DUTY  — Employee is in a work zone and moving.
    IDLE     — Employee is in a work zone but stationary beyond threshold.
    ON_BREAK — Employee is in a designated break zone.
    OFFLINE  — Previously seen employee not detected for timeout period.
"""

import time


class StatusResolver:

    def __init__(self, offline_timeout_seconds=30):
        """
        Args:
            offline_timeout_seconds: Seconds after last detection before
                                     an employee is considered OFFLINE.
        """
        self.offline_timeout = offline_timeout_seconds
        self.last_seen = {}

    def resolve(self, employee_id, zone, is_idle, zone_manager):
        """
        Determine the current status of an employee.

        Args:
            employee_id: Employee ID string (e.g. 'E001').
            zone: Current zone name from ZoneManager.
            is_idle: Boolean from IdleDetector.
            zone_manager: ZoneManager instance (to check break zones).

        Returns:
            Status string: 'ON_DUTY', 'IDLE', or 'ON_BREAK'.
        """
        self.last_seen[employee_id] = time.time()

        if zone_manager.is_break_zone(zone):
            return "ON_BREAK"

        if is_idle:
            return "IDLE"

        return "ON_DUTY"

    def get_offline_employees(self, known_employee_ids):
        """
        Check which previously-seen employees have gone undetected.

        Args:
            known_employee_ids: Set of all employee IDs that have been
                                seen at least once this session.

        Returns:
            List of employee IDs that should be marked OFFLINE.
        """
        now = time.time()
        offline = []

        for emp_id in known_employee_ids:
            last = self.last_seen.get(emp_id)
            if last is not None and (now - last) > self.offline_timeout:
                offline.append(emp_id)

        return offline