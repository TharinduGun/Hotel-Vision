import { useQuery } from "@tanstack/react-query";
import { fetchLiveState } from "@/services/api";
import {
  Alert,
  AlertType,
  Severity,
  Employee,
  EmployeeStatus,
  BackendAlert,
  BackendEmployee,
} from "@/types";

// ── Backend → UI mapping helpers ─────────────────────────────────────

/**
 * Map backend alert type strings to the UI's AlertType category.
 */
function mapAlertType(backendType: string): AlertType {
  const t = backendType.toUpperCase();
  if (t.includes("CASH") || t.includes("UNREGISTERED")) return "cash";
  if (t.includes("WEAPON") || t.includes("GUN")) return "weapon";
  if (t.includes("ZONE_INTRUSION") || t.includes("UNAUTHORIZED")) return "unauthorized";
  if (t.includes("LOITER") || t.includes("MOTION")) return "motion";
  if (t.includes("CROWD") || t.includes("DENSITY") || t.includes("OCCUPANCY")) return "crowd";
  if (t.includes("SECURITY") || t.includes("EMERGENCY")) return "security";
  return "anomaly";
}

/**
 * Map backend severity (uppercase) → UI severity (lowercase).
 */
function mapSeverity(backendSeverity: string): Severity {
  const s = backendSeverity.toLowerCase();
  if (s === "critical") return "critical";
  if (s === "high") return "high";
  if (s === "medium") return "medium";
  return "low";
}

/**
 * Format ISO datetime string into a human-friendly relative time (e.g. "2 min ago").
 */
function formatRelativeTime(isoString: string): string {
  try {
    const dt = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - dt.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 60) return "Just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    return `${Math.floor(diffSec / 86400)}d ago`;
  } catch {
    return isoString;
  }
}

/**
 * Transform a backend Alert into a UI-ready Alert.
 */
function transformAlert(ba: BackendAlert, index: number): Alert {
  return {
    id: ba.id,
    type: mapAlertType(ba.type),
    title: ba.title,
    description: ba.description,
    location: `${ba.zone} • ${ba.cameraId}`,
    timestamp: formatRelativeTime(ba.ts),
    severity: mapSeverity(ba.severity),
    isNew: index < 3, // Top 3 most recent are "new"
    cameraId: ba.cameraId,
  };
}

/**
 * Transform a backend Employee into a UI-ready Employee.
 */
function transformEmployee(be: BackendEmployee): Employee {
  const statusMap: Record<string, EmployeeStatus> = {
    ON_DUTY: "active",
    BREAK: "break",
    OFF_DUTY: "offline",
  };

  return {
    id: be.id,
    name: be.name,
    role: be.role,
    avatar: "", // No avatar from backend — will use initials fallback
    location: be.location,
    status: statusMap[be.status] || "offline",
    lastSeen: formatRelativeTime(be.lastSeen),
    zone: be.zone,
  };
}

// ── Hook ──────────────────────────────────────────────────────────────

export function useDashboardData() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchLiveState,
    refetchInterval: 5000, // fallback polling (5s)
  });

  // Transform backend data into UI-ready shapes
  const alerts: Alert[] = (data?.alerts?.items || []).map(
    (ba: BackendAlert, i: number) => transformAlert(ba, i)
  );

  const employees: Employee[] = (data?.employees?.items || []).map(
    (be: BackendEmployee) => transformEmployee(be)
  );

  return {
    summary: data?.summary,
    snapshots: data?.snapshots || [],
    alerts,
    employees,
    crowd: data?.crowd,
    isLoading,
    error,
  };
}