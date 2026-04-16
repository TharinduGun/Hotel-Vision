// ── Alert types ──────────────────────────────────────────────────────
// Extended to support both backend alert types and UI display categories
export type AlertType =
  | "security"
  | "anomaly"
  | "unauthorized"
  | "motion"
  | "cash"
  | "weapon"
  | "crowd";

export type Severity = "high" | "medium" | "low" | "critical";

export interface Alert {
  id: string;
  type: AlertType;
  title: string;
  description: string;
  location: string;
  timestamp: string;
  severity: Severity;
  isNew?: boolean;
  cameraId?: string;
}

// ── Camera types ─────────────────────────────────────────────────────
export interface Camera {
  id: string;
  name: string;
  location: string;
  image: string;
  isLive: boolean;
  hasAlert: boolean;
}

// ── Employee types ───────────────────────────────────────────────────
export type EmployeeStatus = "active" | "break" | "offline";

export interface Employee {
  id: string;
  name: string;
  role: string;
  avatar: string;
  location: string;
  status: EmployeeStatus;
  lastSeen: string;
  zone: string;
}

// ── Backend API response types (from /api/v1/live/state) ─────────────

export interface BackendAlert {
  id: string;
  type: string;       // "ZONE_INTRUSION" | "LOITERING" | "CASH_POCKET" | etc.
  severity: string;   // "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  title: string;
  description: string;
  cameraId: string;
  zone: string;
  ts: string;         // ISO datetime
  evidence: {
    imageUrl?: string;
    clipUrl?: string;
  };
}

export interface BackendEmployee {
  id: string;
  name: string;
  role: string;
  status: string;   // "ON_DUTY" | "BREAK" | "OFF_DUTY"
  lastSeen: string;  // ISO datetime
  location: string;
  zone: string;
}

export interface KPICard {
  value: number;
  total?: number;
  deltaPct?: number;
}

export interface DashboardSummary {
  ts: string;
  cards: Record<string, KPICard>;
}

export interface LiveState {
  summary: DashboardSummary;
  snapshots: Array<{ cameraId: string; snapshotUrl: string; ts: string }>;
  alerts: { items: BackendAlert[] };
  employees: { items: BackendEmployee[] };
  crowd?: {
    ts: string;
    summary: {
      totalEntries: number;
      totalExits: number;
      peakOccupancy: number;
      avgOccupancy: number;
      currentDensity: string;
      avgDwellSec: number;
      maxDwellSec: number;
      totalUniquePersons: number;
    };
  };
}