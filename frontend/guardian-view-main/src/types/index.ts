export type AlertType = "security" | "anomaly" | "unauthorized" | "motion";

export type Severity = "high" | "medium" | "low";

export interface Alert {
  id: string;
  type: AlertType;
  title: string;
  description: string;
  location: string;
  timestamp: string;
  severity: Severity;
  isNew?: boolean;
}

export interface Camera {
  id: string;
  name: string;
  location: string;
  image: string;
  isLive: boolean;
  hasAlert: boolean;
}

// ── Parking ────────────────────────────────────────────────────────────

export type ParkingLotStatus = "available" | "limited" | "full";

export interface ParkingStatus {
  occupied: number;
  available: number;
  capacity: number;
  occupancyPct: number;
  status: ParkingLotStatus;
  avgDwellSeconds: number;
  totalVehiclesSeen: number;
  vehicleTypes: Record<string, number>;
}

export interface ParkingAlert {
  eventType: "parking_full" | "parking_limited" | "long_dwell";
  severity: string;
  cameraId: string;
  ts: string;
  occupancyPct?: number;
  vehicleId?: number;
  dwellMinutes?: number;
}

export interface ParkingResponse {
  ts: string;
  current: ParkingStatus;
  recentAlerts: ParkingAlert[];
  totalAlerts: number;
}

// ── Employee ───────────────────────────────────────────────────────────

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