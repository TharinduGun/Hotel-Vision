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