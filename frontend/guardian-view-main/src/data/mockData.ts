import { Alert, Camera, Employee } from "@/types";
import cameraLobby from "@/assets/camera-lobby.jpg";
import cameraParking from "@/assets/camera-parking.jpg";
import cameraHallway from "@/assets/camera-hallway.jpg";
import cameraService from "@/assets/camera-service.jpg";

export const mockAlerts: Alert[] = [
  {
    id: "ALT-001",
    type: "security",
    title: "Unauthorized Access Attempt",
    description: "Someone attempted to access restricted area...",
    location: "Floor 2 - Server Room",
    timestamp: "2 min ago",
    severity: "high",
    isNew: true,
  },
  {
    id: "ALT-002",
    type: "motion" as const,
    title: "Unusual Motion Detected",
    description: "Extended loitering detected in parking lot section B for over 15 minutes.",
    location: "Parking Lot - Zone B",
    timestamp: "8 min ago",
    severity: "medium" as const,
    isNew: true,
  },
  {
    id: "ALT-003",
    type: "anomaly" as const,
    title: "Camera Feed Disruption",
    description: "Brief signal interruption on camera CAM-12. Feed restored automatically.",
    location: "Pool Area",
    timestamp: "23 min ago",
    severity: "low" as const,
    isNew: false,
  },
  {
    id: "ALT-004",
    type: "unauthorized" as const,
    title: "Staff Badge Not Detected",
    description: "Employee entered kitchen area without scanning badge at entry point.",
    location: "Kitchen - Main Entry",
    timestamp: "45 min ago",
    severity: "medium" as const,
    isNew: false,
  },
  {
    id: "ALT-005",
    type: "security" as const,
    title: "Emergency Exit Opened",
    description: "Emergency exit door was opened without alarm authorization.",
    location: "Floor 1 - East Wing",
    timestamp: "1 hr ago",
    severity: "high" as const,
    isNew: false,
  },
];

export const mockCameras: Camera[] = [
  {
    id: "CAM-01",
    name: "Main Lobby",
    location: "Ground Floor - Entrance",
    image: cameraLobby,
    isLive: true,
    hasAlert: false,
  },
   {
    id: "CAM-02",
    name: "Parking Lot A",
    location: "Outdoor - North Side",
    image: cameraParking,
    isLive: true,
    hasAlert: true,
  },
  {
    id: "CAM-03",
    name: "Hallway Floor 3",
    location: "Floor 3 - East Wing",
    image: cameraHallway,
    isLive: true,
    hasAlert: false,
  },
  {
    id: "CAM-04",
    name: "Service Entrance",
    location: "Back - Loading Bay",
    image: cameraService,
    isLive: true,
    hasAlert: false,
  },
];

export const mockEmployees: Employee[] = [
  { id: "E001", name: "Anabel",    role: "Cashier", avatar: "", location: "POS Station 1", status: "active",  lastSeen: "Now",     zone: "Cashier 1" },
  { id: "E002", name: "Irving",    role: "Cashier", avatar: "", location: "POS Station 2", status: "active",  lastSeen: "Now",     zone: "Cashier 2" },
  { id: "E003", name: "Rosa",      role: "Cashier", avatar: "", location: "—",             status: "offline", lastSeen: "—",       zone: "—"         },
  { id: "E004", name: "Yareht",    role: "Cashier", avatar: "", location: "Break Room",    status: "break",   lastSeen: "10m ago", zone: "Break"     },
  { id: "E005", name: "Aurora",    role: "Cashier", avatar: "", location: "—",             status: "offline", lastSeen: "—",       zone: "—"         },
  { id: "E006", name: "Alejandra", role: "Cashier", avatar: "", location: "POS Station 1", status: "active",  lastSeen: "Now",     zone: "Cashier 1" },
  { id: "E007", name: "Lilia",     role: "Cashier", avatar: "", location: "—",             status: "offline", lastSeen: "—",       zone: "—"         },
  { id: "E008", name: "Sandra",    role: "Cashier", avatar: "", location: "POS Station 2", status: "active",  lastSeen: "2m ago",  zone: "Cashier 2" },
];