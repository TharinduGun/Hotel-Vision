/* import { AlertItem } from "./AlertItem";
import { Bell, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mockAlerts } from "@/data/mockData";

/* const mockAlerts = [
  {
    id: "ALT-001",
    type: "security" as const,
    title: "Unauthorized Access Attempt",
    description: "Someone attempted to access restricted area near server room using invalid credentials.",
    location: "Floor 2 - Server Room",
    timestamp: "2 min ago",
    severity: "high" as const,
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

export const AlertsPanel = () => {
  return (
    <div className="glass-panel h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Live Alerts</h3>
          <span className="px-2 py-0.5 text-xs font-mono font-bold rounded-full bg-destructive/20 text-destructive">
            {mockAlerts.filter((a) => a.isNew).length} NEW
          </span>
        </div>
        <Button variant="ghost" size="sm" className="text-muted-foreground">
          <Filter className="w-4 h-4 mr-1" />
          Filter
        </Button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {mockAlerts.map((alert, index) => (
          <div
            key={alert.id}
            style={{ animationDelay: `${index * 100}ms` }}
            className="animate-fade-in"
          >
            <AlertItem {...alert} />
          </div>
        ))}
      </div>
    </div>
  );
};
 */
import { useState } from "react";
import { AlertItem } from "./AlertItem";
import { Bell, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mockAlerts } from "@/data/mockData";
import { Alert } from "@/types";

export const AlertsPanel = () => {
  const [alerts] = useState<Alert[]>(mockAlerts);

  const newAlertsCount = alerts.filter((a) => a.isNew).length;

  return (
    <div className="glass-panel h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Live Alerts</h3>
          <span className="px-2 py-0.5 text-xs font-mono font-bold rounded-full bg-destructive/20 text-destructive">
            {newAlertsCount} NEW
          </span>
        </div>
        <Button variant="ghost" size="sm" className="text-muted-foreground">
          <Filter className="w-4 h-4 mr-1" />
          Filter
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {alerts.map((alert, index) => (
          <div
            key={alert.id}
            style={{ animationDelay: `${index * 100}ms` }}
            className="animate-fade-in"
          >
            <AlertItem {...alert} />
          </div>
        ))}
      </div>
    </div>
  );
};