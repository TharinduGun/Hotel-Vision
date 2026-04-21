import { AlertItem } from "./AlertItem";
import { Bell, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert } from "@/types";

interface AlertsPanelProps {
  alerts?: Alert[];
}

export const AlertsPanel = ({ alerts = [] }: AlertsPanelProps) => {
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
        {alerts.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
            No active alerts
          </div>
        ) : (
          alerts.map((alert, index) => (
            <div
              key={alert.id}
              style={{ animationDelay: `${index * 100}ms` }}
              className="animate-fade-in"
            >
              <AlertItem {...alert} />
            </div>
          ))
        )}
      </div>
    </div>
  );
};