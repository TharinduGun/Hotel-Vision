import { useQuery } from "@tanstack/react-query";
import { ParkingSquare, Loader2, AlertTriangle, Clock, Car } from "lucide-react";
import { fetchParkingAnalytics } from "@/services/api";
import { ParkingAlert, ParkingResponse, ParkingStatus } from "@/types";
import { cn } from "@/lib/utils";

// ── helpers ────────────────────────────────────────────────────────────

function formatDwell(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function formatAlertTime(iso: string): string {
  try {
    const diffMs = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diffMs / 60_000);
    if (mins < 1) return "Now";
    if (mins < 60) return `${mins}m ago`;
    const h = Math.floor(mins / 60);
    return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`;
  } catch {
    return "—";
  }
}

function alertLabel(eventType: string): string {
  if (eventType === "parking_full") return "Lot Full";
  if (eventType === "parking_limited") return "Lot Limited";
  if (eventType === "long_dwell") return "Long Dwell";
  return eventType;
}

// ── Occupancy bar ──────────────────────────────────────────────────────

function OccupancyBar({ pct, status }: { pct: number; status: string }) {
  const color =
    status === "full"
      ? "bg-destructive"
      : status === "limited"
      ? "bg-warning"
      : "bg-success";

  return (
    <div className="w-full bg-muted/40 rounded-full h-3 overflow-hidden">
      <div
        className={cn("h-full rounded-full transition-all duration-500", color)}
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  );
}

// ── Alert row ──────────────────────────────────────────────────────────

function AlertRow({ alert }: { alert: ParkingAlert }) {
  const sevColor =
    alert.severity === "high"
      ? "text-destructive bg-destructive/10"
      : alert.severity === "medium"
      ? "text-warning bg-warning/10"
      : "text-muted-foreground bg-muted/20";

  return (
    <div className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
      <div className="flex items-center gap-2">
        <span className={cn("px-2 py-0.5 text-xs font-mono rounded-full", sevColor)}>
          {alertLabel(alert.eventType)}
        </span>
        {alert.dwellMinutes != null && (
          <span className="text-xs text-muted-foreground">
            {Math.round(alert.dwellMinutes)}m dwell
          </span>
        )}
        {alert.occupancyPct != null && alert.eventType !== "long_dwell" && (
          <span className="text-xs text-muted-foreground">
            {alert.occupancyPct.toFixed(0)}%
          </span>
        )}
      </div>
      <span className="text-xs text-muted-foreground">{formatAlertTime(alert.ts)}</span>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────

const FALLBACK: ParkingResponse = {
  ts: new Date().toISOString(),
  current: {
    occupied: 0,
    available: 0,
    capacity: 0,
    occupancyPct: 0,
    status: "available",
    avgDwellSeconds: 0,
    totalVehiclesSeen: 0,
    vehicleTypes: {},
  },
  recentAlerts: [],
  totalAlerts: 0,
};

export const ParkingPanel = () => {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["parking"],
    queryFn: fetchParkingAnalytics,
    refetchInterval: 15_000,
  });

  const { current, recentAlerts, totalAlerts }: ParkingResponse =
    isLoading || isError || !data ? FALLBACK : data;

  const statusColor =
    current.status === "full"
      ? "text-destructive"
      : current.status === "limited"
      ? "text-warning"
      : "text-success";

  const vehicleEntries = Object.entries(current.vehicleTypes ?? {});

  return (
    <div className="glass-panel">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <ParkingSquare className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Parking Lot</h3>
          <span className={cn("px-2 py-0.5 text-xs font-mono rounded-full font-bold uppercase", statusColor,
            current.status === "full" ? "bg-destructive/20" :
            current.status === "limited" ? "bg-warning/20" : "bg-success/20"
          )}>
            {current.status}
          </span>
          {isLoading && <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />}
        </div>
        <span className="text-xs text-muted-foreground font-mono">CAM-02</span>
      </div>

      <div className="p-4 space-y-5">
        {/* Occupancy gauge */}
        <div className="space-y-2">
          <div className="flex items-end justify-between">
            <span className="text-4xl font-bold font-mono tracking-tight text-foreground">
              {current.occupied}
              <span className="text-xl text-muted-foreground">/{current.capacity}</span>
            </span>
            <span className={cn("text-2xl font-bold font-mono", statusColor)}>
              {current.occupancyPct.toFixed(0)}%
            </span>
          </div>
          <OccupancyBar pct={current.occupancyPct} status={current.status} />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{current.occupied} occupied</span>
            <span>{current.available} available</span>
          </div>
        </div>

        {/* Metrics row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-muted/30 rounded-lg p-3 space-y-1">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Clock className="w-3 h-3" />
              <span className="text-xs">Avg Dwell</span>
            </div>
            <p className="text-lg font-bold font-mono text-foreground">
              {formatDwell(current.avgDwellSeconds)}
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 space-y-1">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Car className="w-3 h-3" />
              <span className="text-xs">Total Seen</span>
            </div>
            <p className="text-lg font-bold font-mono text-foreground">
              {current.totalVehiclesSeen}
            </p>
          </div>
        </div>

        {/* Vehicle type breakdown */}
        {vehicleEntries.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground font-medium">Vehicle Types</p>
            <div className="flex flex-wrap gap-2">
              {vehicleEntries.map(([type, count]) => (
                <span
                  key={type}
                  className="px-2 py-0.5 text-xs font-mono rounded-full bg-primary/10 text-primary border border-primary/20"
                >
                  {type}: {count}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Recent alerts */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1 text-muted-foreground">
              <AlertTriangle className="w-3 h-3" />
              <span className="text-xs font-medium">Recent Alerts</span>
            </div>
            {totalAlerts > 0 && (
              <span className="text-xs font-mono text-muted-foreground">{totalAlerts} total</span>
            )}
          </div>
          <div className="min-h-[60px]">
            {recentAlerts.length === 0 ? (
              <p className="text-xs text-muted-foreground py-4 text-center">No alerts</p>
            ) : (
              recentAlerts.slice(-5).reverse().map((alert, i) => (
                <AlertRow key={i} alert={alert} />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
