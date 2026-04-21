import { useQuery } from "@tanstack/react-query";
import { Users, AlertTriangle, ArrowRightLeft, Clock, Activity, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface CrowdInsightsResponse {
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
  recentFootfall: Array<{
    trackId: number;
    direction: string;
    timestampSec: number;
    frameIdx: number;
    edge: string;
    positionX: number;
    positionY: number;
  }>;
  recentDwells: any[];
  heatmapUrl: string | null;
}

const fetchCrowdInsights = async (): Promise<CrowdInsightsResponse> => {
  const res = await fetch("http://localhost:8000/api/v1/crowd/insights");
  if (!res.ok) throw new Error("Failed to fetch crowd insights");
  return res.json();
};

export const CrowdInsightsPanel = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ["crowdInsights"],
    queryFn: fetchCrowdInsights,
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <div className="glass-panel h-full flex flex-col items-center justify-center p-6 text-muted-foreground">
        <Loader2 className="w-6 h-6 animate-spin mb-2" />
        <span className="text-sm font-mono">Loading crowd insights...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="glass-panel h-full flex flex-col items-center justify-center p-6 text-destructive">
        <AlertTriangle className="w-6 h-6 mb-2" />
        <span className="text-sm font-mono">Failed to load crowd data</span>
      </div>
    );
  }

  const s = data.summary;
  const isCritical = s.currentDensity === "critical";
  const isHigh = s.currentDensity === "high";

  return (
    <div className="glass-panel h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Crowd Insights</h3>
          <span
            className={cn(
              "px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded-full ml-2",
              isCritical
                ? "bg-destructive/20 text-destructive"
                : isHigh
                ? "bg-orange-500/20 text-orange-500"
                : "bg-primary/20 text-primary"
            )}
          >
            {s.currentDensity} Density
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-background/50 rounded-lg p-3 border border-border/50">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <ArrowRightLeft className="w-3.5 h-3.5" />
              <span className="text-xs font-mono">Footfall</span>
            </div>
            <div className="text-lg font-semibold flex items-center gap-2">
              <span className="text-emerald-500">+{s.totalEntries}</span>
              <span className="text-destructive">-{s.totalExits}</span>
            </div>
          </div>

          <div className="bg-background/50 rounded-lg p-3 border border-border/50">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Activity className="w-3.5 h-3.5" />
              <span className="text-xs font-mono">Occupancy</span>
            </div>
            <div className="text-lg font-semibold">
              <span className="text-foreground">{Math.round(s.avgOccupancy)} avg</span>
              <span className="text-muted-foreground text-sm ml-2">({s.peakOccupancy} peak)</span>
            </div>
          </div>

          <div className="bg-background/50 rounded-lg p-3 border border-border/50">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Clock className="w-3.5 h-3.5" />
              <span className="text-xs font-mono">Avg Dwell</span>
            </div>
            <div className="text-lg font-semibold text-foreground">
              {s.avgDwellSec.toFixed(1)}s
            </div>
          </div>

          <div className="bg-background/50 rounded-lg p-3 border border-border/50">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Users className="w-3.5 h-3.5" />
              <span className="text-xs font-mono">Unique Visitors</span>
            </div>
            <div className="text-lg font-semibold text-foreground">
              {s.totalUniquePersons}
            </div>
          </div>
        </div>

        {/* Heatmap */}
        <div>
          <h4 className="text-xs font-mono text-muted-foreground mb-2 flex items-center gap-2">
            Movement Heatmap
          </h4>
          <div className="relative rounded-lg overflow-hidden border border-border/50 bg-black aspect-video flex items-center justify-center">
            {data.heatmapUrl ? (
              <img
                src={`http://localhost:8000${data.heatmapUrl}?t=${data.ts}`}
                alt="Crowd Heatmap"
                className="w-full h-full object-cover opacity-80 mix-blend-screen"
              />
            ) : (
              <span className="text-xs font-mono text-muted-foreground">Waiting for heatmap data...</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
