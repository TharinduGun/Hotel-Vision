import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  status?: "online" | "alert" | "warning";
  className?: string;
}

export const StatsCard = ({
  title,
  value,
  icon: Icon,
  trend,
  status,
  className,
}: StatsCardProps) => {
  return (
    <div
      className={cn(
        "glass-panel-glow p-6 animate-fade-in",
        className
      )}
    >
      <div className="relative z-10 flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-bold font-mono tracking-tight text-foreground">
              {value}
            </p>
            {status && (
              <span
                className={cn("status-dot", {
                  "status-online": status === "online",
                  "status-alert": status === "alert",
                  "status-warning": status === "warning",
                })}
              />
            )}
          </div>
          {trend && (
            <p
              className={cn("text-xs font-medium", {
                "text-success": trend.isPositive,
                "text-destructive": !trend.isPositive,
              })}
            >
              {trend.isPositive ? "↑" : "↓"} {Math.abs(trend.value)}% from yesterday
            </p>
          )}
        </div>
        <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
          <Icon className="w-6 h-6 text-primary" />
        </div>
      </div>
    </div>
  );
};
