import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  ShieldAlert,
  UserX,
  Eye,
  Banknote,
  Crosshair,
  Users,
  Camera,
} from "lucide-react";
import { AlertType } from "@/types";

interface AlertItemProps {
  id: string;
  type: AlertType;
  title: string;
  description: string;
  location: string;
  timestamp: string;
  severity: "high" | "medium" | "low" | "critical";
  isNew?: boolean;
  hasEvidence?: boolean;
  onClick?: () => void;
}

const alertIcons: Record<AlertType, typeof AlertTriangle> = {
  security: ShieldAlert,
  anomaly: AlertTriangle,
  unauthorized: UserX,
  motion: Eye,
  cash: Banknote,
  weapon: Crosshair,
  crowd: Users,
};

const severityStyles = {
  critical: "border-l-destructive bg-destructive/10",
  high: "border-l-destructive bg-destructive/5",
  medium: "border-l-warning bg-warning/5",
  low: "border-l-primary bg-primary/5",
};

export const AlertItem = ({
  type,
  title,
  description,
  location,
  timestamp,
  severity,
  isNew = false,
  hasEvidence = false,
  onClick,
}: AlertItemProps) => {
  const Icon = alertIcons[type] || AlertTriangle;

  return (
    <div
      className={cn(
        "relative p-4 rounded-lg border-l-4 border border-border/50 transition-all duration-300 hover:bg-muted/30 cursor-pointer",
        severityStyles[severity] || severityStyles.low,
        isNew && "animate-fade-in"
      )}
      onClick={onClick}
    >
      {isNew && (
        <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-primary animate-pulse" />
      )}

      <div className="flex items-start gap-3">
        <div
          className={cn("p-2 rounded-lg", {
            "bg-destructive/20 text-destructive":
              severity === "critical" || severity === "high",
            "bg-warning/20 text-warning": severity === "medium",
            "bg-primary/20 text-primary": severity === "low",
          })}
        >
          <Icon className="w-4 h-4" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-semibold text-foreground truncate">
              {title}
            </h4>
            <span
              className={cn(
                "px-1.5 py-0.5 text-[10px] font-bold uppercase rounded",
                {
                  "bg-destructive/20 text-destructive":
                    severity === "critical" || severity === "high",
                  "bg-warning/20 text-warning": severity === "medium",
                  "bg-primary/20 text-primary": severity === "low",
                }
              )}
            >
              {severity}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
            {description}
          </p>
          <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
            <div className="flex items-center gap-2">
              <span>{location}</span>
              {hasEvidence && (
                <span className="flex items-center gap-1 text-primary">
                  <Camera className="w-3 h-3" />
                  <span>Image</span>
                </span>
              )}
            </div>
            <span>{timestamp}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
