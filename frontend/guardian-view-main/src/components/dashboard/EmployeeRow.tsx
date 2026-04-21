import { cn } from "@/lib/utils";
import { MapPin, Clock } from "lucide-react";

interface EmployeeRowProps {
  id: string;
  name: string;
  role: string;
  avatar: string;
  location: string;
  status: "active" | "break" | "offline";
  lastSeen: string;
  zone: string;
}

const statusStyles = {
  active: {
    dot: "status-online",
    text: "text-success",
    label: "On Duty",
  },
  break: {
    dot: "status-warning",
    text: "text-warning",
    label: "On Break",
  },
  offline: {
    dot: "bg-muted-foreground",
    text: "text-muted-foreground",
    label: "Offline",
  },
};

export const EmployeeRow = ({
  id,
  name,
  role,
  avatar,
  location,
  status,
  lastSeen,
  zone,
}: EmployeeRowProps) => {
  const styles = statusStyles[status];

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg border border-border/30 bg-card/50 hover:bg-muted/30 transition-colors cursor-pointer">
      <div className="relative">
        <div className="w-10 h-10 rounded-full bg-muted overflow-hidden flex items-center justify-center">
          {avatar ? (
            <img src={avatar} alt={name} className="w-full h-full object-cover" />
          ) : (
            <span className="text-sm font-bold text-muted-foreground">
              {name.split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
            </span>
          )}
        </div>
        <span className={cn("status-dot absolute -bottom-0.5 -right-0.5 w-3 h-3 border-2 border-card", styles.dot)} />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-foreground truncate">{name}</h4>
          <span className="text-[10px] font-mono text-muted-foreground">#{id}</span>
        </div>
        <p className="text-xs text-muted-foreground">{role}</p>
      </div>
      
      <div className="hidden sm:flex flex-col items-end gap-1">
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <MapPin className="w-3 h-3" />
          <span>{location}</span>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-muted text-muted-foreground">
          {zone}
        </span>
      </div>
      
      <div className="flex flex-col items-end gap-1">
        <span className={cn("text-xs font-medium", styles.text)}>
          {styles.label}
        </span>
        <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
          <Clock className="w-3 h-3" />
          <span>{lastSeen}</span>
        </div>
      </div>
    </div>
  );
};
