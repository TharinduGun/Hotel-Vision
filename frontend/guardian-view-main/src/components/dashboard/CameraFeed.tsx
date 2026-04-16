import { cn } from "@/lib/utils";
import { Maximize2, Volume2, Circle, Play } from "lucide-react";

interface CameraFeedProps {
  id: string;
  name: string;
  location: string;
  image: string;
  isLive?: boolean;
  hasAlert?: boolean;
  className?: string;
  onClick?: () => void;
}

export const CameraFeed = ({
  id,
  name,
  location,
  image,
  isLive = true,
  hasAlert = false,
  className,
  onClick,
}: CameraFeedProps) => {
  const timestamp = new Date().toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      className={cn(
        "camera-feed group cursor-pointer transition-all duration-300 hover:border-primary/50 hover:shadow-lg",
        hasAlert && "border-destructive/50 animate-pulse-glow",
        className
      )}
      onClick={onClick}
    >
      <div className="relative aspect-video overflow-hidden">
        <img
          src={image}
          alt={name}
          className="w-full h-full object-cover filter grayscale contrast-110"
        />

        {/* Play button overlay on hover */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-black/30">
          <div className="w-14 h-14 rounded-full bg-primary/90 flex items-center justify-center shadow-lg backdrop-blur-sm">
            <Play className="w-6 h-6 text-primary-foreground ml-0.5" />
          </div>
        </div>
        
        {/* Scan line effect */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-30">
          <div className="w-full h-1 bg-gradient-to-b from-primary/50 to-transparent animate-scan" />
        </div>

        {/* Overlay info */}
        <div className="absolute inset-0 z-10 flex flex-col justify-between p-3 pointer-events-none">
          {/* Top bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isLive && (
                <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-destructive/80 backdrop-blur-sm">
                  <Circle className="w-2 h-2 fill-current text-foreground animate-pulse" />
                  <span className="text-[10px] font-bold font-mono text-destructive-foreground">
                    LIVE
                  </span>
                </div>
              )}
              {hasAlert && (
                <div className="px-2 py-1 rounded bg-warning/80 backdrop-blur-sm">
                  <span className="text-[10px] font-bold font-mono text-warning-foreground">
                    ALERT
                  </span>
                </div>
              )}
            </div>
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-auto">
              <button className="p-1.5 rounded bg-background/50 backdrop-blur-sm hover:bg-background/70 transition-colors">
                <Volume2 className="w-3 h-3 text-foreground" />
              </button>
              <button className="p-1.5 rounded bg-background/50 backdrop-blur-sm hover:bg-background/70 transition-colors">
                <Maximize2 className="w-3 h-3 text-foreground" />
              </button>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-medium text-foreground/90 bg-background/30 backdrop-blur-sm px-2 py-0.5 rounded">
                {id}
              </span>
              <span className="text-[10px] font-mono text-foreground/80 bg-background/30 backdrop-blur-sm px-2 py-0.5 rounded">
                {timestamp}
              </span>
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground/95">{name}</p>
              <p className="text-xs text-foreground/70">{location}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
