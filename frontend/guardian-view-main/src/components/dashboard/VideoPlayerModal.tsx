import { useRef, useEffect } from "react";
import { X, Maximize2, Minimize2, Play, Pause } from "lucide-react";
import { cn } from "@/lib/utils";

interface VideoPlayerModalProps {
  isOpen: boolean;
  onClose: () => void;
  cameraId: string;
  cameraName: string;
  videoUrl: string;
}

export const VideoPlayerModal = ({
  isOpen,
  onClose,
  cameraId,
  cameraName,
  videoUrl,
}: VideoPlayerModalProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (isOpen && videoRef.current) {
      videoRef.current.load();
    }
  }, [isOpen, videoUrl]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />

      {/* Modal */}
      <div
        className="relative z-10 w-full max-w-5xl mx-4 rounded-xl overflow-hidden border border-border/50 bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-card border-b border-border/50">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-primary/20">
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
              <span className="text-[10px] font-bold font-mono text-primary">
                RECORDED
              </span>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">
                {cameraName}
              </h3>
              <p className="text-[10px] font-mono text-muted-foreground">
                {cameraId} — Processed Output
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted/50 transition-colors text-muted-foreground hover:text-foreground"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Video Player */}
        <div className="relative bg-black">
          <video
            ref={videoRef}
            className="w-full aspect-video"
            controls
            autoPlay
            playsInline
            preload="metadata"
          >
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>

        {/* Footer */}
        <div className="px-4 py-2 bg-card border-t border-border/50 flex items-center justify-between">
          <span className="text-[10px] font-mono text-muted-foreground">
            Analytics output with detection overlays
          </span>
          <span className="text-[10px] font-mono text-muted-foreground">
            Use ← → keys to seek • Space to play/pause
          </span>
        </div>
      </div>
    </div>
  );
};
