import { useEffect } from "react";
import { X, AlertTriangle } from "lucide-react";
import { Alert } from "@/types";
import { cn } from "@/lib/utils";

interface AlertSnapshotModalProps {
  isOpen: boolean;
  onClose: () => void;
  alert: Alert | null;
}

export const AlertSnapshotModal = ({
  isOpen,
  onClose,
  alert,
}: AlertSnapshotModalProps) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }
  }, [isOpen, onClose]);

  if (!isOpen || !alert) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl rounded-xl overflow-hidden border border-border/50 bg-card shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 bg-card border-b border-border/50">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "p-2 rounded-lg",
                alert.severity === "critical"
                  ? "bg-destructive/20 text-destructive"
                  : alert.severity === "high"
                  ? "bg-orange-500/20 text-orange-500"
                  : "bg-yellow-500/20 text-yellow-500"
              )}
            >
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">
                {alert.title}
              </h3>
              <p className="text-[10px] font-mono text-muted-foreground flex gap-2">
                <span>{alert.location}</span>
                <span>•</span>
                <span>{alert.timestamp}</span>
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

        <div className="relative bg-black flex-1 min-h-[400px] flex items-center justify-center p-4">
          {alert.evidence?.imageUrl ? (
            <img
              src={`http://localhost:8000${alert.evidence.imageUrl}`}
              alt="Alert Evidence"
              className="max-w-full max-h-[70vh] object-contain rounded border border-border"
            />
          ) : (
            <div className="text-muted-foreground font-mono text-sm flex flex-col items-center">
              <span className="text-2xl mb-2">📷</span>
              No evidence image available
            </div>
          )}
        </div>

        <div className="px-4 py-3 bg-card border-t border-border/50 flex justify-between items-center">
          <p className="text-sm text-foreground">{alert.description}</p>
          <span className="text-[10px] font-mono text-muted-foreground bg-secondary px-2 py-1 rounded">
            {alert.type}
          </span>
        </div>
      </div>
    </div>
  );
};
