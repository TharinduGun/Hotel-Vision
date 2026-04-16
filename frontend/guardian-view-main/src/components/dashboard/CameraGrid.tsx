import { useState } from "react";
import { CameraFeed } from "./CameraFeed";
import { VideoPlayerModal } from "./VideoPlayerModal";
import { Video, Grid3X3, LayoutGrid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mockCameras } from "@/data/mockData";
import { Camera } from "@/types";

const API_BASE = "http://localhost:8000/api/v1";

export const CameraGrid = () => {
  const [cameras] = useState<Camera[]>(mockCameras);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);

  return (
    <>
      <div className="glass-panel">
        <div className="flex items-center justify-between p-4 border-b border-border/50">
          <div className="flex items-center gap-2">
            <Video className="w-4 h-4 text-primary" />
            <h3 className="font-semibold text-foreground">Live Feeds</h3>
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-success/20 text-success">
              {cameras.filter((c) => c.isLive).length} Online
            </span>
          </div>

          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Grid3X3 className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-primary"
            >
              <LayoutGrid className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="p-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {cameras.map((feed, index) => (
              <div
                key={feed.id}
                style={{ animationDelay: `${index * 100}ms` }}
                className="animate-fade-in"
              >
                <CameraFeed
                  {...feed}
                  onClick={() => setSelectedCamera(feed)}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Video Player Modal */}
      <VideoPlayerModal
        isOpen={!!selectedCamera}
        onClose={() => setSelectedCamera(null)}
        cameraId={selectedCamera?.id || ""}
        cameraName={selectedCamera?.name || ""}
        videoUrl={
          selectedCamera
            ? `${API_BASE}/cameras/${selectedCamera.id}/video`
            : ""
        }
      />
    </>
  );
};