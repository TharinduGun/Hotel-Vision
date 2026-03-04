/* import { CameraFeed } from "./CameraFeed";
import { Video, Grid3X3, LayoutGrid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mockCameras } from "@/data/mockData";

import cameraLobby from "@/assets/camera-lobby.jpg";
import cameraParking from "@/assets/camera-parking.jpg";
import cameraHallway from "@/assets/camera-hallway.jpg";
import cameraService from "@/assets/camera-service.jpg";

/* const cameraFeeds = [
  {
    id: "CAM-01",
    name: "Main Lobby",
    location: "Ground Floor - Entrance",
    image: cameraLobby,
    isLive: true,
    hasAlert: false,
  },
  {
    id: "CAM-02",
    name: "Parking Lot A",
    location: "Outdoor - North Side",
    image: cameraParking,
    isLive: true,
    hasAlert: true,
  },
  {
    id: "CAM-03",
    name: "Hallway Floor 3",
    location: "Floor 3 - East Wing",
    image: cameraHallway,
    isLive: true,
    hasAlert: false,
  },
  {
    id: "CAM-04",
    name: "Service Entrance",
    location: "Back - Loading Bay",
    image: cameraService,
    isLive: true,
    hasAlert: false,
  },
];
 
export const CameraGrid = () => {
  return (
    <div className="glass-panel">
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Video className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Live Feeds</h3>
          <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-success/20 text-success">
            {mockCameras.length} Online
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <Grid3X3 className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-primary">
            <LayoutGrid className="w-4 h-4" />
          </Button>
        </div>
      </div>
      
      <div className="p-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {mockCameras.map((feed, index) => (
            <div
              key={feed.id}
              style={{ animationDelay: `${index * 100}ms` }}
              className="animate-fade-in"
            >
              <CameraFeed {...feed} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}; */
import { useState } from "react";
import { CameraFeed } from "./CameraFeed";
import { Video, Grid3X3, LayoutGrid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mockCameras } from "@/data/mockData";
import { Camera } from "@/types";

export const CameraGrid = () => {
  const [cameras] = useState<Camera[]>(mockCameras);

  return (
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
              <CameraFeed {...feed} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};