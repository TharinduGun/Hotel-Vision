import { useLiveSocket } from "@/hooks/useLiveSocket";
import { Header } from "@/components/dashboard/Header";
import { StatsCard } from "@/components/dashboard/StatsCard";
import { CameraGrid } from "@/components/dashboard/CameraGrid";
import { AlertsPanel } from "@/components/dashboard/AlertsPanel";
import { EmployeeTracking } from "@/components/dashboard/EmployeeTracking";

import { 
  Camera, 
  AlertTriangle, 
  Users, 
  ShieldCheck, 
  Activity, 
  Clock   
} from "lucide-react";

import { useDashboardData } from "@/hooks/useDashboardData";

// Fallback KPI cards shown when the backend is unreachable
const FALLBACK_CARDS = {
  camerasOnline:  { value: "—", total: 4 },
  activeAlerts:   { value: "—" },
  staffOnSite:    { value: "—" },
  securityScore:  { value: "—" },
  incidentsToday: { value: "—" },
  uptime:         { value: "—" },
};

const Index = () => {
  useLiveSocket();

  const { summary, isLoading } = useDashboardData();

  const cards = summary?.cards ?? FALLBACK_CARDS;
  return (
    <div className="min-h-screen bg-background grid-pattern">
      <Header />
      
      <main className="container mx-auto px-4 py-6 space-y-6">

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <StatsCard
            title="Cameras Online"
            value={`${cards.camerasOnline.value}/${cards.camerasOnline.total}`}
            icon={Camera}
          />

          <StatsCard
            title="Active Alerts"
            value={cards.activeAlerts.value}
            icon={AlertTriangle}
          />

          <StatsCard
            title="Staff On-Site"
            value={cards.staffOnSite.value}
            icon={Users}
          />

          <StatsCard
            title="Security Score"
            value={`${cards.securityScore.value}%`}
            icon={ShieldCheck}
          />

          <StatsCard
            title="Incidents Today"
            value={cards.incidentsToday.value}
            icon={Activity}
          />

          <StatsCard
            title="Uptime"
            value={`${cards.uptime.value}%`}
            icon={Clock}
          />

        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          <div className="lg:col-span-2">
            <CameraGrid />
          </div>
          
          <div className="lg:col-span-1 h-[600px]">
            <AlertsPanel />
          </div>

        </div>

        <EmployeeTracking />

      </main>
    </div>
  );
};

export default Index;

