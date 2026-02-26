/* import { EmployeeRow } from "./EmployeeRow";
import { Users, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { mockEmployees } from "@/data/mockData";

/* const mockEmployees = [
  {
    id: "E001",
    name: "Sarah Johnson",
    role: "Front Desk Manager",
    avatar: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=100&h=100&fit=crop&crop=face",
    location: "Main Lobby",
    status: "active" as const,
    lastSeen: "Now",
    zone: "Zone A",
  },
  {
    id: "E002",
    name: "Michael Chen",
    role: "Security Officer",
    avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=face",
    location: "Floor 3",
    status: "active" as const,
    lastSeen: "2m ago",
    zone: "Zone C",
  },
  {
    id: "E003",
    name: "Emily Rodriguez",
    role: "Housekeeping Lead",
    avatar: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=100&h=100&fit=crop&crop=face",
    location: "Break Room",
    status: "break" as const,
    lastSeen: "15m ago",
    zone: "Zone B",
  },
  {
    id: "E004",
    name: "James Wilson",
    role: "Maintenance Tech",
    avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&h=100&fit=crop&crop=face",
    location: "Basement",
    status: "active" as const,
    lastSeen: "5m ago",
    zone: "Zone D",
  },
  {
    id: "E005",
    name: "Lisa Park",
    role: "Night Auditor",
    avatar: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop&crop=face",
    location: "—",
    status: "offline" as const,
    lastSeen: "3h ago",
    zone: "—",
  },
]; 

export const EmployeeTracking = () => {
  const activeCount = mockEmployees.filter((e) => e.status === "active").length;
  const onBreakCount = mockEmployees.filter((e) => e.status === "break").length;

  return (
    <div className="glass-panel">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">Employee Tracking</h3>
          <div className="flex items-center gap-2 ml-2">
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-success/20 text-success">
              {activeCount} Active
            </span>
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-warning/20 text-warning">
              {onBreakCount} Break
            </span>
          </div>
        </div>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search employees..."
            className="pl-9 bg-muted/50 border-border/50"
          />
        </div>
      </div>
      
      <div className="p-4 space-y-2">
        {mockEmployees.map((employee, index) => (
          <div
            key={employee.id}
            style={{ animationDelay: `${index * 50}ms` }}
            className="animate-fade-in"
          >
            <EmployeeRow {...employee} />
          </div>
        ))}
      </div>
    </div>
  );
};
 */

import { useState } from "react";
import { EmployeeRow } from "./EmployeeRow";
import { Users, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { mockEmployees } from "@/data/mockData";
import { Employee } from "@/types";

export const EmployeeTracking = () => {
  const [employees] = useState<Employee[]>(mockEmployees);

  const activeCount = employees.filter((e) => e.status === "active").length;
  const onBreakCount = employees.filter((e) => e.status === "break").length;

  return (
    <div className="glass-panel">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-primary" />
          <h3 className="font-semibold text-foreground">
            Employee Tracking
          </h3>
          <div className="flex items-center gap-2 ml-2">
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-success/20 text-success">
              {activeCount} Active
            </span>
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-warning/20 text-warning">
              {onBreakCount} Break
            </span>
          </div>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search employees..."
            className="pl-9 bg-muted/50 border-border/50"
          />
        </div>
      </div>

      <div className="p-4 space-y-2">
        {employees.map((employee, index) => (
          <div
            key={employee.id}
            style={{ animationDelay: `${index * 50}ms` }}
            className="animate-fade-in"
          >
            <EmployeeRow {...employee} />
          </div>
        ))}
      </div>
    </div>
  );
};