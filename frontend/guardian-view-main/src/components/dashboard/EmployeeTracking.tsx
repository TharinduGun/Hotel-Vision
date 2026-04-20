import { useQuery } from "@tanstack/react-query";
import { EmployeeRow } from "./EmployeeRow";
import { Users, Search, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { mockEmployees } from "@/data/mockData";
import { Employee, EmployeeStatus } from "@/types";
import { fetchEmployees, ApiEmployee } from "@/services/api";

// Map backend status strings to frontend EmployeeStatus
function mapStatus(apiStatus: string): EmployeeStatus {
  if (apiStatus === "ON_DUTY") return "active";
  if (apiStatus === "BREAK") return "break";
  return "offline";
}

// Format ISO datetime to a human-readable relative time
function formatLastSeen(isoTimestamp: string): string {
  try {
    const diffMs = Date.now() - new Date(isoTimestamp).getTime();
    const mins = Math.floor(diffMs / 60_000);
    if (mins < 1) return "Now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  } catch {
    return "—";
  }
}

function toEmployee(api: ApiEmployee): Employee {
  return {
    id: api.id,
    name: api.name,
    role: api.role,
    avatar: "",          // backend does not provide avatars; bg-muted circle shown as fallback
    location: api.location,
    status: mapStatus(api.status),
    lastSeen: formatLastSeen(api.lastSeen),
    zone: api.zone,
  };
}

export const EmployeeTracking = () => {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["employees"],
    queryFn: fetchEmployees,
    refetchInterval: 10_000,
  });

  const employees: Employee[] =
    isLoading || isError || !data
      ? mockEmployees
      : data.items.map(toEmployee);

  const activeCount = employees.filter((e) => e.status === "active").length;
  const onBreakCount = employees.filter((e) => e.status === "break").length;

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
          {isLoading && (
            <Loader2 className="w-3 h-3 animate-spin text-muted-foreground ml-1" />
          )}
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
