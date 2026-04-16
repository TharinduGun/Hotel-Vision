import { useState } from "react";
import { EmployeeRow } from "./EmployeeRow";
import { Users, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Employee } from "@/types";

interface EmployeeTrackingProps {
  employees?: Employee[];
}

export const EmployeeTracking = ({
  employees = [],
}: EmployeeTrackingProps) => {
  const [search, setSearch] = useState("");

  const filtered = search
    ? employees.filter(
        (e) =>
          e.name.toLowerCase().includes(search.toLowerCase()) ||
          e.role.toLowerCase().includes(search.toLowerCase())
      )
    : employees;

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
            {onBreakCount > 0 && (
              <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-warning/20 text-warning">
                {onBreakCount} Break
              </span>
            )}
          </div>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search employees..."
            className="pl-9 bg-muted/50 border-border/50"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="p-4 space-y-2">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-20 text-muted-foreground text-sm">
            {employees.length === 0
              ? "No tracked employees in this session"
              : "No matching employees"}
          </div>
        ) : (
          filtered.map((employee, index) => (
            <div
              key={employee.id}
              style={{ animationDelay: `${index * 50}ms` }}
              className="animate-fade-in"
            >
              <EmployeeRow {...employee} />
            </div>
          ))
        )}
      </div>
    </div>
  );
};