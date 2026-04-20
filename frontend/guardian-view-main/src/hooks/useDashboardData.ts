import { useQuery } from "@tanstack/react-query";
import { fetchLiveState } from "@/services/api";

export function useDashboardData() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchLiveState,
    refetchInterval: 5000, // fallback polling (5s)
  });

  return {
    summary: data?.summary,
    snapshots: data?.snapshots || [],
    alerts: data?.alerts?.items || [],
    employees: data?.employees?.items || [],
    isLoading,
    error,
  };
}