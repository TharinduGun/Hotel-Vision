import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

export function useLiveSocket() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/live");

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.kind === "alert_new") {
        queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      }

      if (message.kind === "summary_update") {
        queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      }
    };

    return () => ws.close();
  }, [queryClient]);
}