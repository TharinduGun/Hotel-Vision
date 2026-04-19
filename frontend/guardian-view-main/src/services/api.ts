const BASE_URL = "http://localhost:8000/api/v1";

export async function fetchLiveState() {
  const res = await fetch(`${BASE_URL}/live/state`);
  if (!res.ok) throw new Error("Failed to fetch dashboard state");
  return res.json();
}

export async function fetchEmployees(): Promise<{ items: ApiEmployee[] }> {
  const res = await fetch(`${BASE_URL}/employees`);
  if (!res.ok) throw new Error("Failed to fetch employees");
  return res.json();
}

// Shape returned by GET /api/v1/employees
export interface ApiEmployee {
  id: string;
  name: string;
  role: string;
  status: string;   // "ON_DUTY" | "BREAK" | "OFF_DUTY"
  lastSeen: string; // ISO 8601 datetime
  location: string;
  zone: string;
}