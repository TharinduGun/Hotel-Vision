const BASE_URL = "http://localhost:8000/api/v1";

export async function fetchLiveState() {
  const res = await fetch(`${BASE_URL}/live/state`);
  if (!res.ok) throw new Error("Failed to fetch dashboard state");
  return res.json();
}