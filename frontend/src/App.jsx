import { useEffect, useState } from "react";


const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function App() {
  const [health, setHealth] = useState({ state: "loading", message: "Checking API connection…" });

  useEffect(() => {
    const controller = new AbortController();

    async function checkHealth() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/health`, { signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        setHealth({ state: "connected", message: `${data.project} API: ${data.status}` });
      } catch (error) {
        if (error.name !== "AbortError") {
          setHealth({ state: "error", message: "Backend is not reachable. Start the FastAPI server and retry." });
        }
      }
    }

    checkHealth();
    return () => controller.abort();
  }, []);

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Fictional goods. Real engineering.</p>
        <h1>Impossible Market</h1>
        <p className="tagline">The marketplace for things you should never be able to buy.</p>
        <div className={`status status--${health.state}`} role="status" aria-live="polite">
          <span className="status__dot" aria-hidden="true" />
          {health.message}
        </div>
      </section>
    </main>
  );
}
