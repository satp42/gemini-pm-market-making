"use client";

import { useCallback, useState } from "react";
import type { BotStatus } from "@/lib/types";
import { ApiRequestError, startBot, stopBot } from "@/lib/api";

interface StatusBarProps {
  status: BotStatus | null;
  isConnected: boolean;
}

const formatUptime = (seconds: number): string => {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

const StatusBar = ({ status, isConnected }: StatusBarProps) => {
  const [toggling, setToggling] = useState(false);

  const handleToggle = useCallback(async () => {
    if (!status || toggling) return;
    setToggling(true);
    try {
      if (status.running) {
        await stopBot();
      } else {
        await startBot();
      }
    } catch (err) {
      if (err instanceof ApiRequestError && err.status === 409) return;
      console.error("Failed to toggle bot:", err);
    } finally {
      setToggling(false);
    }
  }, [status, toggling]);

  const running = status?.running ?? false;
  const environment = status?.environment ?? "sandbox";

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      {/* Left: Bot status + environment */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${running ? "bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]" : "bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]"}`}
          />
          <span className="text-sm font-medium text-gray-200">
            {running ? "Running" : "Stopped"}
          </span>
        </div>

        <span
          className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-bold uppercase tracking-wider ${
            environment === "live"
              ? "bg-red-900/60 text-red-300 border border-red-700"
              : "bg-yellow-900/60 text-yellow-300 border border-yellow-700"
          }`}
        >
          {environment}
        </span>
      </div>

      {/* Center: Uptime + Active Markets */}
      <div className="flex items-center gap-6 text-sm text-gray-400">
        <div>
          <span className="text-gray-500 mr-1">Uptime:</span>
          <span className="font-mono text-gray-200">
            {status ? formatUptime(status.uptime) : "--:--:--"}
          </span>
        </div>
        <div>
          <span className="text-gray-500 mr-1">Markets:</span>
          <span className="font-mono text-gray-200">
            {status?.activeMarkets ?? 0}
          </span>
        </div>
      </div>

      {/* Right: Controls + WS indicator */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleToggle}
          disabled={toggling || !status}
          className={`px-4 py-1.5 rounded text-xs font-semibold uppercase tracking-wider transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            running
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-emerald-600 hover:bg-emerald-700 text-white"
          }`}
        >
          {toggling ? "..." : running ? "Stop" : "Start"}
        </button>

        <div className="flex items-center gap-1.5" title={isConnected ? "WebSocket connected" : "WebSocket disconnected"}>
          <span
            className={`inline-block h-2 w-2 rounded-full ${isConnected ? "bg-emerald-400" : "bg-gray-600 animate-pulse"}`}
          />
          <span className="text-xs text-gray-500">WS</span>
        </div>
      </div>
    </header>
  );
};

export default StatusBar;
