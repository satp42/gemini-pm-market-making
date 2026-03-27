"use client";

import { useEffect, useRef } from "react";
import type { ActivityLogEntry } from "@/lib/types";

interface ActivityLogProps {
  logs: ActivityLogEntry[];
}

const typeConfig: Record<
  ActivityLogEntry["type"],
  { color: string; icon: string }
> = {
  order_placed: { color: "text-blue-400", icon: ">>>" },
  order_filled: { color: "text-emerald-400", icon: "[F]" },
  order_cancelled: { color: "text-gray-400", icon: "[X]" },
  risk_alert: { color: "text-red-400", icon: "!!!" },
  info: { color: "text-gray-500", icon: "---" },
};

const formatTimestamp = (ts: string): string => {
  const d = new Date(ts);
  return d.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

const ActivityLog = ({ logs }: ActivityLogProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs.length]);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider mb-3">
        Activity Log
      </h2>

      <div
        ref={containerRef}
        className="h-48 overflow-y-auto scrollbar-thin scrollbar-track-gray-900 scrollbar-thumb-gray-700"
      >
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-600 text-xs font-mono">
            Waiting for activity...
          </div>
        ) : (
          <div className="space-y-0.5">
            {logs.map((entry, i) => {
              const cfg = typeConfig[entry.type] ?? typeConfig.info;
              return (
                <div
                  key={`${entry.timestamp}-${i}`}
                  className="flex items-start gap-2 py-0.5 px-1 rounded hover:bg-gray-800/40 transition-colors"
                >
                  <span className="text-gray-600 font-mono text-xs shrink-0 w-16">
                    {formatTimestamp(entry.timestamp)}
                  </span>
                  <span
                    className={`font-mono text-xs shrink-0 w-6 ${cfg.color}`}
                  >
                    {cfg.icon}
                  </span>
                  {entry.symbol && (
                    <span className="text-yellow-500/80 font-mono text-xs shrink-0">
                      [{entry.symbol}]
                    </span>
                  )}
                  <span className="font-mono text-xs text-gray-300 break-all">
                    {entry.message}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default ActivityLog;
