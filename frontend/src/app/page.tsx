"use client";

import { useEffect, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { getMarkets, getPositions, getStatus } from "@/lib/api";
import type { BotStatus, MarketData, Position } from "@/lib/types";

import StatusBar from "@/components/StatusBar";
import PnlChart from "@/components/PnlChart";
import InventoryHeatmap from "@/components/InventoryHeatmap";
import ReservationPriceChart from "@/components/ReservationPriceChart";
import ActivityLog from "@/components/ActivityLog";

export default function DashboardPage() {
  const ws = useWebSocket();

  // Initial REST data (used before first WS tick arrives)
  const [initialStatus, setInitialStatus] = useState<BotStatus | null>(null);
  const [initialMarkets, setInitialMarkets] = useState<MarketData[]>([]);
  const [initialPositions, setInitialPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch initial data from REST API on mount
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const [status, markets, positions] = await Promise.allSettled([
          getStatus(),
          getMarkets(),
          getPositions(),
        ]);

        if (status.status === "fulfilled") setInitialStatus(status.value);
        if (markets.status === "fulfilled") setInitialMarkets(markets.value);
        if (positions.status === "fulfilled")
          setInitialPositions(positions.value);
      } catch {
        // API may not be running yet -- WS will provide data when available
      } finally {
        setLoading(false);
      }
    };

    fetchInitial();
  }, []);

  // Prefer WebSocket data once available, fall back to REST initial data
  const markets = ws.markets.length > 0 ? ws.markets : initialMarkets;
  const positions = ws.positions.length > 0 ? ws.positions : initialPositions;
  const status = ws.status ?? initialStatus;

  return (
    <div className="flex flex-col min-h-screen">
      {/* Fixed status bar */}
      <StatusBar status={status} isConnected={ws.isConnected} />

      {/* Main content */}
      <main className="flex-1 p-4 lg:p-6 space-y-4 lg:space-y-6 pt-4">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500 text-sm font-mono animate-pulse">
              Connecting to backend...
            </div>
          </div>
        ) : (
          <>
            {/* P&L Chart -- full width */}
            <PnlChart latestPoint={ws.pnl} />

            {/* Middle grid: Inventory + Reservation Price */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
              <InventoryHeatmap markets={markets} positions={positions} />
              <ReservationPriceChart markets={markets} />
            </div>

            {/* Activity Log -- full width */}
            <ActivityLog logs={ws.logs} />
          </>
        )}
      </main>
    </div>
  );
}
