"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ActivityLogEntry,
  BotStatus,
  DashboardTick,
  MarketData,
  PnlDataPoint,
  Position,
} from "@/lib/types";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/dashboard";

const MAX_LOG_ENTRIES = 100;
const MAX_RECONNECT_DELAY_MS = 30_000;
const BASE_RECONNECT_DELAY_MS = 1_000;

export interface WebSocketState {
  markets: MarketData[];
  positions: Position[];
  pnl: PnlDataPoint | null;
  status: BotStatus | null;
  logs: ActivityLogEntry[];
  isConnected: boolean;
}

export const useWebSocket = (): WebSocketState => {
  const [markets, setMarkets] = useState<MarketData[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [pnl, setPnl] = useState<PnlDataPoint | null>(null);
  const [status, setStatus] = useState<BotStatus | null>(null);
  const [logs, setLogs] = useState<ActivityLogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectRef = useRef<() => void>(() => {});

  const appendLog = useCallback((entry: ActivityLogEntry) => {
    setLogs((prev) => {
      const next = [...prev, entry];
      return next.length > MAX_LOG_ENTRIES ? next.slice(-MAX_LOG_ENTRIES) : next;
    });
  }, []);

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as {
          type: string;
          data: unknown;
        };

        switch (msg.type) {
          case "tick": {
            const tick = msg as DashboardTick;
            setMarkets(tick.data.markets);
            setPositions(tick.data.positions);
            setPnl(tick.data.pnl);
            setStatus(tick.data.status);
            break;
          }
          case "log":
          case "order_fill":
          case "risk_alert": {
            appendLog(msg.data as ActivityLogEntry);
            break;
          }
          default:
            break;
        }
      } catch {
        // Ignore malformed messages
      }
    },
    [appendLog],
  );

  // Store connect in a ref so onclose can call it without circular dependency
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      retriesRef.current = 0;
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;

      // Exponential backoff reconnect via ref
      const delay = Math.min(
        BASE_RECONNECT_DELAY_MS * 2 ** retriesRef.current,
        MAX_RECONNECT_DELAY_MS,
      );
      retriesRef.current += 1;
      reconnectTimerRef.current = setTimeout(() => {
        connectRef.current();
      }, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [handleMessage]);

  // Keep the ref in sync and manage connection lifecycle
  useEffect(() => {
    connectRef.current = connect;
    connect();

    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { markets, positions, pnl, status, logs, isConnected };
};
