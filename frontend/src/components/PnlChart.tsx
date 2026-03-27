"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { PnlDataPoint } from "@/lib/types";
import { getPnl } from "@/lib/api";

type TimeRange = "1h" | "6h" | "24h" | "7d";

interface PnlChartProps {
  /** Latest tick from WebSocket to append in real-time */
  latestPoint: PnlDataPoint | null;
}

const RANGE_OPTIONS: { label: string; value: TimeRange }[] = [
  { label: "1H", value: "1h" },
  { label: "6H", value: "6h" },
  { label: "24H", value: "24h" },
  { label: "7D", value: "7d" },
];

const formatTime = (ts: string): string => {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};

const formatCurrency = (v: number): string =>
  v >= 0 ? `$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`;

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-xs shadow-lg">
      <p className="text-gray-400 mb-1">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.color }}>
          {entry.dataKey === "realizedPnl" ? "Realized" : "Unrealized"}:{" "}
          {formatCurrency(entry.value)}
        </p>
      ))}
    </div>
  );
};

const PnlChart = ({ latestPoint }: PnlChartProps) => {
  const [range, setRange] = useState<TimeRange>("1h");
  const [data, setData] = useState<PnlDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const rangeRef = useRef(range);

  // Fetch historical data on range change
  const fetchData = useCallback(async (r: TimeRange) => {
    setLoading(true);
    try {
      const points = await getPnl(r);
      setData(points);
    } catch (err) {
      console.error("Failed to fetch P&L data:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    rangeRef.current = range;
    fetchData(range);
  }, [range, fetchData]);

  // Append real-time data from WebSocket
  useEffect(() => {
    if (!latestPoint) return;
    setData((prev) => {
      const next = [...prev, latestPoint];
      // Keep a reasonable buffer based on range
      const maxPoints =
        rangeRef.current === "1h" ? 360 :
        rangeRef.current === "6h" ? 720 :
        rangeRef.current === "24h" ? 1440 : 2016;
      return next.length > maxPoints ? next.slice(-maxPoints) : next;
    });
  }, [latestPoint]);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider">
          Profit & Loss
        </h2>
        <div className="flex gap-1">
          {RANGE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setRange(opt.value)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                range === opt.value
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-750"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="h-64 min-w-0 overflow-hidden">
        {loading && data.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            Loading...
          </div>
        ) : data.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            No data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={256}>
            <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gradRealized" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradUnrealized" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTime}
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                axisLine={{ stroke: "#374151" }}
              />
              <YAxis
                tickFormatter={(v: number) => formatCurrency(v)}
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                axisLine={{ stroke: "#374151" }}
                width={70}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11, color: "#9ca3af" }}
                iconType="line"
              />
              <Area
                type="monotone"
                dataKey="realizedPnl"
                name="Realized P&L"
                stroke="#10b981"
                fill="url(#gradRealized)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: "#10b981" }}
              />
              <Area
                type="monotone"
                dataKey="unrealizedPnl"
                name="Unrealized P&L"
                stroke="#3b82f6"
                fill="url(#gradUnrealized)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: "#3b82f6" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default PnlChart;
