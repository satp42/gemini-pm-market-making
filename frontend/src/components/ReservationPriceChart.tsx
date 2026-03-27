"use client";

import { useMemo, useState } from "react";
import {
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ComposedChart,
} from "recharts";
import type { MarketData } from "@/lib/types";

interface ReservationPriceChartProps {
  markets: MarketData[];
}

interface ChartPoint {
  label: string;
  midPrice: number;
  reservationPrice: number;
  bidPrice: number;
  askPrice: number;
  inventory: number;
}

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;

  const mid = payload.find((p) => p.dataKey === "midPrice")?.value ?? 0;
  const reservation =
    payload.find((p) => p.dataKey === "reservationPrice")?.value ?? 0;
  const inv = payload.find((p) => p.dataKey === "inventory")?.value ?? 0;
  const bid = payload.find((p) => p.dataKey === "bidPrice")?.value ?? 0;
  const ask = payload.find((p) => p.dataKey === "askPrice")?.value ?? 0;
  const skew = reservation - mid;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-xs shadow-lg max-w-xs">
      <div className="grid grid-cols-2 gap-x-3 gap-y-1">
        <span className="text-gray-400">Mid:</span>
        <span className="text-blue-400 font-mono">${mid.toFixed(4)}</span>
        <span className="text-gray-400">Reservation:</span>
        <span className="text-orange-400 font-mono">
          ${reservation.toFixed(4)}
        </span>
        <span className="text-gray-400">Bid:</span>
        <span className="text-gray-300 font-mono">${bid.toFixed(4)}</span>
        <span className="text-gray-400">Ask:</span>
        <span className="text-gray-300 font-mono">${ask.toFixed(4)}</span>
      </div>
      <div className="mt-2 pt-2 border-t border-gray-700 text-gray-300">
        Reservation price is{" "}
        <span className={skew >= 0 ? "text-emerald-400" : "text-red-400"}>
          {Math.abs(skew).toFixed(4)} {skew >= 0 ? "above" : "below"}
        </span>{" "}
        mid because inventory is{" "}
        <span className="text-yellow-300">
          {inv > 0 ? "+" : ""}
          {inv.toFixed(1)}
        </span>
      </div>
    </div>
  );
};

const ReservationPriceChart = ({ markets }: ReservationPriceChartProps) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");

  // Auto-select first market if none selected
  const activeSymbol =
    selectedSymbol && markets.some((m) => m.symbol === selectedSymbol)
      ? selectedSymbol
      : markets[0]?.symbol ?? "";

  const selectedMarket = useMemo(
    () => markets.find((m) => m.symbol === activeSymbol),
    [markets, activeSymbol],
  );

  // Show all markets as data points on the chart
  const allMarketsData: ChartPoint[] = useMemo(
    () =>
      markets.map((m) => ({
        label: m.symbol,
        midPrice: m.midPrice,
        reservationPrice: m.reservationPrice,
        bidPrice: m.bidPrice,
        askPrice: m.askPrice,
        inventory: m.inventory,
      })),
    [markets],
  );

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider">
          Reservation Price
        </h2>
        <select
          value={activeSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          {markets.map((m) => (
            <option key={m.symbol} value={m.symbol}>
              {m.symbol}
            </option>
          ))}
        </select>
      </div>

      {/* Chart */}
      <div className="h-64 min-w-0 overflow-hidden">
        {markets.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            No market data
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={256}>
            <ComposedChart
              data={allMarketsData}
              margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="gradBidAsk" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6b7280" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#6b7280" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="label"
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                axisLine={{ stroke: "#374151" }}
                angle={-30}
                textAnchor="end"
                height={50}
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                axisLine={{ stroke: "#374151" }}
                tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                width={60}
                domain={["auto", "auto"]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11, color: "#9ca3af" }}
                iconType="line"
              />
              {/* Bid/Ask band */}
              <Area
                type="monotone"
                dataKey="askPrice"
                name="Ask"
                stroke="#4b5563"
                fill="url(#gradBidAsk)"
                strokeWidth={1}
                strokeDasharray="4 2"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="bidPrice"
                name="Bid"
                stroke="#4b5563"
                fill="url(#gradBidAsk)"
                strokeWidth={1}
                strokeDasharray="4 2"
                dot={false}
              />
              {/* Mid price */}
              <Line
                type="monotone"
                dataKey="midPrice"
                name="Mid Price"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 3, fill: "#3b82f6" }}
                activeDot={{ r: 5 }}
              />
              {/* Reservation price */}
              <Line
                type="monotone"
                dataKey="reservationPrice"
                name="Reservation Price"
                stroke="#f97316"
                strokeWidth={2}
                dot={{ r: 3, fill: "#f97316" }}
                activeDot={{ r: 5 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Selected market detail */}
      {selectedMarket && (
        <>
          <div className="mt-3 grid grid-cols-4 gap-3 text-xs">
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Mid</span>
              <span className="text-blue-400 font-mono">
                ${selectedMarket.midPrice.toFixed(4)}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Reservation</span>
              <span className="text-orange-400 font-mono">
                ${selectedMarket.reservationPrice.toFixed(4)}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Gamma</span>
              <span className="text-gray-200 font-mono">
                {selectedMarket.gamma.toFixed(4)}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Sigma^2</span>
              <span className="text-gray-200 font-mono">
                {selectedMarket.sigmaSquared.toFixed(6)}
              </span>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-4 gap-3 text-xs">
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Xi</span>
              <span className="text-purple-400 font-mono">
                {selectedMarket.xi?.toFixed(3) ?? "--"}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Theta0</span>
              <span className="text-gray-200 font-mono">
                {selectedMarket.theta0?.toFixed(3) ?? "--"}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Theta1</span>
              <span className="text-gray-200 font-mono">
                {selectedMarket.theta1?.toFixed(3) ?? "--"}
              </span>
            </div>
            <div className="bg-gray-800/50 rounded px-2 py-1.5">
              <span className="text-gray-500 block">Theta2</span>
              <span className="text-gray-200 font-mono">
                {selectedMarket.theta2?.toFixed(3) ?? "--"}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ReservationPriceChart;
