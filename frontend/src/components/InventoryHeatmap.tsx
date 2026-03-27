"use client";

import { useMemo } from "react";
import type { MarketData, Position } from "@/lib/types";

interface InventoryHeatmapProps {
  markets: MarketData[];
  positions: Position[];
  maxInventory?: number;
}

const getInventoryColor = (
  absInventory: number,
  maxInventory: number,
): { bg: string; text: string } => {
  const ratio = maxInventory > 0 ? absInventory / maxInventory : 0;

  if (ratio > 0.8) {
    return { bg: "bg-red-900/40", text: "text-red-400" };
  }
  if (ratio > 0.3) {
    return { bg: "bg-yellow-900/40", text: "text-yellow-400" };
  }
  return { bg: "bg-emerald-900/40", text: "text-emerald-400" };
};

const formatPrice = (v: number): string => `$${v.toFixed(4)}`;
const formatPnl = (v: number): string =>
  v >= 0 ? `+$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`;

const InventoryHeatmap = ({
  markets,
  positions,
  maxInventory = 100,
}: InventoryHeatmapProps) => {
  // Merge market data with position data, sorted by |inventory| descending
  const rows = useMemo(() => {
    const posMap = new Map(positions.map((p) => [p.symbol, p]));

    return [...markets]
      .map((m) => ({
        ...m,
        position: posMap.get(m.symbol),
      }))
      .sort((a, b) => Math.abs(b.inventory) - Math.abs(a.inventory));
  }, [markets, positions]);

  if (rows.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider mb-4">
          Inventory
        </h2>
        <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
          No active markets
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider mb-4">
        Inventory
      </h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase tracking-wider border-b border-gray-800">
              <th className="text-left py-2 px-3 font-medium">Symbol</th>
              <th className="text-right py-2 px-3 font-medium">Net Inv.</th>
              <th className="text-right py-2 px-3 font-medium">Bid</th>
              <th className="text-right py-2 px-3 font-medium">Ask</th>
              <th className="text-right py-2 px-3 font-medium">Spread</th>
              <th className="text-right py-2 px-3 font-medium">Unrl. P&L</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const absInv = Math.abs(row.inventory);
              const invColor = getInventoryColor(absInv, maxInventory);
              const unrealizedPnl = row.position?.unrealizedPnl ?? 0;

              return (
                <tr
                  key={row.symbol}
                  className={`border-b border-gray-800/50 cursor-pointer transition-colors hover:bg-gray-800/60 ${
                    i % 2 === 0 ? "bg-gray-900" : "bg-gray-900/60"
                  }`}
                >
                  <td className="py-2.5 px-3 font-medium text-gray-200">
                    {row.symbol}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <span
                      className={`inline-flex items-center justify-end rounded px-2 py-0.5 font-mono text-xs ${invColor.bg} ${invColor.text}`}
                    >
                      {row.inventory > 0 ? "+" : ""}
                      {row.inventory.toFixed(1)}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-gray-300">
                    {formatPrice(row.bidPrice)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-gray-300">
                    {formatPrice(row.askPrice)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-gray-400">
                    {(row.spread * 100).toFixed(2)}%
                  </td>
                  <td
                    className={`py-2.5 px-3 text-right font-mono ${
                      unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"
                    }`}
                  >
                    {formatPnl(unrealizedPnl)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default InventoryHeatmap;
