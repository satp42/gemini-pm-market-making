// ── Core domain types matching backend API ──

export interface MarketData {
  symbol: string;
  eventTitle?: string;
  midPrice: number;
  reservationPrice: number;
  bidPrice: number;
  askPrice: number;
  spread: number;
  inventory: number;
  sigmaSquared: number;
  gamma: number;
  timeRemaining: number;
  xi?: number;
  theta0?: number;
  theta1?: number;
  theta2?: number;
  quotingMode?: string;
}

export interface Position {
  symbol: string;
  eventTitle?: string;
  yesQuantity: number;
  noQuantity: number;
  netInventory: number;
  unrealizedPnl: number;
}

export interface PnlDataPoint {
  timestamp: string;
  realizedPnl: number;
  unrealizedPnl: number;
  totalExposure: number;
}

export interface BotStatus {
  running: boolean;
  uptime: number;
  activeMarkets: number;
  environment: "sandbox" | "live";
}

export interface BotConfig {
  gamma: number;
  maxInventory: number;
  minSpread: number;
  updateIntervalMs: number;
}

export interface ActivityLogEntry {
  timestamp: string;
  type:
    | "order_placed"
    | "order_filled"
    | "order_cancelled"
    | "risk_alert"
    | "mode_switch"
    | "info";
  message: string;
  symbol?: string;
}

export interface OrderEntry {
  id: number;
  timestamp: string;
  symbol: string;
  geminiOrderId: number;
  side: string;
  outcome: string;
  price: number;
  quantity: number;
  status: string;
  fillPrice: number | null;
}

export interface DashboardTick {
  type: "tick";
  data: {
    markets: MarketData[];
    positions: Position[];
    pnl: PnlDataPoint;
    status: BotStatus;
  };
}

export interface WebSocketEvent {
  type: "tick" | "order_fill" | "risk_alert" | "log";
  data: unknown;
}
