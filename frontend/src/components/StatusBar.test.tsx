import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { getQuotingModeLabel } from "./StatusBar";
import type { MarketData, ActivityLogEntry } from "@/lib/types";

// ── T030: MarketData interface fields ──
describe("T030: MarketData interface", () => {
  it("accepts optional xi, theta0, theta1, theta2, quotingMode fields", () => {
    const market: MarketData = {
      symbol: "TEST",
      midPrice: 0.5,
      reservationPrice: 0.48,
      bidPrice: 0.47,
      askPrice: 0.53,
      spread: 0.06,
      inventory: 10,
      sigmaSquared: 0.001,
      gamma: 0.1,
      timeRemaining: 3600,
      xi: 0.123,
      theta0: 1.0,
      theta1: 2.0,
      theta2: 3.0,
      quotingMode: "performative",
    };
    expect(market.xi).toBe(0.123);
    expect(market.theta0).toBe(1.0);
    expect(market.theta1).toBe(2.0);
    expect(market.theta2).toBe(3.0);
    expect(market.quotingMode).toBe("performative");
  });

  it("allows all new fields to be undefined", () => {
    const market: MarketData = {
      symbol: "TEST",
      midPrice: 0.5,
      reservationPrice: 0.48,
      bidPrice: 0.47,
      askPrice: 0.53,
      spread: 0.06,
      inventory: 10,
      sigmaSquared: 0.001,
      gamma: 0.1,
      timeRemaining: 3600,
    };
    expect(market.xi).toBeUndefined();
    expect(market.theta0).toBeUndefined();
    expect(market.theta1).toBeUndefined();
    expect(market.theta2).toBeUndefined();
    expect(market.quotingMode).toBeUndefined();
  });
});

// ── T031: ActivityLogEntry type union ──
describe("T031: ActivityLogEntry type union", () => {
  it("accepts mode_switch as a valid type", () => {
    const entry: ActivityLogEntry = {
      timestamp: "2026-01-01T00:00:00Z",
      type: "mode_switch",
      message: "Switched to performative mode",
      symbol: "TEST",
    };
    expect(entry.type).toBe("mode_switch");
  });
});

// ── T032: getQuotingModeLabel ──
describe("T032: getQuotingModeLabel", () => {
  it('maps "as" to "Avellaneda-Stoikov"', () => {
    expect(getQuotingModeLabel("as")).toBe("Avellaneda-Stoikov");
  });

  it('maps "performative" to "Performative"', () => {
    expect(getQuotingModeLabel("performative")).toBe("Performative");
  });

  it('maps "theta" to "Theta-Enhanced"', () => {
    expect(getQuotingModeLabel("theta")).toBe("Theta-Enhanced");
  });

  it('returns "--" for unknown mode', () => {
    expect(getQuotingModeLabel("unknown")).toBe("--");
  });

  it('returns "--" for undefined', () => {
    expect(getQuotingModeLabel(undefined)).toBe("--");
  });

  it('returns "--" for empty string', () => {
    expect(getQuotingModeLabel("")).toBe("--");
  });
});

// ── T033: xi/theta formatting ──
describe("T033: xi and theta value formatting", () => {
  it("formats xi to 3 decimal places", () => {
    const xi = 0.12345;
    expect(xi.toFixed(3)).toBe("0.123");
  });

  it("formats theta values to 3 decimal places", () => {
    expect((1.23456).toFixed(3)).toBe("1.235");
    expect((0.001).toFixed(3)).toBe("0.001");
    expect((99.999).toFixed(3)).toBe("99.999");
  });

  it('returns "--" when value is null/undefined using optional chaining', () => {
    const val: number | undefined = undefined;
    expect(val?.toFixed(3) ?? "--").toBe("--");
  });

  it("formats value when present using optional chaining", () => {
    const val: number | undefined = 1.5;
    expect(val?.toFixed(3) ?? "--").toBe("1.500");
  });
});
