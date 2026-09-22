import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const mockApiFetch = vi.hoisted(() => vi.fn());
const mockPathname = vi.hoisted(() => "/research");

vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
}));

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return { ...actual, apiFetch: mockApiFetch };
});

import ResearchPage from "@/app/research/page";

const experiment = {
  id: "exp_microstructure_v1",
  name: "Microstructure WAL v1",
  status: "APPROVED",
  experiment_sha256: "abc123def456abc123def456",
  dataset_partitions: [
    {
      role: "TRAINING",
      start_at: "2026-01-01T00:00:00Z",
      end_at: "2026-03-01T00:00:00Z",
    },
    {
      role: "PROSPECTIVE_SHADOW",
      start_at: "2026-03-01T00:00:00Z",
      end_at: "2026-03-21T00:00:00Z",
    },
  ],
  safety: {
    research_only: true,
    order_submission_allowed: false,
    execution_authorization: "none",
  },
  created_at: "2026-03-22T12:00:00Z",
};

const evidenceGate = {
  artifact_available: true,
  audited_start_date: "2026-01-01",
  audited_end_date: "2026-03-01",
  audited_days: 60,
  latest_daily_status: "VALID",
  latest_daily_manifest_sha256: "daily123",
  book_evidence_gate: {
    eligible: false,
    required_complete_days: 60,
    required_streams: ["TRADE", "DEPTH", "MARK_PRICE", "SPOT_TRADE"],
    complete_days: 60,
    longest_complete_streak_days: 60,
    streak_start: "2026-01-01",
    streak_end: "2026-03-01",
    reasons: ["spot_trade_missing_from_evidence_gate"],
  },
  status_reasons: ["spot_trade_missing_from_evidence_gate"],
  safety: {
    research_only: true,
    order_submission_allowed: false,
    execution_authorization: "none",
  },
  generated_at: "2026-03-22T12:10:00Z",
};

const testnetEligibility = {
  experiment_id: "exp_microstructure_v1",
  experiment_status: "APPROVED",
  eligible: false,
  reasons: ["book_evidence_gate_not_eligible"],
  book_evidence_eligible: false,
  book_evidence_contiguous_days: 60,
  prospective_shadow_days: 20,
  prospective_shadow_outcome_days: 20,
  prospective_shadow_signal_count: 18,
  prospective_shadow_outcome_signal_count: 18,
  prospective_shadow_expected_mean_bps: 3.2,
  prospective_shadow_stress_mean_bps: 1.1,
  prospective_shadow_positive: true,
  approved_statistical_gate_verified: true,
  unresolved_failures: 0,
  explicit_testnet_release: false,
  release_request_required: true,
  evidence_artifact_available: true,
  order_submission_allowed: false,
  execution_authorization: "none",
  generated_at: "2026-03-22T12:11:00Z",
};

const experimentReport = {
  experiment_id: "exp_microstructure_v1",
  status: "APPROVED",
  decision_reasons: ["book_evidence_gate_not_eligible"],
  metrics: {
    book_evidence: {
      eligible: false,
      longest_complete_streak_days: 60,
      complete_days: 60,
      status_reasons: [],
      gate_reasons: ["spot_trade_missing_from_evidence_gate"],
    },
    shadow: {
      signal_count: 18,
      outcome_signal_count: 18,
      decision_days: 20,
      outcome_days: 20,
      expected_mean_bps: 3.2,
      stress_mean_bps: 1.1,
      positive: true,
      complete: true,
    },
    hypothesis_ledger: {
      attempt_count: 1,
      attempts: [
        {
          kind: "top_p_entry",
          status: "APPROVED",
          fingerprint_sha256: "fingerprint123456",
        },
      ],
    },
    testnet_boundary: {
      approved_statistical_gate_verified: true,
      order_submission_allowed: false,
      execution_authorization: "none",
    },
  },
  artifact_sha256: "report123456789",
  generated_at: "2026-03-22T12:12:00Z",
};

describe("ResearchPage", () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
    mockApiFetch.mockImplementation((path: string) => {
      if (path === "/api/v1/research/microstructure/experiments") {
        return Promise.resolve([experiment]);
      }
      if (path === "/api/v1/research/microstructure/evidence-gate") {
        return Promise.resolve(evidenceGate);
      }
      if (path === "/api/v1/research/microstructure/experiments/exp_microstructure_v1/testnet-eligibility") {
        return Promise.resolve(testnetEligibility);
      }
      if (path === "/api/v1/research/microstructure/experiments/exp_microstructure_v1/report") {
        return Promise.resolve(experimentReport);
      }
      return Promise.reject(new Error(`Unexpected path: ${path}`));
    });
  });

  it("keeps Testnet locked when the 60-day streak lacks the required spot WAL gate", async () => {
    render(<ResearchPage />);

    expect(await screen.findByText("Microstructure WAL v1")).toBeInTheDocument();
    expect(screen.getByText(/trade, depth, mark_price, spot_trade/)).toBeInTheDocument();
    expect(screen.getByText("book_evidence_gate_not_eligible")).toBeInTheDocument();
    expect(screen.getByText("spot_trade_missing_from_evidence_gate")).toBeInTheDocument();
    expect(screen.getByText("Sem release")).toBeInTheDocument();
    expect(screen.getByText("Incompleto")).toBeInTheDocument();
    expect(screen.getAllByText("Book gate")).toHaveLength(2);
    expect(screen.getAllByText("Stat gate")).toHaveLength(2);
    expect(screen.getAllByText("travado")).toHaveLength(2);
    expect(screen.getAllByText("60/60").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("Comprar")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /ordem|comprar|vender/i })).not.toBeInTheDocument();
  });
});
