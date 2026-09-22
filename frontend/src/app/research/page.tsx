"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, AlertTriangle, Ban, Database, RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/ui/page-header";
import { Spinner } from "@/components/ui/progress";
import { apiFetch } from "@/lib/utils";

type ExperimentStatus = "DRAFT" | "FROZEN" | "REJECTED" | "INCONCLUSIVE" | "APPROVED";

interface Experiment {
  id: string;
  name: string;
  status: ExperimentStatus;
  experiment_sha256: string | null;
  dataset_partitions: Array<{ role: string; start_at: string; end_at: string }>;
  safety: {
    research_only: true;
    order_submission_allowed: false;
    execution_authorization: "none";
  };
  created_at: string;
}

interface EvidenceGateStatus {
  artifact_available: boolean;
  audited_start_date: string | null;
  audited_end_date: string | null;
  audited_days: number;
  latest_daily_status: "VALID" | "PARTIAL" | "MISSING" | "INVALID" | null;
  latest_daily_manifest_sha256: string | null;
  book_evidence_gate: {
    eligible: boolean;
    required_complete_days: 60;
    required_streams: string[];
    complete_days: number;
    longest_complete_streak_days: number;
    streak_start: string | null;
    streak_end: string | null;
    reasons: string[];
  };
  status_reasons: string[];
  safety: {
    research_only: true;
    order_submission_allowed: false;
    execution_authorization: "none";
  };
  generated_at: string;
}

interface TestnetEligibility {
  experiment_id: string;
  experiment_status: ExperimentStatus;
  eligible: boolean;
  reasons: string[];
  book_evidence_eligible: boolean;
  book_evidence_contiguous_days: number;
  prospective_shadow_days: number;
  prospective_shadow_outcome_days: number;
  prospective_shadow_signal_count: number;
  prospective_shadow_outcome_signal_count: number;
  prospective_shadow_expected_mean_bps: number | null;
  prospective_shadow_stress_mean_bps: number | null;
  prospective_shadow_positive: boolean;
  shadow_ledger_verified: boolean;
  shadow_ledger_reasons: string[];
  approved_statistical_gate_verified: boolean;
  unresolved_failures: number;
  explicit_testnet_release: boolean;
  release_request_required: boolean;
  evidence_artifact_available: boolean;
  order_submission_allowed: false;
  execution_authorization: "none";
  generated_at: string;
}

interface ExperimentReport {
  experiment_id: string;
  status: Exclude<ExperimentStatus, "DRAFT">;
  decision_reasons: string[];
  metrics: {
    book_evidence?: {
      eligible?: boolean;
      longest_complete_streak_days?: number;
      complete_days?: number;
      status_reasons?: string[];
      gate_reasons?: string[];
    };
    shadow?: {
      signal_count?: number;
      outcome_signal_count?: number;
      decision_days?: number;
      outcome_days?: number;
      expected_mean_bps?: number | null;
      stress_mean_bps?: number | null;
      positive?: boolean;
      complete?: boolean;
    };
    shadow_ledger?: {
      signal_count?: number;
      outcome_count?: number;
      signal_event_count?: number;
      outcome_event_count?: number;
      verified?: boolean;
      reasons?: string[];
    };
    hypothesis_ledger?: {
      attempt_count?: number;
      attempts?: Array<{
        kind: string;
        status: string;
        fingerprint_sha256: string;
      }>;
    };
    testnet_boundary?: {
      approved_statistical_gate_verified?: boolean;
      order_submission_allowed: false;
      execution_authorization: "none";
    };
    experiment_event_chain?: {
      event_count?: number;
      verified?: boolean;
      latest_event_sha256?: string | null;
      reasons?: string[];
    };
  };
  artifact_sha256: string | null;
  generated_at: string;
}

const statusVariant: Record<ExperimentStatus, "default" | "primary" | "danger" | "warning" | "success"> = {
  DRAFT: "default",
  FROZEN: "primary",
  REJECTED: "danger",
  INCONCLUSIVE: "warning",
  APPROVED: "success",
};

export default function ResearchPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [evidenceStatus, setEvidenceStatus] = useState<EvidenceGateStatus | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [testnetEligibility, setTestnetEligibility] = useState<TestnetEligibility | null>(null);
  const [testnetError, setTestnetError] = useState<string | null>(null);
  const [experimentReport, setExperimentReport] = useState<ExperimentReport | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setEvidenceError(null);
    setTestnetError(null);
    setReportError(null);
    try {
      const [experimentsResult, evidenceResult] = await Promise.allSettled([
        apiFetch<Experiment[]>("/api/v1/research/microstructure/experiments"),
        apiFetch<EvidenceGateStatus>("/api/v1/research/microstructure/evidence-gate"),
      ]);
      if (experimentsResult.status === "rejected") {
        throw experimentsResult.reason;
      }
      const loadedExperiments = experimentsResult.value;
      setExperiments(loadedExperiments);
      if (evidenceResult.status === "fulfilled") {
        setEvidenceStatus(evidenceResult.value);
      } else {
        setEvidenceStatus(null);
        setEvidenceError(
          evidenceResult.reason instanceof Error
            ? evidenceResult.reason.message
            : "Falha ao carregar gate de evidência",
        );
      }
      if (loadedExperiments.length > 0) {
        const firstExperimentId = loadedExperiments[0].id;
        const [eligibilityResult, reportResult] = await Promise.allSettled([
          apiFetch<TestnetEligibility>(
            `/api/v1/research/microstructure/experiments/${firstExperimentId}/testnet-eligibility`,
          ),
          apiFetch<ExperimentReport>(
            `/api/v1/research/microstructure/experiments/${firstExperimentId}/report`,
          ),
        ]);
        if (eligibilityResult.status === "fulfilled") {
          setTestnetEligibility(eligibilityResult.value);
        } else {
          setTestnetEligibility(null);
          setTestnetError(
            eligibilityResult.reason instanceof Error
              ? eligibilityResult.reason.message
              : "Falha ao carregar checklist Testnet",
          );
        }
        if (reportResult.status === "fulfilled") {
          setExperimentReport(reportResult.value);
        } else {
          setExperimentReport(null);
          setReportError(
            reportResult.reason instanceof Error
              ? reportResult.reason.message
              : "Falha ao carregar relatório do experimento",
          );
        }
      } else {
        setTestnetEligibility(null);
        setExperimentReport(null);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Falha ao carregar evidências");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const approved = experiments.filter((experiment) => experiment.status === "APPROVED").length;
  const rejected = experiments.filter((experiment) => experiment.status === "REJECTED").length;
  const longestBookStreak = evidenceStatus?.book_evidence_gate.longest_complete_streak_days ?? 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Pesquisa de Microestrutura"
        description="Evidência auditável de sinais BTCUSDT. Este painel não possui controles de execução."
        actions={
          <Button variant="ghost" size="sm" onClick={() => void load()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Atualizar
          </Button>
        }
      />

      <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
          <div>
            <p className="font-medium text-[var(--color-text)]">Barreira de segurança ativa</p>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              Somente pesquisa e shadow. Envio de ordens: bloqueado. Autorização de execução: nenhuma.
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card><CardContent><Metric icon={Activity} label="Experimentos" value={experiments.length} /></CardContent></Card>
        <Card><CardContent><Metric icon={ShieldCheck} label="Aprovados" value={approved} tone="text-emerald-400" /></CardContent></Card>
        <Card><CardContent><Metric icon={Ban} label="Rejeitados" value={rejected} tone="text-red-400" /></CardContent></Card>
      </div>

      <EvidenceGatePanel status={evidenceStatus} error={evidenceError} longestBookStreak={longestBookStreak} />
      <TestnetEligibilityPanel status={testnetEligibility} error={testnetError} hasExperiments={experiments.length > 0} />
      <ExperimentReportPanel report={experimentReport} error={reportError} hasExperiments={experiments.length > 0} />

      {loading ? (
        <div className="flex justify-center py-16"><Spinner size="lg" /></div>
      ) : error ? (
        <Card className="border-red-500/30"><CardContent><p className="text-sm text-red-400">{error}</p></CardContent></Card>
      ) : experiments.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center">
            <Database className="mx-auto h-8 w-8 text-[var(--color-text-faint)]" />
            <p className="mt-3 font-medium text-[var(--color-text)]">Nenhum experimento registrado</p>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              Resultados locais de desenvolvimento não aparecem como candidatos até entrarem no registro imutável.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {experiments.map((experiment) => (
            <Card key={experiment.id}>
              <CardContent>
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="font-semibold text-[var(--color-text)]">{experiment.name}</h2>
                      <Badge variant={statusVariant[experiment.status]}>{experiment.status}</Badge>
                    </div>
                    <p className="mt-1 font-mono text-xs text-[var(--color-text-faint)]">
                      {experiment.experiment_sha256 ?? "Rascunho ainda sem hash imutável"}
                    </p>
                  </div>
                  <div className="flex gap-6 text-sm">
                    <div><p className="text-[var(--color-text-faint)]">Partições</p><p className="font-semibold text-[var(--color-text)]">{experiment.dataset_partitions.length}</p></div>
                    <div><p className="text-[var(--color-text-faint)]">Criado</p><p className="font-semibold text-[var(--color-text)]">{new Date(experiment.created_at).toLocaleDateString("pt-BR")}</p></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function ExperimentReportPanel({
  report,
  error,
  hasExperiments,
}: {
  report: ExperimentReport | null;
  error: string | null;
  hasExperiments: boolean;
}) {
  const shadow = report?.metrics.shadow;
  const book = report?.metrics.book_evidence;
  const testnetBoundary = report?.metrics.testnet_boundary;
  const eventChain = report?.metrics.experiment_event_chain;
  const shadowLedger = report?.metrics.shadow_ledger;
  const reportReady = (
    book?.eligible === true
    && shadow?.positive === true
    && shadowLedger?.verified === true
    && testnetBoundary?.approved_statistical_gate_verified === true
    && eventChain?.verified === true
  );
  const hypothesisLedger = report?.metrics.hypothesis_ledger;
  const visibleAttempts = hypothesisLedger?.attempts?.slice(0, 3) ?? [];
  const visibleReasons = collectReasons(
    error,
    hasExperiments ? null : "nenhum_experimento_registrado",
    report?.decision_reasons,
    book?.status_reasons,
    book?.gate_reasons,
    shadowLedger?.reasons,
    eventChain?.reasons,
  );

  return (
    <Card className={reportReady ? "border-emerald-500/30" : "border-slate-700/50"}>
      <CardContent>
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex gap-3">
            <Activity className="mt-0.5 h-5 w-5 shrink-0 text-blue-400" />
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-semibold text-[var(--color-text)]">Relatório do experimento</h2>
                <Badge variant={reportReady ? "success" : "default"}>
                  {reportReady ? "Auditável" : "Incompleto"}
                </Badge>
              </div>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Métricas consolidadas do primeiro experimento listado. Report hash:{" "}
                <span className="font-mono text-xs">{shortHash(report?.artifact_sha256)}</span>
              </p>
              <GateReasonList title="Razões do relatório" reasons={visibleReasons} tone="warning" />
            </div>
          </div>

          <div className="grid min-w-72 grid-cols-2 gap-3 text-sm md:grid-cols-5">
            <GateMetric label="Book streak" value={`${book?.longest_complete_streak_days ?? 0}/60`} />
            <GateMetric label="Book gate" value={book?.eligible ? "ok" : "travado"} />
            <GateMetric label="Stat gate" value={testnetBoundary?.approved_statistical_gate_verified ? "ok" : "sem hash"} />
            <GateMetric label="Ledger imutável" value={eventChain?.verified ? "ok" : "quebrado"} />
            <GateMetric label="Eventos ledger" value={eventChain?.event_count ?? 0} />
            <GateMetric label="Último evento" value={shortHash(eventChain?.latest_event_sha256)} />
            <GateMetric label="Ledger shadow" value={shadowLedger?.verified ? "ok" : "quebrado"} />
            <GateMetric label="Eventos shadow" value={`${shadowLedger?.outcome_event_count ?? 0}/${shadowLedger?.signal_event_count ?? 0}`} />
            <GateMetric label="Shadow" value={`${shadow?.outcome_signal_count ?? 0}/${shadow?.signal_count ?? 0}`} />
            <GateMetric label="Expected" value={formatBps(shadow?.expected_mean_bps)} />
            <GateMetric label="Stress" value={formatBps(shadow?.stress_mean_bps)} />
            <GateMetric label="Hipóteses" value={hypothesisLedger?.attempt_count ?? 0} />
          </div>
        </div>
        {visibleAttempts.length > 0 ? (
          <div className="mt-4 grid gap-2 text-xs text-[var(--color-text-muted)] md:grid-cols-3">
            {visibleAttempts.map((attempt) => (
              <div key={attempt.fingerprint_sha256} className="rounded-lg border border-slate-800/80 p-2">
                <div className="font-medium text-[var(--color-text)]">{attempt.kind}</div>
                <div className="mt-1">{attempt.status}</div>
                <div className="mt-1 font-mono">{shortHash(attempt.fingerprint_sha256)}</div>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function TestnetEligibilityPanel({
  status,
  error,
  hasExperiments,
}: {
  status: TestnetEligibility | null;
  error: string | null;
  hasExperiments: boolean;
}) {
  const checklistReady = status?.eligible === true;
  const bookGateReady = status?.book_evidence_eligible === true;
  const statGateReady = status?.approved_statistical_gate_verified === true;
  const shadowLedgerReady = status?.shadow_ledger_verified === true;
  const visibleReasons = collectReasons(
    error,
    status?.reasons,
    status?.shadow_ledger_reasons,
    hasExperiments ? null : "nenhum_experimento_registrado",
  );
  const badgeLabel = checklistReady
    ? "Release registrado"
    : status?.explicit_testnet_release
      ? "Release incompleto"
      : "Sem release";

  return (
    <Card className={checklistReady ? "border-blue-500/30" : "border-red-500/30"}>
      <CardContent>
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex gap-3">
            <Ban className={`mt-0.5 h-5 w-5 shrink-0 ${checklistReady ? "text-blue-400" : "text-red-400"}`} />
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-semibold text-[var(--color-text)]">Checklist Testnet</h2>
                <Badge variant={checklistReady ? "primary" : "danger"}>
                  {badgeLabel}
                </Badge>
              </div>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Cruza experimento aprovado, gate book/spot WAL elegível, 60 dias de book e 20–30 dias de shadow com outcomes positivos.
                Mesmo com release registrado, este painel continua sem autorização de execução e sem botão de ordem.
              </p>
              <GateReasonList title="Razões do checklist Testnet" reasons={visibleReasons} tone="danger" />
            </div>
          </div>

          <div className="grid min-w-72 grid-cols-2 gap-3 text-sm md:grid-cols-5">
            <GateMetric label="Book gate" value={bookGateReady ? "ok" : "travado"} />
            <GateMetric label="Stat gate" value={statGateReady ? "ok" : "sem hash"} />
            <GateMetric label="Shadow ledger" value={shadowLedgerReady ? "ok" : "quebrado"} />
            <GateMetric label="Book" value={`${status?.book_evidence_contiguous_days ?? 0}/60`} />
            <GateMetric label="Shadow" value={`${status?.prospective_shadow_days ?? 0}/20`} />
            <GateMetric
              label="Outcomes"
              value={`${status?.prospective_shadow_outcome_signal_count ?? 0}/${status?.prospective_shadow_signal_count ?? 0}`}
            />
            <GateMetric label="Falhas" value={status?.unresolved_failures ?? 0} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function EvidenceGatePanel({
  status,
  error,
  longestBookStreak,
}: {
  status: EvidenceGateStatus | null;
  error: string | null;
  longestBookStreak: number;
}) {
  const gate = status?.book_evidence_gate;
  const gateReady = gate?.eligible === true;
  const requiredStreams = formatRequiredStreams(gate?.required_streams);
  const badgeVariant = gateReady ? "success" : "warning";
  const visibleReasons = collectReasons(error, status?.status_reasons, gate?.reasons);

  return (
    <Card className={gateReady ? "border-emerald-500/30" : "border-amber-500/30"}>
      <CardContent>
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex gap-3">
            {gateReady ? (
              <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
            ) : (
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
            )}
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-semibold text-[var(--color-text)]">Gate de book/WAL prospectivo</h2>
                <Badge variant={badgeVariant}>{gateReady ? "Elegível" : "Travado"}</Badge>
              </div>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Exige 60 dias UTC completos e consecutivos de {requiredStreams} antes de qualquer canário Testnet.
                O painel só lê artefato offline; não escaneia dados brutos nem possui botão de execução.
              </p>
              <GateReasonList title="Razões do gate book/WAL" reasons={visibleReasons} tone="warning" />
            </div>
          </div>

          <div className="grid min-w-72 grid-cols-3 gap-3 text-sm">
            <GateMetric label="Streak" value={`${longestBookStreak}/60`} />
            <GateMetric label="Dias auditados" value={status?.audited_days ?? 0} />
            <GateMetric label="Último dia" value={status?.latest_daily_status ?? "n/a"} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function GateReasonList({
  title,
  reasons,
  tone,
}: {
  title: string;
  reasons: string[];
  tone: "danger" | "warning";
}) {
  if (reasons.length === 0) return null;
  const textColor = tone === "danger" ? "text-red-300" : "text-amber-300";
  return (
    <div className={`mt-2 font-mono text-xs ${textColor}`}>
      <p className="sr-only">{title}</p>
      <ul aria-label={title} className="space-y-1">
        {reasons.map((reason) => (
          <li key={reason}>• {reason}</li>
        ))}
      </ul>
    </div>
  );
}

function GateMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-muted)] p-3">
      <p className="text-xs text-[var(--color-text-faint)]">{label}</p>
      <p className="mt-1 font-semibold tabular-nums text-[var(--color-text)]">{value}</p>
    </div>
  );
}

function Metric({ icon: Icon, label, value, tone = "text-[var(--color-text)]" }: { icon: typeof Activity; label: string; value: number | string; tone?: string }) {
  return (
    <div className="flex items-center justify-between">
      <div><p className="text-sm text-[var(--color-text-muted)]">{label}</p><p className={`mt-1 text-2xl font-bold tabular-nums ${tone}`}>{value}</p></div>
      <Icon className="h-5 w-5 text-[var(--color-text-faint)]" />
    </div>
  );
}

function shortHash(value: string | null | undefined): string {
  return value ? `${value.slice(0, 10)}…` : "n/a";
}

function formatBps(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} bps`;
}

function formatRequiredStreams(streams: string[] | undefined): string {
  const requiredStreams = streams?.length
    ? streams
    : ["TRADE", "DEPTH", "MARK_PRICE", "SPOT_TRADE"];
  return requiredStreams.map((stream) => stream.toLowerCase()).join(", ");
}

function collectReasons(
  ...sources: Array<string | string[] | null | undefined>
): string[] {
  const seen = new Set<string>();
  const reasons: string[] = [];
  for (const source of sources) {
    const values = Array.isArray(source) ? source : [source];
    for (const value of values) {
      const reason = value?.trim();
      if (reason && !seen.has(reason)) {
        seen.add(reason);
        reasons.push(reason);
      }
    }
  }
  return reasons;
}
