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
  book_evidence_contiguous_days: number;
  prospective_shadow_days: number;
  prospective_shadow_outcome_days: number;
  prospective_shadow_signal_count: number;
  prospective_shadow_expected_mean_bps: number | null;
  prospective_shadow_stress_mean_bps: number | null;
  prospective_shadow_positive: boolean;
  unresolved_failures: number;
  explicit_testnet_release: false;
  release_request_required: true;
  evidence_artifact_available: boolean;
  order_submission_allowed: false;
  execution_authorization: "none";
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setEvidenceError(null);
    setTestnetError(null);
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
        try {
          setTestnetEligibility(
            await apiFetch<TestnetEligibility>(
              `/api/v1/research/microstructure/experiments/${loadedExperiments[0].id}/testnet-eligibility`,
            ),
          );
        } catch (requestError) {
          setTestnetEligibility(null);
          setTestnetError(
            requestError instanceof Error
              ? requestError.message
              : "Falha ao carregar checklist Testnet",
          );
        }
      } else {
        setTestnetEligibility(null);
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
  const visibleReason = error ?? status?.reasons[0] ?? (hasExperiments ? null : "nenhum_experimento_registrado");

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
                  {checklistReady ? "Pronto para release manual" : "Sem release"}
                </Badge>
              </div>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Cruza experimento, 60 dias de book e 20–30 dias de shadow com outcomes positivos.
                Mesmo quando ficar pronto, este painel continua sem autorização de execução e exige release explícito separado.
              </p>
              {visibleReason ? (
                <p className="mt-2 font-mono text-xs text-red-300">{visibleReason}</p>
              ) : null}
            </div>
          </div>

          <div className="grid min-w-72 grid-cols-2 gap-3 text-sm md:grid-cols-4">
            <GateMetric label="Book" value={`${status?.book_evidence_contiguous_days ?? 0}/60`} />
            <GateMetric label="Shadow" value={`${status?.prospective_shadow_days ?? 0}/20`} />
            <GateMetric
              label="Outcome"
              value={`${status?.prospective_shadow_outcome_days ?? 0}/${status?.prospective_shadow_days ?? 20}`}
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
  const badgeVariant = gateReady ? "success" : "warning";
  const visibleReason = error ?? status?.status_reasons[0] ?? gate?.reasons[0] ?? null;

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
                Exige 60 dias UTC completos e consecutivos de book antes de qualquer canário Testnet.
                O painel só lê artefato offline; não escaneia dados brutos nem possui botão de execução.
              </p>
              {visibleReason ? (
                <p className="mt-2 font-mono text-xs text-amber-300">{visibleReason}</p>
              ) : null}
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
