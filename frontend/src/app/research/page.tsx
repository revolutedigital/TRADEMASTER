"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, Ban, Database, RefreshCw, ShieldCheck } from "lucide-react";
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

const statusVariant: Record<ExperimentStatus, "default" | "primary" | "danger" | "warning" | "success"> = {
  DRAFT: "default",
  FROZEN: "primary",
  REJECTED: "danger",
  INCONCLUSIVE: "warning",
  APPROVED: "success",
};

export default function ResearchPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setExperiments(await apiFetch<Experiment[]>("/api/v1/research/microstructure/experiments"));
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

function Metric({ icon: Icon, label, value, tone = "text-[var(--color-text)]" }: { icon: typeof Activity; label: string; value: number; tone?: string }) {
  return (
    <div className="flex items-center justify-between">
      <div><p className="text-sm text-[var(--color-text-muted)]">{label}</p><p className={`mt-1 text-2xl font-bold tabular-nums ${tone}`}>{value}</p></div>
      <Icon className="h-5 w-5 text-[var(--color-text-faint)]" />
    </div>
  );
}
