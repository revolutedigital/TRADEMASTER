"use client";

import { useOnboardingStore } from "@/stores/onboardingStore";
import {
  Activity,
  Key,
  Shield,
  FlaskConical,
  ChevronRight,
  ChevronLeft,
  X,
} from "lucide-react";

const steps = [
  {
    icon: Activity,
    title: "Bem-vindo ao TradeMaster",
    subtitle: "Trading Cripto com Inteligência Artificial",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          TradeMaster é uma plataforma de trading de criptomoedas com inteligência artificial que combina
          modelos de machine learning com dados de mercado em tempo real para gerar sinais de trading
          para BTC e ETH.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Dados em Tempo Real</div>
            <p className="mt-1 text-xs">Preços ao vivo da Binance com gráficos de candlestick e indicadores técnicos</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Sinais de ML</div>
            <p className="mt-1 text-xs">Modelos ensemble LSTM + XGBoost geram sinais de COMPRA/MANUTENÇÃO/VENDA</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Paper Trading</div>
            <p className="mt-1 text-xs">Pratique com ordens simuladas antes de operar ao vivo</p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Gestão de Risco</div>
            <p className="mt-1 text-xs">Circuit breakers, dimensionamento de posição e limites de drawdown</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: Key,
    title: "Configuração de API",
    subtitle: "Conectar à Binance Testnet",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          O TradeMaster se conecta à Binance Testnet por padrão para prática segura e sem riscos.
          Nenhum fundo real é utilizado até que você mude explicitamente para o modo ao vivo.
        </p>
        <div className="rounded-lg bg-[var(--color-background)] p-4 space-y-3">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">1</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Criar Conta Testnet</div>
              <p className="text-xs">Acesse testnet.binance.vision e crie uma conta testnet gratuita</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">2</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Gerar Chaves de API</div>
              <p className="text-xs">Gere uma chave de API e segredo no painel da testnet</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--color-primary)]/20 text-xs font-bold text-[var(--color-primary)]">3</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Configurar nas Configurações</div>
              <p className="text-xs">Vá para a página de Configurações e cole suas chaves de API com segurança</p>
            </div>
          </div>
        </div>
        <p className="text-xs text-yellow-400">
          Suas chaves de API são criptografadas em repouso e nunca expostas no frontend.
        </p>
      </div>
    ),
  },
  {
    icon: Shield,
    title: "Gestão de Risco",
    subtitle: "Entendendo Sua Rede de Proteção",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          O TradeMaster inclui múltiplas camadas de proteção de risco para proteger seu capital.
        </p>
        <div className="space-y-3">
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Circuit Breaker</div>
            <p className="mt-1 text-xs">
              Reduz ou interrompe automaticamente o trading quando limites de drawdown são ultrapassados.
              Estados: NORMAL &rarr; REDUZIDO (-50% tamanho) &rarr; PAUSADO &rarr; PARADO
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Limites de Posição</div>
            <p className="mt-1 text-xs">
              Máximo de 3 posições simultâneas com limite de 40% de exposição total do portfólio.
              Tamanho individual da posição limitado a 20% do patrimônio.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Stop Loss / Take Profit</div>
            <p className="mt-1 text-xs">
              Toda operação inclui stop-loss automático (2%) e take-profit (4%).
              Estes são monitorados a cada 5 segundos e executados automaticamente.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] p-3">
            <div className="font-medium text-[var(--color-text)]">Limites de Drawdown</div>
            <p className="mt-1 text-xs">
              Diário: -3% | Semanal: -7% | Mensal: -15%.
              Ultrapassar estes limites aciona o circuit breaker para preservação de capital.
            </p>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: FlaskConical,
    title: "Seu Primeiro Backtest",
    subtitle: "Teste Antes de Operar",
    content: (
      <div className="space-y-4 text-sm text-[var(--color-text-muted)]">
        <p>
          Antes de operar com sinais reais, execute um backtest para ver como os modelos de ML
          teriam performado historicamente.
        </p>
        <div className="rounded-lg bg-[var(--color-background)] p-4 space-y-3">
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">1</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Navegue até o Backtest</div>
              <p className="text-xs">Clique em &ldquo;Backtest&rdquo; na barra lateral para abrir a página de backtesting</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">2</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Configure os Parâmetros</div>
              <p className="text-xs">Selecione BTCUSDT, intervalo de 1h, capital inicial de $10.000 e limiar de sinal padrão</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">3</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Analise os Resultados</div>
              <p className="text-xs">Revise a curva de patrimônio, índice Sharpe, drawdown máximo, taxa de acerto e fator de lucro</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-400">4</span>
            <div>
              <div className="font-medium text-[var(--color-text)]">Inicie o Paper Trading</div>
              <p className="text-xs">Se os resultados parecerem promissores, volte ao Painel e inicie o motor de trading</p>
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs text-[var(--color-primary)]">
          Dica: Um índice Sharpe acima de 1.0 e um fator de lucro acima de 1.5 são geralmente considerados bons indicadores.
        </p>
      </div>
    ),
  },
];

export function OnboardingWizard() {
  const { currentStep, totalSteps, nextStep, prevStep, complete } = useOnboardingStore();
  const step = steps[currentStep];
  const Icon = step.icon;
  const isLast = currentStep === totalSteps - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="relative w-full max-w-lg rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl">
        {/* Close button */}
        <button
          onClick={complete}
          className="absolute right-4 top-4 rounded-md p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
          aria-label="Pular introdução"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-6 py-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--color-primary)]/10">
            <Icon className="h-5 w-5 text-[var(--color-primary)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-text)]">{step.title}</h2>
            <p className="text-sm text-[var(--color-text-muted)]">{step.subtitle}</p>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-5">{step.content}</div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[var(--color-border)] px-6 py-4">
          {/* Progress dots */}
          <div className="flex gap-1.5">
            {Array.from({ length: totalSteps }).map((_, i) => (
              <div
                key={i}
                className={`h-2 w-2 rounded-full transition-colors ${
                  i === currentStep
                    ? "bg-[var(--color-primary)]"
                    : i < currentStep
                    ? "bg-[var(--color-primary)]/40"
                    : "bg-[var(--color-border)]"
                }`}
              />
            ))}
          </div>

          {/* Navigation buttons */}
          <div className="flex gap-2">
            {currentStep > 0 && (
              <button
                onClick={prevStep}
                className="flex items-center gap-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)] transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                Voltar
              </button>
            )}
            <button
              onClick={isLast ? complete : nextStep}
              className="flex items-center gap-1 rounded-lg bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--color-primary)]/90 transition-colors"
            >
              {isLast ? "Começar" : "Próximo"}
              {!isLast && <ChevronRight className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
