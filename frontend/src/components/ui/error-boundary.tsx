"use client";

import { Component, type ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "./button";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  showDetails: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, showDetails: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    if (this.props.fallback) return this.props.fallback;

    return (
      <div className="flex flex-col items-center justify-center py-12 px-4 text-center animate-fade-in">
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-[var(--color-danger-light)] text-[var(--color-danger)]">
          <AlertCircle className="h-7 w-7" />
        </div>
        <h3 className="text-lg font-semibold">Algo deu errado</h3>
        <p className="mt-1 text-sm text-[var(--color-text-muted)] max-w-md">
          Um erro inesperado ocorreu. Tente atualizar a página.
        </p>
        <div className="mt-4 flex items-center gap-3">
          <Button
            variant="primary"
            size="sm"
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
          >
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Atualizar
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => this.setState((s) => ({ showDetails: !s.showDetails }))}
          >
            {this.state.showDetails ? "Ocultar Detalhes" : "Ver Detalhes"}
          </Button>
        </div>
        {this.state.showDetails && this.state.error && (
          <pre className="mt-4 max-w-lg overflow-auto rounded-[var(--radius-md)] bg-[var(--color-background)] p-4 text-left text-xs text-[var(--color-danger)]">
            {this.state.error.message}
            {"\n\n"}
            {this.state.error.stack}
          </pre>
        )}
      </div>
    );
  }
}
