import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

const mockOnboardingState = {
  currentStep: 0,
  totalSteps: 4,
  nextStep: vi.fn(),
  prevStep: vi.fn(),
  complete: vi.fn(),
};

vi.mock("@/stores/onboardingStore", () => ({
  useOnboardingStore: () => mockOnboardingState,
}));

import { OnboardingWizard } from "@/components/onboarding/wizard";

describe("OnboardingWizard", () => {
  beforeEach(() => {
    mockOnboardingState.currentStep = 0;
    vi.clearAllMocks();
  });

  it("renders first step title", () => {
    render(<OnboardingWizard />);
    expect(screen.getByText("Bem-vindo ao TradeMaster")).toBeInTheDocument();
  });

  it("renders step subtitle", () => {
    render(<OnboardingWizard />);
    expect(screen.getByText("Trading Cripto com Inteligência Artificial")).toBeInTheDocument();
  });

  it("shows Próximo button on first step", () => {
    render(<OnboardingWizard />);
    expect(screen.getByText("Próximo")).toBeInTheDocument();
  });

  it("does not show Voltar button on first step", () => {
    render(<OnboardingWizard />);
    expect(screen.queryByText("Voltar")).not.toBeInTheDocument();
  });

  it("calls nextStep on Próximo click", () => {
    render(<OnboardingWizard />);
    fireEvent.click(screen.getByText("Próximo"));
    expect(mockOnboardingState.nextStep).toHaveBeenCalled();
  });

  it("shows Voltar button on non-first step", () => {
    mockOnboardingState.currentStep = 1;
    render(<OnboardingWizard />);
    expect(screen.getByText("Voltar")).toBeInTheDocument();
  });

  it("calls prevStep on Voltar click", () => {
    mockOnboardingState.currentStep = 2;
    render(<OnboardingWizard />);
    fireEvent.click(screen.getByText("Voltar"));
    expect(mockOnboardingState.prevStep).toHaveBeenCalled();
  });

  it("shows Começar on last step", () => {
    mockOnboardingState.currentStep = 3;
    render(<OnboardingWizard />);
    expect(screen.getByText("Começar")).toBeInTheDocument();
    expect(screen.queryByText("Próximo")).not.toBeInTheDocument();
  });

  it("calls complete on Começar click", () => {
    mockOnboardingState.currentStep = 3;
    render(<OnboardingWizard />);
    fireEvent.click(screen.getByText("Começar"));
    expect(mockOnboardingState.complete).toHaveBeenCalled();
  });

  it("has skip button (Pular introdução)", () => {
    render(<OnboardingWizard />);
    const skip = screen.getByLabelText("Pular introdução");
    expect(skip).toBeInTheDocument();
    fireEvent.click(skip);
    expect(mockOnboardingState.complete).toHaveBeenCalled();
  });

  it("renders progress dots", () => {
    const { container } = render(<OnboardingWizard />);
    const dots = container.querySelectorAll(".rounded-full.h-2.w-2");
    expect(dots.length).toBe(4);
  });
});
