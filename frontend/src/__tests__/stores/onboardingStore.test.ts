import { describe, it, expect, beforeEach } from "vitest";

describe("onboardingStore", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("initializes with step 0 and not completed", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    const state = useOnboardingStore.getState();
    expect(state.currentStep).toBe(0);
    expect(state.totalSteps).toBe(4);
  });

  it("nextStep increments currentStep", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    useOnboardingStore.getState().nextStep();
    expect(useOnboardingStore.getState().currentStep).toBe(1);
  });

  it("nextStep does not exceed totalSteps - 1", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    // Go to last step
    for (let i = 0; i < 10; i++) {
      useOnboardingStore.getState().nextStep();
    }
    expect(useOnboardingStore.getState().currentStep).toBe(3);
  });

  it("prevStep decrements currentStep", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    useOnboardingStore.getState().setStep(2);
    useOnboardingStore.getState().prevStep();
    expect(useOnboardingStore.getState().currentStep).toBe(1);
  });

  it("prevStep does not go below 0", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    useOnboardingStore.getState().setStep(0);
    useOnboardingStore.getState().prevStep();
    expect(useOnboardingStore.getState().currentStep).toBe(0);
  });

  it("complete marks onboarding as completed", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    useOnboardingStore.getState().complete();
    expect(useOnboardingStore.getState().completed).toBe(true);
    expect(localStorage.getItem("trademaster_onboarding_completed")).toBe("true");
  });

  it("reset clears completion state", async () => {
    const { useOnboardingStore } = await import("@/stores/onboardingStore");
    useOnboardingStore.getState().complete();
    useOnboardingStore.getState().reset();
    expect(useOnboardingStore.getState().completed).toBe(false);
    expect(useOnboardingStore.getState().currentStep).toBe(0);
  });
});
