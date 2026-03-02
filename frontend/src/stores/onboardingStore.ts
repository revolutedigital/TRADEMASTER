import { create } from "zustand";

interface OnboardingState {
  currentStep: number;
  completed: boolean;
  totalSteps: number;
  setStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  complete: () => void;
  reset: () => void;
}

const STORAGE_KEY = "trademaster_onboarding_completed";

export const useOnboardingStore = create<OnboardingState>((set, get) => ({
  currentStep: 0,
  completed: typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) === "true" : false,
  totalSteps: 4,

  setStep: (step) => set({ currentStep: step }),

  nextStep: () => {
    const { currentStep, totalSteps } = get();
    if (currentStep < totalSteps - 1) {
      set({ currentStep: currentStep + 1 });
    }
  },

  prevStep: () => {
    const { currentStep } = get();
    if (currentStep > 0) {
      set({ currentStep: currentStep - 1 });
    }
  },

  complete: () => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, "true");
    }
    set({ completed: true });
  },

  reset: () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(STORAGE_KEY);
    }
    set({ completed: false, currentStep: 0 });
  },
}));
