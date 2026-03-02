/**
 * i18n foundation for TradeMaster.
 *
 * Simple translation system using JSON files.
 * Can be upgraded to next-intl for full SSR i18n support.
 */

import en from "./en.json";
import pt from "./pt.json";

export type Locale = "en" | "pt";

const translations: Record<Locale, typeof en> = { en, pt };

let currentLocale: Locale = "en";

export function setLocale(locale: Locale) {
  currentLocale = locale;
}

export function getLocale(): Locale {
  return currentLocale;
}

/**
 * Get a translated string by dot-notation key.
 * Example: t("nav.dashboard") -> "Dashboard"
 */
export function t(key: string): string {
  const parts = key.split(".");
  let result: unknown = translations[currentLocale];

  for (const part of parts) {
    if (result && typeof result === "object" && part in result) {
      result = (result as Record<string, unknown>)[part];
    } else {
      // Fallback to English
      result = translations.en;
      for (const p of parts) {
        if (result && typeof result === "object" && p in result) {
          result = (result as Record<string, unknown>)[p];
        } else {
          return key; // Return key if not found
        }
      }
      break;
    }
  }

  return typeof result === "string" ? result : key;
}
