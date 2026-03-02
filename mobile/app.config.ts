export default {
  name: "TradeMaster",
  slug: "trademaster-mobile",
  version: "1.0.0",
  orientation: "portrait",
  icon: "./assets/icon.png",
  splash: {
    image: "./assets/splash.png",
    resizeMode: "contain",
    backgroundColor: "#0a0e17",
  },
  ios: {
    supportsTablet: true,
    bundleIdentifier: "com.trademaster.mobile",
    infoPlist: {
      NSFaceIDUsageDescription: "Authenticate with Face ID to access your trading account",
    },
  },
  android: {
    package: "com.trademaster.mobile",
    adaptiveIcon: {
      foregroundImage: "./assets/adaptive-icon.png",
      backgroundColor: "#0a0e17",
    },
  },
  extra: {
    apiBaseUrl: process.env.API_BASE_URL || "https://backendtrademaster.up.railway.app",
    wsUrl: process.env.WS_URL || "wss://backendtrademaster.up.railway.app/ws",
  },
};
