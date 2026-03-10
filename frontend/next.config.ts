import type { NextConfig } from "next";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_API_URL: API_URL,
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000",
  },
  experimental: {
    staleTimes: {
      dynamic: 0,
      static: 0,
    },
  },
  // Proxy API requests to backend so pages using relative fetch("/api/v1/...")
  // work correctly. Also avoids cross-origin cookie issues since requests
  // become same-origin from the browser's perspective.
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${API_URL}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
