FROM node:22-alpine AS base

# Install dependencies
FROM base AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

# Build
FROM base AS builder
WORKDIR /app

# Next.js bakes NEXT_PUBLIC_ vars at build time
ARG NEXT_PUBLIC_API_URL=https://backendtrademaster.up.railway.app
ARG NEXT_PUBLIC_WS_URL=wss://backendtrademaster.up.railway.app
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .
RUN npm run build

# Production
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
