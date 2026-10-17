# ---------------------------------------------------------------------------
# Development stage — Vite dev server with hot reload.
# docker compose mounts ./frontend/src over /app/src.
# ---------------------------------------------------------------------------
FROM node:20-alpine AS development

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]

# ---------------------------------------------------------------------------
# Build stage — compiles the static bundle.
#
# VITE_API_KEY is baked into the bundle at build time (public demo key, see
# the note in src/api.ts); pass it with `--build-arg VITE_API_KEY=...`.
# ---------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_API_KEY=change-me
ENV VITE_API_KEY=$VITE_API_KEY

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------------------------------------------------------------------------
# Production stage — static files served by Nginx with SPA fallback.
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS production

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -q -O /dev/null http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
