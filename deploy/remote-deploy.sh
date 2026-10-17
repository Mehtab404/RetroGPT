#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Apply a pushed image tag on the deployment host (Issue #26).
#
# Run on the EC2 host from the repository checkout that holds
# .env.production and ./data (raw game documents):
#
#   IMAGE_TAG=<tag> ./deploy/remote-deploy.sh
#
# Pulls the images, reconciles the stack and smoke-tests /api/health.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:?set IMAGE_TAG (the tag CI pushed, e.g. short git SHA)}"
export IMAGE_TAG

COMPOSE=(docker compose --env-file .env.production -f docker-compose.prod.yml)

echo "==> Pulling images ($IMAGE_TAG)"
"${COMPOSE[@]}" pull

echo "==> Applying stack"
"${COMPOSE[@]}" up -d --remove-orphans

echo "==> Smoke test: /api/health"
for _ in $(seq 1 30); do
  if curl -fsS http://localhost/api/health >/dev/null 2>&1; then
    echo "==> Healthy. Deployed $IMAGE_TAG"
    exit 0
  fi
  sleep 2
done

echo "Smoke test failed after 60s — recent logs:" >&2
"${COMPOSE[@]}" logs --tail 50 nginx api-gateway >&2
exit 1
