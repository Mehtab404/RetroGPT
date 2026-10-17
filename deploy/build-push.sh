#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build and push the RetroGPT production images to Amazon ECR (Issue #26).
#
# Usage (CI or a workstation with AWS credentials):
#
#   API_KEY=<public-demo-key> \
#   ECR_REGISTRY=123456789012.dkr.ecr.eu-central-1.amazonaws.com \
#   AWS_REGION=eu-central-1 \
#     ./deploy/build-push.sh
#
# IMAGE_TAG defaults to the short git SHA; each image is also tagged :latest.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

ECR_REGISTRY="${ECR_REGISTRY:?set ECR_REGISTRY, e.g. 123456789012.dkr.ecr.eu-central-1.amazonaws.com}"
AWS_REGION="${AWS_REGION:-eu-central-1}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

# The frontend bundle bakes the public demo API key at build time.
API_KEY="${API_KEY:?set API_KEY (public frontend key)}"

# Runtime-only secrets are irrelevant to `docker compose build`, but the prod
# compose file interpolates them anyway — provide placeholders when absent.
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-build-placeholder}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-build-placeholder}"
export MONGO_PASSWORD="${MONGO_PASSWORD:-build-placeholder}"
export DOMAIN="${DOMAIN:-localhost}"

export ECR_REGISTRY IMAGE_TAG API_KEY

SERVICES=(ai-service api-gateway frontend)

echo "==> Logging in to ECR ($ECR_REGISTRY)"
aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "==> Building production images (tag: $IMAGE_TAG)"
docker compose -f docker-compose.prod.yml build

for service in "${SERVICES[@]}"; do
  image="$ECR_REGISTRY/retrogpt-$service"
  echo "==> Pushing $image:$IMAGE_TAG"
  docker push "$image:$IMAGE_TAG"
  docker tag "$image:$IMAGE_TAG" "$image:latest"
  docker push "$image:latest"
done

echo "==> Done. Deploy with IMAGE_TAG=$IMAGE_TAG"
