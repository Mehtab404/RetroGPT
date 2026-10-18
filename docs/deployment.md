# Deployment Guide (AWS EC2 + ECR)

Issue #26. This guide deploys the production stack from
`docker-compose.prod.yml` to a single AWS EC2 host, with images stored in
Amazon ECR. Secrets come from the host environment (`.env.production`, kept in
SSM Parameter Store on AWS) — never from files committed to the repo.

> **Status:** the files and scripts below are complete and validated locally
> (config parse, image builds, full-stack smoke test). The public-URL
> acceptance step is **pending an AWS account + credentials**; follow this
> guide to bring the demo up. Continuous deployment is wired up in
> [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) and activates
> once the repository secrets in section 5 are configured.

- [Architecture](#architecture)
- [1. One-time AWS setup](#1-one-time-aws-setup)
- [2. First deployment](#2-first-deployment)
- [3. Seed the vector database](#3-seed-the-vector-database)
- [4. HTTPS with Let's Encrypt (optional)](#4-https-with-lets-encrypt-optional)
- [5. Updates and rollback](#5-updates-and-rollback)
- [6. Operations](#6-operations)

---

## Architecture

```
             Internet
                │  :80/:443
        ┌───────▼────────┐
        │  edge Nginx     │  (nginx/prod/http(s).conf.template)
        └───┬────────┬────┘
            │        │
   /  ┌─────▼──┐  ┌──▼────────┐   /api/*
      │frontend│  │api-gateway│──────► mongo (password)
      │(static)│  │  :4000    │──────► redis (password)
      └────────┘  └─────┬─────┘
                       │ HTTP + SSE
                 ┌─────▼──────┐
                 │ ai-service │──────► qdrant
                 │  :8000     │──────► Gemini / OpenAI
                 └────────────┘
```

All services run on one host (EC2, e.g. `t3.medium`: 2 vCPU / 4 GB) on a
private Docker network; only Nginx publishes ports. Application images are
built in CI (or locally) and pulled from ECR; `qdrant`, `redis`, `mongo` are
the stock images with named volumes.

---

## 1. One-time AWS setup

**ECR repositories** (one per image):

```bash
aws ecr create-repository --repository-name retrogpt-ai-service
aws ecr create-repository --repository-name retrogpt-api-gateway
aws ecr create-repository --repository-name retrogpt-frontend
```

**EC2 instance** (Ubuntu 24.04 LTS recommended). Security group:

| Port | Source   | Purpose            |
|------|----------|--------------------|
| 22   | your IP  | SSH                |
| 80   | 0.0.0.0/0| HTTP / ACME        |
| 443  | 0.0.0.0/0| HTTPS (if domain)  |

**Docker on the host:**

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"   # re-login afterwards
docker compose version            # plugin ships with the script above
```

**Repository + secrets on the host** (e.g. `/opt/retrogpt`):

```bash
git clone <repo-url> /opt/retrogpt && cd /opt/retrogpt
cp .env.production.example .env.production
# Fill in DOMAIN/PUBLIC_ORIGIN, ECR_REGISTRY, API_KEY, GOOGLE_API_KEY
# (or GROQ_API_KEY / OPENAI_API_KEY for other providers),
# REDIS_PASSWORD, MONGO_PASSWORD. Keep the raw game documents in ./data.
```

> On AWS, store the secret values in SSM Parameter Store (SecureString) and
> render `.env.production` with `aws ssm get-parameters-by-path` during
> deploy — or inject them as GitHub Secrets and let the CD workflow (#27)
> write them on the host. Ports `6379`/`27017`/`6333`/`4000`/`8000` must stay
> closed in the security group; they are only reachable inside the Docker
> network.

---

## 2. First deployment

**Build and push images** (from a workstation or CI with AWS credentials):

```bash
API_KEY=<public-demo-key> \
ECR_REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com \
AWS_REGION=<region> \
  ./deploy/build-push.sh
```

The script tags images with the short git SHA and `latest`.

**Apply on the host:**

```bash
cd /opt/retrogpt
IMAGE_TAG=<sha> ./deploy/remote-deploy.sh
```

`remote-deploy.sh` pulls the images, reconciles the stack and smoke-tests
`http://localhost/api/health`. When it prints `==> Healthy`, the app is live:

- `http://<ec2-public-dns>/` — UI
- `http://<ec2-public-dns>/api/health` — gateway health

Without a domain, keep `.env.production` as-is (`DOMAIN=<ec2-public-dns>` or
`localhost` for a local test) and the HTTP template is used.

**Local dry run** (no AWS needed — builds the same images and runs the same
stack):

```bash
cp .env.production.example .env.production   # set API_KEY, DOMAIN=localhost
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
curl http://localhost/api/health
```

---

## 3. Seed the vector database

The prod stack starts with an empty Qdrant collection. Run ingestion once
(and after data changes):

```bash
cd /opt/retrogpt
docker compose --env-file .env.production -f docker-compose.prod.yml \
  exec ai-service python -m scripts.ingest /data/raw
```

This writes vectors to Qdrant and BM25 statistics to `./data/bm25_stats.json`
on the host. Requires a valid embedding path: `GOOGLE_API_KEY` (default),
`OPENAI_API_KEY`, or `EMBEDDING_PROVIDER=fastembed` (local ONNX, no key).

---

## 4. HTTPS with Let's Encrypt (optional)

When deploying under a real domain that resolves to the host:

1. Set `DOMAIN=<your-domain>` and `PUBLIC_ORIGIN=https://<your-domain>` in
   `.env.production`.
2. Request the certificate (stack must be running; ports 80 reachable):

   ```bash
   DOMAIN=<your-domain> EMAIL=you@example.com ./deploy/init-letsencrypt.sh
   ```

   Use `STAGING=1` while testing against the Let's Encrypt staging CA.
3. The script switches the edge Nginx to `nginx/prod/https.conf.template`
   (redirect 80 → 443, HSTS, TLS 1.2/1.3). Renewals: add a cron/systemd
   timer running

   ```bash
   docker run --rm \
     -v "$PWD/deploy/certbot/conf:/etc/letsencrypt" \
     -v "$PWD/deploy/certbot/www:/var/www/certbot" \
     certbot/certbot renew --quiet
   ```

---

## 5. Updates and rollback

**Continuous deployment**
([`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)) automates
the update path: merging to `main` builds the production images, pushes them
to ECR tagged with the merge commit's short SHA, SSHes to the host to
fast-forward the checkout and run `IMAGE_TAG=<sha> ./deploy/remote-deploy.sh`
(pull + apply + on-host health check), then smoke-tests `/api/health` on
`DEPLOY_HOST` from the runner.

Until the AWS account exists the deploy job **skips with a notice** that lists
the missing secrets, so `main` stays green without AWS. Configure these under
Settings → Secrets and variables → Actions:

| Secret | Required | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | yes | ECR push credentials (least-privilege IAM user) |
| `ECR_REGISTRY` | yes | `<account>.dkr.ecr.<region>.amazonaws.com` |
| `API_KEY` | yes | public demo key baked into the frontend bundle |
| `DEPLOY_HOST` | yes | EC2 public DNS name or IP (smoke-test target) |
| `DEPLOY_SSH_KEY` | yes | private key for `DEPLOY_USER` (public half in the host's `authorized_keys`) |
| `AWS_REGION` | no | defaults to `eu-central-1` |
| `DEPLOY_USER` | no | defaults to `ubuntu` |
| `DEPLOY_PATH` | no | host checkout, defaults to `/opt/retrogpt` |

**Rollback / redeploy** a specific revision: Actions → *CD* → *Run workflow*
with `image_tag` set to a previous SHA (images are immutable in ECR). The
equivalent manual commands:

```bash
# Build + push (workstation or CI with AWS credentials)
API_KEY=<public-demo-key> ECR_REGISTRY=<registry> AWS_REGION=<region> \
  ./deploy/build-push.sh

# On the host
IMAGE_TAG=<sha> ./deploy/remote-deploy.sh
```

---

## 6. Operations

```bash
COMPOSE="docker compose --env-file .env.production -f docker-compose.prod.yml"

$COMPOSE ps                     # service status + health
$COMPOSE logs -f api-gateway    # JSON logs (grep by x-request-id)
$COMPOSE logs -f ai-service
$COMPOSE restart nginx          # after editing nginx templates
$COMPOSE exec redis redis-cli -a "$REDIS_PASSWORD" FLUSHDB   # clear the answer cache
```

Every response carries an `x-request-id`; the same id appears in both
services' JSON logs (see README → Observability), which makes a live issue
traceable end-to-end.

---

## Related

- [`docker-compose.prod.yml`](../docker-compose.prod.yml) — the stack
- [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) — continuous deployment (build → ECR → host → smoke test)
- [`deploy/`](../deploy) — build-push, remote-deploy, Let's Encrypt scripts
- [`.env.production.example`](../.env.production.example) — secrets template
- [ADR 0005](../docs/decisions.md) — structured logging / request IDs
