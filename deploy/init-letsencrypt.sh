#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Issue a Let's Encrypt certificate for $DOMAIN and switch the edge Nginx to
# HTTPS (Issue #26, optional step: only needed when deploying with a domain).
#
# Usage on the deployment host (stack already running):
#
#   DOMAIN=demo.example.com EMAIL=you@example.com ./deploy/init-letsencrypt.sh
#
#   STAGING=1   # use the Let's Encrypt staging CA while testing
#
# Certificates land in ./deploy/certbot/conf (mounted into nginx) and the
# ACME HTTP-01 challenge is served from ./deploy/certbot/www.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

DOMAIN="${DOMAIN:?set DOMAIN (e.g. demo.example.com)}"
EMAIL="${EMAIL:?set EMAIL (ACME contact address)}"
STAGING="${STAGING:-0}"

mkdir -p deploy/certbot/conf deploy/certbot/www

# 1. Make sure nginx is serving the HTTP template so the ACME webroot works.
echo "==> Starting edge nginx with the HTTP template"
NGINX_CONF_FILE=./nginx/prod/http.conf.template \
  docker compose --env-file .env.production -f docker-compose.prod.yml up -d nginx

# 2. Request the certificate.
staging_flag=()
if [ "$STAGING" != "0" ]; then
  staging_flag=(--staging)
fi

echo "==> Requesting certificate for $DOMAIN"
docker run --rm \
  -v "$PWD/deploy/certbot/conf:/etc/letsencrypt" \
  -v "$PWD/deploy/certbot/www:/var/www/certbot" \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  "${staging_flag[@]}" \
  -d "$DOMAIN" --email "$EMAIL" --agree-tos --no-eff-email

# 3. Switch nginx to the HTTPS template.
echo "==> Switching edge nginx to HTTPS"
NGINX_CONF_FILE=./nginx/prod/https.conf.template \
  docker compose --env-file .env.production -f docker-compose.prod.yml up -d nginx

echo "==> Done. https://$DOMAIN should now be live."
echo "    Remember to set PUBLIC_ORIGIN=https://$DOMAIN in .env.production."
