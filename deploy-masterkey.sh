#!/usr/bin/env bash
# Deploy OpenSEO to Cloudflare with the account-owned API token instead of
# `alchemy login` (OAuth). Alchemy reads CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID
# from the environment when CI=1 (its "env" credential method) and, with
# --yes, bootstraps its state-store worker on first run.
set -euo pipefail
cd "$(dirname "$0")"
KEYFILE="${CLOUDFLARE_MASTERKEY_FILE:-$HOME/workshop/cloudflare_masterkey.txt}"
export CLOUDFLARE_API_TOKEN="$(grep '^API_TOKEN' "$KEYFILE" | sed 's/.*"\(.*\)".*/\1/')"
export CLOUDFLARE_ACCOUNT_ID="$(grep '^ACCOUNT_ID' "$KEYFILE" | sed 's/.*"\(.*\)".*/\1/')"
export CI=1
[ -n "$CLOUDFLARE_API_TOKEN" ] && [ -n "$CLOUDFLARE_ACCOUNT_ID" ] || { echo "masterkey file unreadable: $KEYFILE" >&2; exit 1; }
exec corepack pnpm deploy:selfhost --yes "$@"
