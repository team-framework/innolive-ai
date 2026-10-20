#!/usr/bin/env bash
# Optional user-service renewal for a Tailscale HTTPS certificate.
set -euo pipefail
: "${TLS_DOMAIN:?Set TLS_DOMAIN in .runtime/server.env}"
: "${SSL_CERTFILE:?Set SSL_CERTFILE in .runtime/server.env}"
: "${SSL_KEYFILE:?Set SSL_KEYFILE in .runtime/server.env}"
before=$(sha256sum "$SSL_CERTFILE" 2>/dev/null || true)
tailscale cert --cert-file "$SSL_CERTFILE" --key-file "$SSL_KEYFILE" "$TLS_DOMAIN"
chmod 600 "$SSL_KEYFILE"
after=$(sha256sum "$SSL_CERTFILE")
if [[ "$before" != "$after" ]]; then
  systemctl --user try-restart mobilefaceswap.service
fi
