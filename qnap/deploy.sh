#!/usr/bin/env bash
# Build the agent image on this Mac for the NAS's actual CPU architecture and
# stream it straight into `docker load` over a single SSH pipe, then bring the
# stack up there.
#
# Why a stream and not a copy: scp and sftp DO NOT WORK against this QNAP. Its
# sshd exposes no SFTP subsystem, so scp fails with "subsystem request failed on
# channel 0" and exit 255. Plain SSH command execution with stdin piped through
# is what every SSH server supports, so that is what this uses. No source code
# reaches the NAS — only the finished image plus the small config files that
# DEPLOY-NAS.md lists.
#
# Why a cross-build: this Mac is arm64 and the NAS is x86_64. Building for the
# wrong architecture produces an image that dies with "exec format error" on
# first start — not a partial failure, a silent non-start. NAS_PLATFORM is
# therefore pinned here rather than inferred, and was confirmed with
# `ssh -p "$NAS_SSH_PORT" "$NAS_HOST" uname -m`.
#
# Usage:  ./qnap/deploy.sh            # build, ship, restart
#         ./qnap/deploy.sh --no-up    # build and ship only
#
# Re-deploying later is this script again: it is idempotent, and `compose up -d`
# only recreates containers whose definition or image actually changed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Real per-deployment values live in .deploy.env, git-ignored (see
# deploy.env.example) — auto-sourced here so nothing needs exporting by hand.
[ -f "${REPO_ROOT}/.deploy.env" ] && . "${REPO_ROOT}/.deploy.env"

NAS_HOST="${NAS_HOST:-deploy@nas.local}"
NAS_SSH_PORT="${NAS_SSH_PORT:-22}"
NAS_PLATFORM="${NAS_PLATFORM:-linux/amd64}"
NAS_APP_DIR="${NAS_APP_DIR:-/share/Container/podcast-digest}"
# Container Station's docker is not on PATH for non-interactive SSH commands —
# QNAP only wires it in for interactive logins — so it is addressed by full path.
NAS_DOCKER_BIN="${NAS_DOCKER_BIN:-/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker}"
# Must match the `image:` in docker-compose.yml, or compose will ignore what we
# just loaded and try to pull a tag that does not exist in any registry.
TAG="podcast-agent:1.0.0"

ssh_nas() { ssh -p "$NAS_SSH_PORT" "$NAS_HOST" "$@"; }

echo "== building $TAG for $NAS_PLATFORM =="
docker buildx build --platform "$NAS_PLATFORM" -t "$TAG" --load "$REPO_ROOT"

echo "== streaming into 'docker load' on $NAS_HOST (no intermediate file either end) =="
docker save "$TAG" | gzip | ssh_nas "gunzip -c | '$NAS_DOCKER_BIN' load"

# Confirm the NAS actually has it. `docker load` failing mid-stream still exits
# 0 often enough that trusting it is how you end up debugging the wrong thing.
echo "== verifying the image landed =="
ssh_nas "'$NAS_DOCKER_BIN' image inspect '$TAG' --format 'loaded: {{.Id}} ({{.Architecture}})'"

if [[ "${1:-}" == "--no-up" ]]; then
  echo "done (--no-up: stack not restarted)."
  exit 0
fi

echo "== bringing the stack up on the NAS =="
# Both files, in this order: the base is the portable deployment, the .nas one
# carries everything specific to this machine. See docker-compose.nas.yml.
ssh_nas "cd '$NAS_APP_DIR' && '$NAS_DOCKER_BIN' compose \
  -f docker-compose.yml -f docker-compose.nas.yml up -d"

# NAS_LAN_IP is docker-compose.nas.yml's own var (see .env.example) — read
# directly rather than sourced, so this script doesn't pull in the rest of
# .env's secrets just to print an address.
nas_lan_ip="$(grep -E '^NAS_LAN_IP=' "${REPO_ROOT}/.env" 2>/dev/null | cut -d= -f2-)"
nas_lan_ip="${nas_lan_ip:-<NAS_LAN_IP from .env>}"

echo
echo "done. Console: http://${nas_lan_ip}:8080/admin"
echo "Logs:  ssh -p $NAS_SSH_PORT $NAS_HOST \"cd '$NAS_APP_DIR' && '$NAS_DOCKER_BIN' compose logs -f podcast-agent\""
echo
echo "Note: the NAS host itself cannot reach ${nas_lan_ip} (macvlan); check the"
echo "console from another machine on the LAN."
