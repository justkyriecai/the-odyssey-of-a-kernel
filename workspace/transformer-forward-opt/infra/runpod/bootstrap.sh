#!/usr/bin/env bash
# One-time setup of a RunPod network volume for this workspace.
#
# Run once per VOLUME, not once per pod. Everything it installs lands under
# /workspace, which is the network volume, so a pod can be terminated and a new
# one -- a different GPU, a different image -- attached to the same volume picks
# up where the last one left off in the time it takes to source env.sh.
#
#   /workspace/tools/            node (via nvm), uv, uv's caches and pythons,
#                                claude, the npm cache
#   /workspace/tools/.claude*    Claude Code state, linked from ~ by pod.sh
#   /workspace/odyssey/          this repository, with its .venv
#   /workspace/env.sh            sourced by every shell; maintained by hand,
#                                copied from infra/runpod/env.sh if absent
#
# Two facts about /workspace worth knowing before editing this file:
#   - it is a network filesystem (MooseFS over FUSE) in a user namespace:
#     chown is refused and `cp -p` / `cp -a` fail on ownership. Use `install`.
#   - it is shared: never run two pods against the same volume at once.
#
# After this: bash pod.sh (on this pod and every later one).
set -euo pipefail

WORKSPACE=/workspace
TOOLS="$WORKSPACE/tools"
REPO="$WORKSPACE/odyssey"
REPO_URL="${REPO_URL:-https://github.com/justkyriecai/the-odyssey-of-a-kernel.git}"
NODE_VERSION="${NODE_VERSION:-24}"
NVM_VERSION="${NVM_VERSION:-v0.40.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '   %s\n' "$1"; }
die()  { printf '   %s\n' "$1" >&2; exit 1; }

[[ -d "$WORKSPACE" ]] || die "$WORKSPACE does not exist -- no volume is mounted"
if command -v mountpoint >/dev/null 2>&1 && ! mountpoint -q "$WORKSPACE"; then
  echo "WARNING: $WORKSPACE is not a mount point. Without a network volume attached,"
  echo "         nothing installed here survives the pod. Re-create the pod with one."
fi
command -v curl >/dev/null || die "curl is missing"
command -v git  >/dev/null || die "git is missing"

# Every cache and install location points into the volume. NPM_CONFIG_PREFIX
# must stay unset: nvm refuses to work with it, and nvm's own tree is already
# on the volume so global npm packages persist without it.
export NVM_DIR="$TOOLS/nvm"
export UV_INSTALL_DIR="$TOOLS/uv"
export UV_CACHE_DIR="$TOOLS/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TOOLS/uv-python"
export UV_TOOL_DIR="$TOOLS/uv-tools"
export UV_TOOL_BIN_DIR="$TOOLS/bin"
export NPM_CONFIG_CACHE="$TOOLS/npm-cache"
unset NPM_CONFIG_PREFIX PREFIX 2>/dev/null || true

mkdir -p "$TOOLS"/{bin,nvm,uv,uv-cache,uv-python,uv-tools,npm-cache,.claude}
[[ -s "$TOOLS/.claude.json" ]] || echo '{}' > "$TOOLS/.claude.json"

step "[1/5] nvm + node $NODE_VERSION"
if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  # PROFILE=/dev/null: nvm must not edit .bashrc; env.sh owns PATH.
  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" \
    | PROFILE=/dev/null bash || die "nvm install failed"
fi
# nvm.sh is not compatible with `set -u`; load it with the guards off.
set +euo pipefail
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm install "$NODE_VERSION" >/dev/null
nvm alias default "$NODE_VERSION" >/dev/null
nvm use "$NODE_VERSION" >/dev/null
set -euo pipefail
command -v node >/dev/null || die "node did not enter PATH"
ok "node $(node --version) -> $(dirname "$(command -v node)")"

step "[2/5] uv"
if [[ ! -x "$UV_INSTALL_DIR/bin/uv" && ! -x "$UV_INSTALL_DIR/uv" ]]; then
  curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null || die "uv install failed"
fi
if   [[ -x "$UV_INSTALL_DIR/bin/uv" ]]; then UV_BIN="$UV_INSTALL_DIR/bin"
elif [[ -x "$UV_INSTALL_DIR/uv"     ]]; then UV_BIN="$UV_INSTALL_DIR"
else die "uv installed but no executable under $UV_INSTALL_DIR"; fi
export PATH="$UV_BIN:$TOOLS/bin:$PATH"
ok "uv $(uv --version | awk '{print $2}') -> $UV_BIN"

step "[3/5] claude code"
command -v claude >/dev/null || npm i -g @anthropic-ai/claude-code >/dev/null
ok "$(claude --version)"

step "[4/5] this repository -> $REPO"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --recurse-submodules "$REPO_URL" "$REPO"
else
  git -C "$REPO" pull --ff-only || true
  git -C "$REPO" submodule update --init --recursive
fi
# uv picks a managed interpreter (installed under the volume) for the venv,
# so the venv keeps working when the pod image changes. Not exported: the
# preference is for this one command, not for every uv call afterwards.
( cd "$REPO" && UV_PYTHON_PREFERENCE=only-managed PYTHON_VERSION="$PYTHON_VERSION" ./scripts/setup_env.sh )

step "[5/5] env.sh"
if [[ ! -f "$WORKSPACE/env.sh" ]]; then
  install -m 0644 "$INFRA/env.sh" "$WORKSPACE/env.sh"
  ok "installed $WORKSPACE/env.sh (maintained by hand from here on)"
else
  ok "$WORKSPACE/env.sh already present, not overwritten"
fi

echo
echo "bootstrap complete. On this pod and every pod after it:"
echo "  bash $REPO/workspace/transformer-forward-opt/infra/runpod/pod.sh"
