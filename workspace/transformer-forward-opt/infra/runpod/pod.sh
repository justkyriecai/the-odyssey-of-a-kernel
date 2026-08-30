#!/usr/bin/env bash
# Per-pod setup on a volume that bootstrap.sh has already prepared.
#
# Run it on every new pod, first thing. Everything it touches lives on the
# container disk, which does not survive a terminate: the links from ~ into the
# volume, the .bashrc hook, tmux. It is idempotent and takes seconds; the tools
# themselves are already on the volume.
#
# Must be run with `bash pod.sh`, not sourced: `set -e` would otherwise stay
# behind in the interactive shell and close it on the next failing command.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this with: bash ${BASH_SOURCE[0]}" >&2
  return 1
fi
set -euo pipefail

WORKSPACE=/workspace
TOOLS="$WORKSPACE/tools"
REPO="$WORKSPACE/odyssey"
ENV_SH="$WORKSPACE/env.sh"
BEGIN_MARK="# >>> odyssey workspace >>>"
END_MARK="# <<< odyssey workspace <<<"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '   %s\n' "$1"; }
warn() { printf '   WARNING: %s\n' "$1" >&2; }
die()  { printf '   %s\n' "$1" >&2; exit 1; }

[[ -d "$TOOLS" && -f "$ENV_SH" ]] || die "volume not bootstrapped: run bootstrap.sh first"

# Link a path in ~ to its home on the volume. If the target already has real
# content and the volume copy is empty, move the content into the volume first;
# otherwise a non-empty directory is left alone and reported.
link() {
  local src="$1" dst="$2"
  if [[ -L "$dst" && "$(readlink "$dst")" == "$src" ]]; then
    ok "$dst -> $src"
    return 0
  fi
  if [[ -d "$dst" && ! -L "$dst" && -n "$(ls -A "$dst" 2>/dev/null)" ]]; then
    if [[ -z "$(ls -A "$src" 2>/dev/null)" ]]; then
      # `install`, not `cp -a`: the volume refuses ownership changes.
      (cd "$dst" && find . -type d -exec mkdir -p "$src/{}" \; \
                 && find . -type f -exec install -m 0644 "{}" "$src/{}" \;)
      mv "$dst" "$dst.pre-link.bak"
      ok "moved existing $dst into the volume (backup at $dst.pre-link.bak)"
    else
      warn "$dst is a real directory with content and the volume copy is not empty; left alone"
      return 0
    fi
  elif [[ -f "$dst" && ! -L "$dst" ]]; then
    if [[ ! -s "$src" || "$(cat "$src")" == "{}" ]]; then
      install -m 0644 "$dst" "$src"
    fi
    mv "$dst" "$dst.pre-link.bak"
  fi
  rm -rf "$dst" 2>/dev/null || true
  ln -s "$src" "$dst"
  ok "$dst -> $src"
}

step "agent state -> volume"
link "$TOOLS/.claude"      "$HOME/.claude"
link "$TOOLS/.claude.json" "$HOME/.claude.json"

step "tmux"
if ! command -v tmux >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1 && ok "installed" || warn "could not install tmux"
else
  ok "$(tmux -V)"
fi

step ".bashrc hook"
rc="$HOME/.bashrc"
[[ -f "$rc" ]] || touch "$rc"
tmp="$(mktemp)"
awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
  $0 == b { skip = 1; next }
  $0 == e { skip = 0; next }
  skip    { next }
  { print }
' "$rc" > "$tmp"
{
  echo "$BEGIN_MARK"
  echo "[ -f $ENV_SH ] && source $ENV_SH"
  echo "$END_MARK"
} >> "$tmp"
cat "$tmp" > "$rc"      # keep the inode; the file may be a link
rm -f "$tmp"
ok "$rc sources $ENV_SH"

step "D0 gate"
# shellcheck disable=SC1090
source "$ENV_SH"
if [[ -x "$REPO/scripts/check_gpu.sh" ]]; then
  "$REPO/scripts/check_gpu.sh" transformer-forward-opt || warn "check_gpu.sh reported NOT READY -- read it before renting more hours"
else
  warn "$REPO/scripts/check_gpu.sh not found"
fi

echo
echo "pod ready. Open a new shell (or: source $ENV_SH), then:"
echo "  tmux new -s odyssey"
echo "  cd $REPO && ./scripts/setup_agent.sh      # first pod on this volume only"
echo "  cd $REPO/workspace/transformer-forward-opt && claude"
