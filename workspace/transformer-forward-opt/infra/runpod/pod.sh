#!/usr/bin/env bash
# Per-pod setup on a volume that bootstrap.sh has already prepared.
#
# Run it on every new pod, first thing. Everything it touches lives on the
# container disk, which does not survive a terminate: the links from ~ into the
# volume, the .bashrc hook, tmux, nsys, and the venv. It is idempotent and takes
# seconds; the tools that persist are already on the volume.
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

# Everything below needs the volume's PATH and VENV_DIR.
# shellcheck disable=SC1090
source "$ENV_SH"

step "nsight systems"
# The CUDA images ship ncu but not nsys, and ncu is the one the driver refuses
# on a container without CAP_SYS_ADMIN. nsys traces through CUPTI rather than
# the performance counters, so it works where ncu does not -- per-kernel time,
# launch counts, the gaps between kernels. docs/runpod.md, "The NCU question".
# It installs onto the container disk, which is why this is here and not in
# bootstrap.sh: it has to happen again on every pod.
if command -v nsys >/dev/null 2>&1; then
  ok "$(nsys --version | tail -1)"
else
  pkg="nsight-systems"
  ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')"
  if [[ -n "$ver" ]]; then pkg="cuda-nsight-systems-$ver"; fi
  if (apt-get update -qq && apt-get install -y -qq "$pkg") >/dev/null 2>&1; then
    ok "installed $pkg"
  else
    warn "could not install $pkg -- nsys profiling unavailable"
  fi
fi

step "python environment"
# The venv lives on the pod's local disk (VENV_DIR, set in env.sh) so that the
# network volume never holds torch. The repository survives a terminate and the
# venv does not, so rebuild it here. It costs seconds: torch comes from the
# image and only the small packages are installed.
venv="${VENV_DIR:-$REPO/.venv}"
if [[ -x "$venv/bin/python" ]]; then
  ok "$("$venv/bin/python" -c 'import torch,sys;print(f"python {sys.version.split()[0]}  torch {torch.__version__}")')"
else
  ( cd "$REPO" && ./scripts/setup_env.sh ) || warn "setup_env.sh failed -- no venv, nothing can run"
fi

step "D0 gate"
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
