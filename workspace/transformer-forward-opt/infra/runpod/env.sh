# /workspace/env.sh -- sourced by every interactive shell on the pod.
#
# Installed once by bootstrap.sh, then maintained by hand on the volume. No
# `set -e` / `set -u` in here: it runs inside interactive shells, where a
# failing command would close the terminal.

export TOOLS="/workspace/tools"
export NVM_DIR="$TOOLS/nvm"

# uv: installs, caches, managed pythons and tools all live on the volume.
export UV_INSTALL_DIR="$TOOLS/uv"
export UV_CACHE_DIR="$TOOLS/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TOOLS/uv-python"
export UV_TOOL_DIR="$TOOLS/uv-tools"
export UV_TOOL_BIN_DIR="$TOOLS/bin"

# npm: the cache on the volume; no NPM_CONFIG_PREFIX, which breaks nvm.
export NPM_CONFIG_CACHE="$TOOLS/npm-cache"

# node: found by directory, never by a pinned minor version. nvm installs
# "the latest 24.x" and a hard-coded v24.19.0 breaks silently the next time
# bootstrap installs 24.20.0. nvm.sh itself is not sourced here -- it is over a
# thousand lines on a network disk and would slow every shell down.
_node_bins=("$NVM_DIR"/versions/node/*/bin)
if [ "${#_node_bins[@]}" -eq 1 ] && [ -d "${_node_bins[0]}" ]; then
  NODE_BIN="${_node_bins[0]}"
elif [ "${#_node_bins[@]}" -gt 1 ]; then
  NODE_BIN="$(printf '%s\n' "${_node_bins[@]}" | sort -V | tail -n1)"
else
  NODE_BIN=""
fi
unset _node_bins

if [ -x "$UV_INSTALL_DIR/bin/uv" ]; then
  UV_BIN="$UV_INSTALL_DIR/bin"
else
  UV_BIN="$UV_INSTALL_DIR"
fi

# ${VAR:+...}: an empty PATH component means the current directory.
export PATH="${NODE_BIN:+$NODE_BIN:}$UV_BIN:$TOOLS/bin:$PATH"

# nvm on demand only.
nvm() {
  unset -f nvm
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm "$@"
}

export ODYSSEY_ROOT="/workspace/odyssey"

# Machine-local additions (API keys belong here, not in this file).
[ -f "/workspace/env.local.sh" ] && . "/workspace/env.local.sh"
