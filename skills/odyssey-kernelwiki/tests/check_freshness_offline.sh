#!/usr/bin/env bash
# AC-8 static-analysis guard: scripts/check_version_freshness.py must NOT
# perform network calls. Reject any matching pattern.
set -euo pipefail
script="scripts/check_version_freshness.py"
if [[ ! -f "$script" ]]; then
  echo "ERROR: $script not found" >&2
  exit 2
fi
hits=$(grep -cE "(^import urllib|^import requests|^from urllib|^from requests|subprocess\.[a-zA-Z_]+\(.*['\"]gh |subprocess\.[a-zA-Z_]+\(.*['\"]git fetch|urlopen|urllib\.request)" "$script" || true)
if [[ "$hits" -ne 0 ]]; then
  echo "ERROR: $script appears to use network APIs (grep returned $hits matches)" >&2
  grep -nE "(^import urllib|^import requests|^from urllib|^from requests|subprocess\.[a-zA-Z_]+\(.*['\"]gh |subprocess\.[a-zA-Z_]+\(.*['\"]git fetch|urlopen|urllib\.request)" "$script" >&2 || true
  exit 1
fi
echo "OK: $script is network-free"
