#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
set -euo pipefail
package_root="$(cd "$(dirname "$0")" && pwd)"
setup_env="${XDG_DATA_HOME:-$HOME/.local/share}/owner-mode/venv"
if [ ! -x "$setup_env/bin/python" ]; then
  python3 -m venv "$setup_env"
fi
setup_banner=true
for setup_arg in "$@"; do
  case "$setup_arg" in --detect|--version|--config|--config=*|--init-key|--init-key=*|--stop-engine|--stop-engine=*) setup_banner=false ;; esac
done
if "$setup_banner"; then
  printf 'Preparing Aevonix FlashAlign setup...\n'
fi
"$setup_env/bin/python" -m pip install --quiet --disable-pip-version-check -r "$package_root/hermes-owner-mode/requirements-setup.txt"
exec "$setup_env/bin/python" "$package_root/hermes-owner-mode/setup.py" "$@"
