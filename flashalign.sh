#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
setup_env="${XDG_DATA_HOME:-$HOME/.local/share}/owner-mode/venv"
if [ ! -x "$setup_env/bin/flashalign" ]; then
  printf 'Run bash ./install.sh to set up FlashAlign first.\n' >&2
  exit 1
fi
exec "$setup_env/bin/flashalign" "$@"
