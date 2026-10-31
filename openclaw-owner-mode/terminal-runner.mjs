// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
/* Executable entrypoint kept separate from the synchronously loaded plugin. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runNativeTerminal } from './terminal.mjs';

const fd = Number(process.argv[2]);
const receipt = JSON.parse(readFileSync(fd, 'utf8'));
process.env.OPENCLAW_STATE_DIR = join(receipt.home, '.openclaw');
process.env.OPENCLAW_CONFIG_PATH = join(receipt.home, '.openclaw', 'openclaw.json');
await runNativeTerminal(receipt.root, receipt.id);
// The native runtime retains background handles after its awaited shutdown.
process.exit(0);
