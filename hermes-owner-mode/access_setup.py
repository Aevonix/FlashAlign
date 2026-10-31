# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Wizard choices derived from the installed transport registry."""
import os

from .contacts import pair


def terminal_harnesses(contacts):
    return {row.get('account') if isinstance(row, dict) else row[1]
            for row in contacts if (row.get('channel') if isinstance(row, dict) else row[0]) == 'local-terminal'}


def choose_verified_contact(console, harnesses, store, *, ask=None):
    ask = input if ask is None else ask
    registry = store.registry
    choices = sorted(registry.ids('native_message') | registry.ids('local_interactive'))
    if not choices:
        raise ValueError('No installed adapter can verify a contact or local terminal session')
    for channel in choices:
        console.line(f'  {channel}: {registry.get(channel).label or channel}')
    default = 'whatsapp' if 'whatsapp' in choices else choices[0]
    channel = ask(f'Verified channel [{"/".join(choices)}] [{default}]: ').strip() or default
    if channel not in choices:
        raise ValueError('Choose an installed transport adapter')
    if registry.supports(channel, 'local_interactive'):
        default_account = harnesses[0] if len(harnesses) == 1 else 'hermes'
        account = ask(f'Terminal harness [{"/".join(harnesses)}] [{default_account}]: ').strip() or default_account
        if account not in harnesses:
            raise ValueError('Choose a selected harness for the local terminal')
        contact = 'uid:'+str(os.getuid())
        console.line('Local terminal access also requires an explicitly authorized, expiring session.')
    else:
        contact = ask('Approved contact ID or phone number: ').strip()
        account = ask('Channel account [default]: ').strip() or 'default'
    return pair(channel, account, contact, registry=registry)
