# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Normalize vLLM long-option aliases before enforcing launcher boundaries."""


def normalize_arguments(arguments):
    result = []
    for value in arguments:
        if value.startswith('--'):
            option, separator, inline = value.partition('=')
            name, dot, nested = option.partition('.')
            value = name.replace('_', '-') + dot + nested + separator + inline
        result.append(value)
    return result


def option_name(value):
    return value.split('=', 1)[0].split('.', 1)[0]


def require_full_options(arguments, names):
    for value in arguments:
        name = option_name(value)
        if (name.startswith('--') and name != '--' and name not in names
                and any(option.startswith(name) for option in names)):
            raise ValueError('Use a complete vLLM option name instead of '+name)
