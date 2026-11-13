# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Patch changed JSON5 properties while retaining unrelated source text."""
from dataclasses import dataclass
import json
import re

import json5


_TOKENS = re.compile(r'''[\s\ufeff]+|//[^\r\n\u2028\u2029]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|[{}\[\]:,]|[^\s\ufeff{}\[\]:,'"/]+''')


@dataclass
class _Value:
    start: int
    end: int
    value: object
    members: object = None
    close: int = 0


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def patch_json5(source, config):
    """Replace changed values, remove deleted properties, and append new keys.

    Objects are patched recursively. Arrays and scalar values are replaced as
    a whole when changed; their internal comments and formatting are not kept.
    """
    original = json5.loads(source)
    if _same(original, config):
        return source
    tokens = [match for match in _TOKENS.finditer(source)
              if not (not match.group().replace('\ufeff', '').strip()
                      or match.group().startswith(('//', '/*')))]
    position = 0

    def parse():
        nonlocal position
        first = tokens[position]
        position += 1
        members = None
        if first.group() == '{':
            members = []
            while tokens[position].group() != '}':
                key = tokens[position]
                name = next(iter(json5.loads('{' + key.group() + ':null}')))
                position += 2  # property name and colon (validated by json5.loads)
                value = parse()
                comma = None
                if tokens[position].group() == ',':
                    comma = tokens[position]
                    position += 1
                members.append((name, key.start(), value, comma))
            last = tokens[position]
            position += 1
        elif first.group() == '[':
            while tokens[position].group() != ']':
                parse()
                if tokens[position].group() == ',':
                    position += 1
            last = tokens[position]
            position += 1
        else:
            last = first
        return _Value(first.start(), last.end(), json5.loads(source[first.start():last.end()]),
                      members, last.start())

    root = parse()
    edits = []

    def encoded(value, depth):
        return json.dumps(value, indent=2, ensure_ascii=False).replace('\n', '\n' + '  ' * depth)

    def patch(node, desired, depth=0):
        if _same(node.value, desired):
            return
        if node.members is None or not isinstance(desired, dict):
            edits.append((node.start, node.end, encoded(desired, depth)))
            return
        kept = []
        present = set()
        for name, start, value, comma in node.members:
            present.add(name)
            if name not in desired:
                edits.append((start, comma.end() if comma else value.end, ''))
            else:
                kept.append((value, comma))
                if not _same(node.value[name], desired[name]):
                    patch(value, desired[name], depth + 1)
        added = [name for name in desired if name not in present]
        if added:
            if kept and kept[-1][1] is None:
                end = kept[-1][0].end
                edits.append((end, end, ','))
            indent = '  ' * (depth + 1)
            fields = [indent + json.dumps(name, ensure_ascii=False) + ': ' + encoded(desired[name], depth + 1)
                      for name in added]
            edits.append((node.close, node.close, '\n' + ',\n'.join(fields) + '\n' + '  ' * depth))
        elif kept and kept[-1][1] is not None and node.members[-1][3] is None:
            # Deleting the last property must not introduce a trailing comma
            # into a plain JSON file that is also read by strict JSON clients.
            comma = kept[-1][1]
            edits.append((comma.start(), comma.end(), ''))

    patch(root, config)
    result = source
    ordered = sorted(enumerate(edits), key=lambda item: (*item[1][:2], item[0]), reverse=True)
    for _, (start, end, replacement) in ordered:
        result = result[:start] + replacement + result[end:]
    if not _same(json5.loads(result), config):
        raise ValueError('JSON5 configuration patch did not preserve its values')
    return result
