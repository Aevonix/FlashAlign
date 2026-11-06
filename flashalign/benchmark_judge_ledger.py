# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Durable judge attempts bound to one collection record and judge run."""
import os
from pathlib import Path
import re

from .benchmark_io import atomic_json, digest, private_directory, read_json, text_digest


BEGIN_SCHEMA = 'flashalign.judge-attempt-begin.v2'
END_SCHEMA = 'flashalign.judge-attempt-end.v2'
_EVENT = re.compile(r'[0-9]{3}\.(begin|end)\.json\Z')
_TEMPORARY = re.compile(r'\.[0-9]{3}\.(begin|end)\.json\.[0-9a-f]{24}\Z')


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_directory(path):
    path = private_directory(path, create=True)
    sync_directory(path.parent)
    return path


def validate_inventory(root, count):
    root = private_directory(root)
    for path in root.iterdir():
        if not re.fullmatch(r'[0-9]{6}', path.name) or int(path.name) >= count:
            raise ValueError('Unexpected judge attempt record')
        private_directory(path)


def ledger_digest(rows):
    return digest(rows)


class JudgeLedger:
    """The caller holds the run lock for every mutation.

    Published begin records are never removed. An incomplete end therefore
    consumes the attempt and permanently prevents another semantic draw.
    """
    def __init__(self, root, ordinal, record, descriptor):
        if (descriptor.get('protocol') != 'flashalign.fourway-judge.v2'
                or not isinstance(descriptor.get('judge_run_id'), str)
                or re.fullmatch(r'[0-9a-f]{32}', descriptor['judge_run_id']) is None
                or type(descriptor.get('maximum_transport_attempts')) is not int
                or descriptor['maximum_transport_attempts'] != 3):
            raise ValueError('Invalid durable judge descriptor')
        self.directory = Path(root)/f'{ordinal:06d}'
        self.binding = {'judge_run_id': descriptor['judge_run_id'],
                        'judge_descriptor_sha256': digest(descriptor),
                        'record_id': record['record_id'], 'record_sha256': record['record_sha256']}
        self.maximum = descriptor['maximum_transport_attempts']

    def _event(self, schema, **fields):
        value = {'schema': schema, **self.binding, **fields}
        return {**value, 'event_sha256': digest(value)}

    def _validate_event(self, value, schema, fields):
        expected = {'schema', *self.binding, *fields, 'event_sha256'}
        if not isinstance(value, dict) or set(value) != expected or value.get('schema') != schema:
            raise ValueError('Invalid judge attempt event')
        if (value['event_sha256'] != digest({key: item for key, item in value.items() if key != 'event_sha256'})
                or any(value.get(key) != item for key, item in self.binding.items())):
            raise ValueError('Judge attempt provenance mismatch')

    def read(self):
        if not self.directory.exists():
            return []
        private_directory(self.directory)
        files = list(self.directory.iterdir())
        if any(not (_EVENT.fullmatch(path.name) or _TEMPORARY.fullmatch(path.name)) for path in files):
            raise ValueError('Unexpected judge attempt artifact')
        starts = sorted(path for path in files if path.name.endswith('.begin.json'))
        ends = {path.name for path in files if path.name.endswith('.end.json')}
        if len(starts) > self.maximum:
            raise ValueError('Judge attempt budget exceeded')
        result, previous = [], None
        terminal = False
        for number, path in enumerate(starts, 1):
            if path.name != f'{number:03d}.begin.json' or terminal:
                raise ValueError('Judge attempt sequence permits a semantic reroll')
            begin = read_json(path)
            self._validate_event(begin, BEGIN_SCHEMA, {'attempt', 'previous_end_sha256'})
            if type(begin['attempt']) is not int or begin['attempt'] != number or begin['previous_end_sha256'] != previous:
                raise ValueError('Judge attempt chain mismatch')
            end_path = self.directory/f'{number:03d}.end.json'
            end = None
            if end_path.exists():
                end = read_json(end_path)
                self._validate_event(end, END_SCHEMA, {'attempt', 'begin_sha256', 'outcome', 'error',
                    'error_family', 'retryable', 'judgment', 'raw_judge_answer', 'response_sha256'})
                if type(end['attempt']) is not int or end['attempt'] != number or end['begin_sha256'] != begin['event_sha256']:
                    raise ValueError('Judge attempt end is bound to another begin')
                if type(end['retryable']) is not bool:
                    raise ValueError('Invalid judge retry declaration')
                raw = end['raw_judge_answer']
                if raw is not None and not isinstance(raw, str):
                    raise ValueError('Invalid raw judge response')
                if end['response_sha256'] != (text_digest(raw) if raw is not None else None):
                    raise ValueError('Judge response hash mismatch')
                if end['outcome'] == 'response':
                    if (not isinstance(raw, str) or not isinstance(end['judgment'], dict) or end['error'] is not None
                            or end['error_family'] is not None or end['retryable']):
                        raise ValueError('Invalid completed judge response')
                elif end['outcome'] == 'error':
                    if (end['judgment'] is not None or not isinstance(end['error'], str)
                            or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,127}', end['error']) is None
                            or end['error_family'] not in {'transport', 'semantic', 'other'}
                            or end['retryable'] and (end['error_family'] != 'transport' or raw is not None)):
                        raise ValueError('Invalid judge error event')
                else:
                    raise ValueError('Unknown judge attempt outcome')
                previous = end['event_sha256']
                terminal = end['outcome'] == 'response' or not end['retryable']
            else:
                terminal = True
            result.append({'begin': begin, 'end': end})
        if ends != {f'{index:03d}.end.json' for index, row in enumerate(result, 1) if row['end'] is not None}:
            raise ValueError('Orphan judge attempt end')
        return result

    def begin(self):
        rows = self.read()
        if rows:
            last = rows[-1]['end']
            if last is None or last['outcome'] != 'error' or not last['retryable']:
                raise ValueError('A terminal judge attempt cannot be repeated')
        if len(rows) >= self.maximum:
            raise ValueError('Judge attempt budget exhausted')
        if not self.directory.exists():
            create_directory(self.directory)
        number = len(rows)+1
        event = self._event(BEGIN_SCHEMA, attempt=number,
                            previous_end_sha256=rows[-1]['end']['event_sha256'] if rows else None)
        atomic_json(self.directory/f'{number:03d}.begin.json', event, replace=False)
        return event

    def end(self, begin, *, judgment=None, raw=None, error=None, error_family=None, retryable=False):
        rows = self.read()
        if not rows or rows[-1]['begin'] != begin or rows[-1]['end'] is not None:
            raise ValueError('Judge end has no matching unfinished attempt')
        value = self._event(END_SCHEMA, attempt=begin['attempt'], begin_sha256=begin['event_sha256'],
            outcome='response' if judgment is not None else 'error', judgment=judgment,
            raw_judge_answer=raw, response_sha256=text_digest(raw) if raw is not None else None,
            error=error, error_family=error_family, retryable=retryable)
        atomic_json(self.directory/f"{begin['attempt']:03d}.end.json", value, replace=False)
        return value


def retry_available(rows, maximum):
    return len(rows) < maximum and (not rows or rows[-1]['end'] is not None
            and rows[-1]['end']['outcome'] == 'error' and rows[-1]['end']['retryable'])
