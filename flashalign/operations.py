# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Read-only runtime discovery and explicitly requested synthetic checks."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

from owner_mode import __version__
from owner_mode.auth import read_api_key
from owner_mode.client import Client
from owner_mode.profiles import read_json
from owner_mode.terminal_ui import TerminalUI, clean_text


_PROBES = {}
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}')
_DIGEST = re.compile(r'[0-9a-f]{64}')


def register_probe(name, factory):
    """Registration is performed by trusted installed code, never metadata."""
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name):
        raise ValueError('Invalid probe adapter name')
    if name in _PROBES or not callable(factory):
        raise ValueError('Probe adapter must be unique and callable')
    _PROBES[name] = factory


def _label(value):
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate status field')
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError('Non-finite status value')


class OpenAIProbe:
    """Discovery adapter shared by compatible engines; no model-name dispatch."""
    def __init__(self, client):
        self.client = client
        self.headers = {}
        if client.api_key_file:
            self.headers['Authorization'] = 'Bearer ' + read_api_key(client.api_key_file)

    def get(self, suffix, *, text=False):
        try:
            with self.client.http.stream('GET', self.client.base_url + suffix,
                                         headers=self.headers, timeout=5.) as response:
                if response.status_code in (404, 405):
                    return None, 'unsupported'
                response.raise_for_status()
                parts, size = [], 0
                for part in response.iter_bytes():
                    size += len(part)
                    if size > 1024 * 1024:
                        return None, 'oversized_response'
                    parts.append(part)
                raw = b''.join(parts)
                value = raw.decode('utf-8') if text else json.loads(
                    raw, object_pairs_hook=_json_object, parse_constant=_invalid_number)
            if not text and not isinstance(value, dict):
                return None, 'invalid_response'
            return value, None
        except Exception as error:
            # Exception text may contain URLs, credentials or server content.
            return None, type(error).__name__

    def snapshot(self):
        models, error = self.get('/models')
        result = {'reachable': error is None, 'error': error,
                  'model_listed': None, 'profiles': None,
                  'runtime': None, 'capabilities': None, 'counters': {}}
        if models is not None:
            rows = models.get('data')
            if isinstance(rows, list):
                result['model_listed'] = any(isinstance(row, dict) and row.get('id') == self.client.model
                                             for row in rows[:1024])
        inventory, _ = self.get('/owner-mode/profiles')
        if inventory is not None and isinstance(inventory.get('data'), list):
            result['profiles'] = []
            for row in inventory['data'][:256]:
                if not isinstance(row, dict) or _label(row.get('name')) is None:
                    continue
                result['profiles'].append({'name': row['name'], 'loaded': row.get('loaded') is True})
        capabilities, _ = self.get('/flashalign/capabilities')
        if capabilities is not None and capabilities.get('schema') == 'flashalign.capabilities.v1':
            result['capabilities'] = {name: capabilities.get(name) is True
                                      for name in ('hot_loading', 'execution_attestation', 'status')}
        runtime, _ = self.get('/flashalign/status')
        if runtime is not None and runtime.get('schema') == 'flashalign.runtime-status.v1':
            result['runtime'] = self.runtime_fields(runtime)
        return result

    @staticmethod
    def runtime_fields(value):
        result = {'schema': 'flashalign.runtime-status.v1'}
        for key in ('engine', 'version', 'model', 'instance_id', 'generation', 'graph_mode', 'speculative_method'):
            result[key] = _label(value.get(key))
        for key in ('profile_digest', 'model_digest'):
            item = value.get(key)
            result[key] = item if isinstance(item, str) and _DIGEST.fullmatch(item) else None
        for key in ('expected_sites', 'installed_sites', 'exercised_sites', 'active_requests',
                    'gpu_memory_bytes', 'kv_cache_bytes', 'cache_hits', 'draft_proposed',
                    'draft_accepted', 'output_tokens_total', 'request_errors_total',
                    'ttft_seconds_sum', 'ttft_seconds_count'):
            result[key] = _number(value.get(key))
        for key in ('speculation_enabled', 'numerical_attestation'):
            result[key] = value.get(key) if type(value.get(key)) is bool else None
        return result


register_probe('openai', OpenAIProbe)


def snapshot(config, *, client_factory=Client.from_config, monotonic=time.monotonic):
    settings = read_json(config)
    adapter = settings.get('probe_adapter', 'openai')
    if adapter not in _PROBES:
        raise ValueError('Unknown installed probe adapter')
    with client_factory(config) as client:
        observed = _PROBES[adapter](client).snapshot()
    profile = settings.get('profile')
    inventory = observed['profiles']
    loaded = None if inventory is None or not profile else any(row['name'] == profile and row['loaded'] for row in inventory)
    return {'schema': 'flashalign.status.v1',
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'sample_time': monotonic(), 'model': _label(settings.get('model')),
            'configured_profile': _label(profile), 'mode': settings.get('mode', 'verified'),
            'profile_loaded': loaded, 'effective_profile': None, **observed}


def rates(previous, current):
    """Rates use measured counter deltas and never reinterpret missing values."""
    result = {'output_tokens_per_second': None, 'mean_ttft_seconds': None,
              'draft_acceptance': None}
    elapsed = current['sample_time'] - previous['sample_time']
    old, new = previous.get('runtime'), current.get('runtime')
    if elapsed <= 0 or not old or not new:
        return result
    if not old.get('instance_id') or not new.get('instance_id'):
        return result
    identity = ('instance_id', 'model', 'model_digest', 'generation')
    if any(old.get(k) != new.get(k) for k in identity):
        return result
    def delta(name):
        a, b = _number(old.get(name)), _number(new.get(name))
        return b - a if a is not None and b is not None and b >= a else None
    tokens = delta('output_tokens_total')
    if tokens is not None:
        result['output_tokens_per_second'] = tokens / elapsed
    ttft, count = delta('ttft_seconds_sum'), delta('ttft_seconds_count')
    if ttft is not None and count:
        result['mean_ttft_seconds'] = ttft / count
    proposed, accepted = delta('draft_proposed'), delta('draft_accepted')
    if proposed and accepted is not None and accepted <= proposed:
        result['draft_acceptance'] = accepted / proposed
    return result


def doctor(config, *, probe=False, client_factory=Client.from_config):
    value = snapshot(config, client_factory=client_factory)
    issues = []
    if not value['reachable']:
        issues.append({'code': 'engine_unreachable', 'action': 'Check the engine process and configured endpoint.'})
    elif value['model_listed'] is False:
        issues.append({'code': 'model_not_listed', 'action': 'Select a model advertised by this engine.'})
    if value['profile_loaded'] is False:
        issues.append({'code': 'profile_not_loaded', 'action': 'Load the compatible profile before selecting it.'})
    if probe and value['reachable']:
        settings = read_json(config)
        checks = []
        with client_factory(config) as client:
            profiles = list(dict.fromkeys([settings.get('profile')] if client.mode == 'always_on'
                                          else [None, settings.get('profile')]))
            for profile in profiles:
                if profile is None and client.mode == 'always_on':
                    continue
                try:
                    response = client.chat_response([{'role': 'user', 'content': 'Reply OK.'}],
                                                    profile=profile, max_tokens=1, temperature=0)
                    checks.append({'requested_profile': profile or 'off', 'confirmed': True,
                                   'effective_profile': response.metadata['effective_profile']})
                except Exception as error:
                    checks.append({'requested_profile': profile or 'off', 'confirmed': False,
                                   'error': type(error).__name__})
                    issues.append({'code': 'activation_probe_failed',
                                   'action': 'Check profile compatibility, signing configuration and engine status.'})
        value['activation_checks'] = checks
    value['issues'] = issues
    value['ok'] = not issues
    return value


def _display(value, ui):
    ui.context(model=value.get('model') or 'Unavailable',
               profile=value.get('configured_profile') or 'Not configured',
               access=value.get('mode', 'verified'))
    ui.line('ENGINE  ' + ('Reachable' if value.get('reachable') else 'Unavailable'))
    loaded = value.get('profile_loaded')
    ui.line('PROFILE  ' + ('Loaded' if loaded else 'Not loaded' if loaded is False else 'Inventory unavailable'))
    runtime = value.get('runtime') or {}
    for name, label in (('graph_mode', 'GRAPH MODE'), ('generation', 'GENERATION')):
        ui.line(label + '  ' + (runtime.get(name) or 'Unavailable'))
    for name, label, unit in (('expected_sites', 'EXPECTED SITES', ''),
                              ('installed_sites', 'INSTALLED SITES', ''),
                              ('exercised_sites', 'EXERCISED SITES', ''),
                              ('active_requests', 'ACTIVE REQUESTS', ''),
                              ('gpu_memory_bytes', 'GPU MEMORY', ' bytes')):
        number = runtime.get(name)
        ui.line(label + '  ' + (f'{number}{unit}' if number is not None else 'Unavailable'))
    spec = runtime.get('speculation_enabled')
    ui.line('SPECULATION  ' + ('Enabled' if spec else 'Disabled' if spec is False else 'Unavailable'))
    metrics = value.get('rates') or {}
    for name, label, unit in (('output_tokens_per_second', 'SERVER OUTPUT', 'tok/s'),
                              ('mean_ttft_seconds', 'MEAN TTFT', 's'),
                              ('draft_acceptance', 'DRAFT ACCEPTANCE', 'ratio')):
        number = metrics.get(name)
        ui.line(label + '  ' + (f'{number:.3f} {unit}' if number is not None else 'Unavailable'))
    for issue in value.get('issues', []):
        ui.result(clean_text(issue['code'] + ': ' + issue['action']), ok=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flashalign')
    parser.add_argument('command', choices=('status', 'doctor', 'dashboard'))
    parser.add_argument('--config', type=Path, default=Path.home()/'.config/flashalign/client.json')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--probe', action='store_true', help='Send synthetic activation requests (doctor only)')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args(argv)
    if args.probe and args.command != 'doctor':
        parser.error('--probe is only valid for doctor')
    if not math.isfinite(args.interval) or args.interval < .25 or (args.samples is not None and args.samples < 1):
        parser.error('Use a finite interval of at least 0.25 seconds and a positive sample count')
    ui = TerminalUI(color=not args.no_color, animation=False)
    if not args.json:
        ui.banner(__version__, subtitle=args.command.title())
    previous = None
    index = 0
    try:
        while True:
            value = doctor(args.config, probe=args.probe) if args.command == 'doctor' else snapshot(args.config)
            value['rates'] = rates(previous, value) if previous else rates(value, value)
            if args.json:
                # Internal monotonic timestamps do not belong in public output.
                print(json.dumps({k: v for k, v in value.items() if k != 'sample_time'}, allow_nan=False))
            else:
                if index:
                    ui.line()
                _display(value, ui)
            index += 1
            if args.command != 'dashboard' or args.samples is not None and index >= args.samples:
                return 1 if value.get('issues') else 0
            previous = value
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
