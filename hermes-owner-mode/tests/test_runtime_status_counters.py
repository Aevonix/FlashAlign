# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU-only status collection and display, with an in-process metrics fixture."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_setup import wizard
from owner_mode import profile_admin_server
from flashalign.operations import OpenAIProbe, _display, rates

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('output_tokens_total', 'ttft_seconds_sum', 'ttft_seconds_count')


def registry(engine, values):
    names = ('generation_tokens_total', 'time_to_first_token_seconds_sum', 'time_to_first_token_seconds_count')
    samples = [SimpleNamespace(name=engine + ':' + name, value=value)
               for name, group in zip(names, values) for value in group]
    samples.append(SimpleNamespace(name=engine + ':time_to_first_token_seconds_bucket', value=999))
    return SimpleNamespace(REGISTRY=SimpleNamespace(collect=lambda: [SimpleNamespace(samples=samples)]))


def load_package(engine, package):
    name = 'status_fixture_' + engine.replace('-', '_')
    directory = ROOT/'engines'/engine/package
    spec = importlib.util.spec_from_file_location(name, directory/'__init__.py',
                                                 submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def immediate(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError('The status fixture must not wait for external I/O')


class RuntimeCounterTests(unittest.TestCase):
    def test_existing_metrics_are_summed_and_missing_or_invalid_are_unavailable(self):
        for engine in ('vllm', 'sglang'):
            with self.subTest(engine=engine):
                with patch.dict(sys.modules, {'prometheus_client': registry(engine, ([10, 20], [.25, .5], [1, 2]))}):
                    self.assertEqual(profile_admin_server.runtime_counters(engine),
                                     dict(zip(FIELDS, (30, .75, 3))))
                for values, expected in (
                        (([], [], []), (None, None, None)),
                        (([0], [0.], [0]), (0, 0., 0)),
                        (([-1], [float('nan')], [1]), (None, None, None)),
                        (([1.5], [1.], [2.5]), (None, None, None)),
                        (([True], [1.], []), (None, None, None))):
                    with patch.dict(sys.modules, {'prometheus_client': registry(engine, values)}):
                        self.assertEqual(profile_admin_server.runtime_counters(engine), dict(zip(FIELDS, expected)))
        with patch.dict(sys.modules, {'prometheus_client': None}):
            self.assertEqual(profile_admin_server.runtime_counters('vllm'), dict.fromkeys(FIELDS))

    def test_vllm_status_carries_measured_counters_into_rates(self):
        for engine, package in (('vllm', 'flashalign_vllm'), ('vllm-gemma4', 'flashalign_vllm'),
                                ('vllm-deepseek-native', 'flashalign_native')):
            with self.subTest(engine=engine):
                with patch.object(sys, 'path', [str(ROOT/'engines'/engine), *sys.path]):
                    module = importlib.import_module(load_package(engine, package) + '.lifecycle_service')
                service = module.LifecycleService.__new__(module.LifecycleService)
                service.manager = SimpleNamespace(status=lambda: {'aliases': {}, 'generations': []})
                service.backend = SimpleNamespace(worker_ids=['worker-0'], _call=lambda op: [{'worker_id': 'worker-0'}])
                service.identity, service.instance_id = ('fixture/model', 'r1'), 'fixture-instance'
                snapshots = []
                for time, values in ((1, ([10], [.25], [1])), (3, ([30], [.75], [3]))):
                    with patch.dict(sys.modules, {'prometheus_client': registry('vllm', values)}):
                        snapshots.append({'sample_time': time, 'runtime': OpenAIProbe.runtime_fields(service.public_status())})
                self.assertEqual(rates(*snapshots)['output_tokens_per_second'], 10)
                self.assertEqual(rates(*snapshots)['mean_ttft_seconds'], .25)

    def test_sglang_status_uses_one_worker_for_tokens_and_api_for_ttft(self):
        for engine in ('sglang', 'sglang-gemma4', 'sglang-qwen4', 'sglang-glm53-flash'):
            with self.subTest(engine=engine):
                name = load_package(engine, 'flashalign_sglang')
                api = importlib.import_module(name + '.lifecycle_api')
                worker = importlib.import_module(name + '.lifecycle_worker')
                runtime = ModuleType(name + '.runtime')
                runtime.registry = lambda: ({}, None)
                scheduler = SimpleNamespace(is_fully_idle=lambda: True)
                with patch.dict(sys.modules, {name + '.runtime': runtime,
                        'prometheus_client': registry('sglang', ([40], [], []))}):
                    detail = worker._apply(scheduler, {'action': 'inspect', 'transaction': {'id': 't-' + 'a'*48}})
                self.assertEqual(detail['output_tokens_total'], 40)
                backend = SimpleNamespace(models=[json.dumps({'id': 'fixture/model'})],
                    tokenizer=SimpleNamespace(server_args=SimpleNamespace()),
                    call=lambda op: [{**detail, 'worker_id': 'tp-0'}, {**detail, 'worker_id': 'tp-1'}])
                api._CONTROLLER = SimpleNamespace(backend=backend,
                    manager=SimpleNamespace(status=lambda: {'aliases': {}, 'generations': []}))
                async def in_process(function, *args):
                    return function(*args)
                with (patch.object(api.asyncio, 'to_thread', in_process),
                      patch.dict(sys.modules, {'prometheus_client': registry('sglang', ([], [1.5], [3]))})):
                    result = immediate(api.public_status())
                self.assertEqual({key: result[key] for key in FIELDS}, dict(zip(FIELDS, (40, 1.5, 3))))

    def test_llamacpp_explicitly_reports_unavailable_counters(self):
        path = ROOT/'engines/llamacpp'
        with patch.object(sys, 'path', [str(path), *sys.path]):
            spec = importlib.util.spec_from_file_location('status_fixture_llamacpp', path/'lifecycle.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        service = module.NativeLifecycle.__new__(module.NativeLifecycle)
        service.manager = SimpleNamespace(status=lambda: {'aliases': {}})
        service.backend = SimpleNamespace(profile_model={'id': 'fixture/model'})
        service.instance_id, service._active_requests = 'fixture', 0
        import threading
        service._requests_lock = threading.Lock()
        result = service.safe_status()
        self.assertEqual({key: result[key] for key in FIELDS}, dict.fromkeys(FIELDS))

    def test_dashboard_displays_reported_state_and_removes_attestation(self):
        ui = Mock()
        _display({'runtime': {'graph_mode': 'full', 'generation': 'generation-one', 'expected_sites': 4,
                             'installed_sites': 4, 'exercised_sites': 0, 'active_requests': 0,
                             'gpu_memory_bytes': 4096, 'numerical_attestation': True}}, ui)
        lines = [call.args[0] for call in ui.line.call_args_list]
        for line in ('GRAPH MODE  full', 'GENERATION  generation-one', 'EXPECTED SITES  4',
                     'INSTALLED SITES  4', 'EXERCISED SITES  0', 'ACTIVE REQUESTS  0', 'GPU MEMORY  4096 bytes'):
            self.assertIn(line, lines)
        self.assertFalse(any('attest' in line.lower() or line.startswith('EXECUTION') for line in lines))
        ui.reset_mock()
        _display({}, ui)
        self.assertIn('GPU MEMORY  Unavailable', [call.args[0] for call in ui.line.call_args_list])
