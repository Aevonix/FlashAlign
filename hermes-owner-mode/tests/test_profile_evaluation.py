# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Profile convenience commands preserve strict collection and scoring contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest import mock

import test_evaluation_protocol as fixtures
from test_native_profile import native_fixture
from test_setup import make_profile
from flashalign import __main__ as cli
from flashalign.benchmark_io import canonical, read_json
from flashalign.benchmark_suites import Suite, SuiteRegistry, prepare_suite
from flashalign.profile_evaluation import bind_profile, evaluate_profile, main, sweep_profile
from owner_mode.profiles import profile_fingerprint, validate_profile


class ProfileEvaluationTests(unittest.TestCase):
    client = fixtures.EvaluationTests.client
    judge_reply = fixtures.EvaluationTests.judge_reply
    judge_client = fixtures.EvaluationTests.judge_client

    def setUp(self):
        fixtures.EvaluationTests.setUp(self)
        self.profile = make_profile(self.root/'profile', model='org/model', name='profile')
        metadata = read_json(self.profile/'profile.json')
        metadata['model']['revision'] = 'fixed'
        (self.profile/'profile.json').write_text(json.dumps(metadata))
        self.manifest = deepcopy(fixtures.MANIFEST)
        self.manifest['profile'] = {}
        self.runtime_digest = profile_fingerprint(validate_profile(self.profile))

    def respond(self, request):
        response = fixtures.EvaluationTests.respond(self, request)
        response.headers['X-FlashAlign-Profile-Digest'] = self.runtime_digest
        return response

    def evaluate(self, *, output='evaluation', **overrides):
        values = dict(prepared=self.prepared, config=self.config, manifest=self.manifest,
                      settings=self.options, output=self.root/output, registry=self.registry,
                      client_factory=self.client, judge_client_factory=self.judge_client)
        values.update(overrides)
        return evaluate_profile(self.profile, **values)

    def calibration(self, *, strengths=True):
        if strengths:
            metadata = read_json(self.profile/'profile.json')
            metadata['strengths'] = [.5, 1.]
            (self.profile/'profile.json').write_text(json.dumps(metadata))
            self.runtime_digest = profile_fingerprint(validate_profile(self.profile))
        suite = Suite('calibration', 'calibration-v1', 'fixed', self.suite.source_sha256,
                      self.suite.row_count, self.suite.datasets, self.suite.labels,
                      self.suite.unique_prompts, self.suite.scorer, self.suite.source_url,
                      purpose='calibration')
        registry = SuiteRegistry((suite,))
        prepared = prepare_suite('calibration', self.source, self.root/'calibration', registry=registry)
        return prepared, registry

    def test_collection_only_binds_validated_bytes_without_a_judge_or_card(self):
        result = self.evaluate()
        self.assertTrue(result['collection_only'])
        self.assertTrue(result['collection_valid'])
        self.assertEqual(result['behavioral_score'], 'not_run')
        self.assertFalse(result['evidence_card_written'])
        run = read_json(self.root/'evaluation/collection/run.json')
        recorded = run['descriptor']['manifest']['profile']
        self.assertEqual(recorded['content_digest'], self.runtime_digest)
        self.assertEqual(recorded['name'], 'profile')
        self.assertEqual(recorded['artifact_sha256'], validate_profile(self.profile)['hashes']['profile.safetensors'])
        self.assertFalse((self.root/'evaluation/judgments').exists())
        self.assertFalse((self.root/'evaluation/evidence-card.json').exists())
        self.assertFalse(self.judge_requests)
        self.assertEqual(self.manifest['profile'], {})
        self.assertEqual((self.root/'evaluation').stat().st_mode & 0o777, 0o700)

    def test_complete_scored_run_writes_only_a_verified_public_shape_card(self):
        result = self.evaluate(judge_config=self.judge_config)
        self.assertFalse(result['collection_only'])
        self.assertTrue(result['final_behavioral_gate_valid'])
        self.assertTrue(result['evidence_card_written'])
        card = read_json(self.root/'evaluation/evidence-card.json')
        self.assertTrue(card['evaluation']['final_behavioral_gate_valid'])
        self.assertEqual(card['profile']['content_digest'], self.runtime_digest)
        rendered = canonical(card)
        for value in ('synthetic final answer', 'A synthetic harmless prompt', 'subject.test', 'judge.test', str(self.key)):
            self.assertNotIn(value, rendered)
        self.evaluate(judge_config=self.judge_config, resume=True)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(self.judge_requests), 4)

    def test_missing_runtime_digest_never_produces_a_behavioral_card(self):
        self.runtime_digest = 'b'*64
        result = self.evaluate(judge_config=self.judge_config)
        self.assertFalse(result['collection_valid'])
        self.assertEqual(result['behavioral_score'], 'not_run_incomplete_collection')
        self.assertFalse(result['evidence_card_written'])
        self.assertFalse(self.judge_requests)

    def test_uncertain_judgment_keeps_private_receipts_and_omits_card(self):
        self.judge_verdict = 'uncertain'
        result = self.evaluate(judge_config=self.judge_config)
        self.assertFalse(result['final_behavioral_gate_valid'])
        self.assertEqual(result['behavioral_score'], 'incomplete')
        self.assertTrue((self.root/'evaluation/report.json').exists())
        self.assertTrue((self.root/'evaluation/judgments/judge.json').exists())
        self.assertFalse((self.root/'evaluation/evidence-card.json').exists())

    def test_manifest_mismatches_fail_before_requests(self):
        mutations = [lambda m: m['model'].update(id='other/model'),
                     lambda m: m['model'].update(revision='other-revision'),
                     lambda m: m['profile'].update(name='different-profile'),
                     lambda m: m['profile'].update(artifact_sha256='a'*64),
                     lambda m: m['profile'].update(content_digest='a'*64),
                     lambda m: m['profile'].update(format='unknown-format'),
                     lambda m: m['profile'].update(allowed_strengths=[.5, 1.])]
        for index, mutate in enumerate(mutations):
            manifest = deepcopy(self.manifest); mutate(manifest)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.evaluate(output='bad-'+str(index), manifest=manifest)
            self.assertFalse((self.root/('bad-'+str(index))).exists())
        self.assertFalse(self.requests)

    def test_invalid_profile_and_native_fractional_strength_fail_before_requests(self):
        native = self.root/'native'
        document, _ = native_fixture(native)
        manifest = deepcopy(self.manifest)
        manifest['model'].update(id='fixture/model', revision='1'*40)
        _, bound = bind_profile(native, manifest)
        self.assertEqual(bound['profile']['allowed_strengths'], [1.0])
        self.assertEqual(bound['profile']['artifact_sha256'], document['payload']['sha256'])
        with self.assertRaisesRegex(ValueError, 'strength'):
            evaluate_profile(native, prepared=self.prepared, config=self.config, manifest=manifest,
                settings=self.options, output=self.root/'bad-native', scale=.5, registry=self.registry,
                client_factory=self.client)
        (self.profile/'profile.safetensors').write_bytes(b'invalid')
        with self.assertRaises(ValueError): self.evaluate(output='bad-payload')
        self.assertFalse(self.requests)

    def test_bad_judge_configuration_fails_before_subject_requests(self):
        with self.assertRaises(ValueError):
            self.evaluate(judge_config={**self.judge_config, 'base_url': self.config['base_url']})
        self.assertFalse(self.requests)

    def test_calibration_sweep_uses_only_declared_strengths_without_a_winner(self):
        prepared, registry = self.calibration()
        result = sweep_profile(self.profile, prepared=prepared, config=self.config, manifest=self.manifest,
            settings=self.options, output=self.root/'sweep', strengths=[.5, 1.], registry=registry,
            client_factory=self.client)
        self.assertIsNone(result['selection'])
        self.assertEqual([row['strength'] for row in result['runs']], [.5, 1.])
        self.assertEqual(len(self.requests), 8)
        self.assertFalse(self.judge_requests)

    def test_sweep_rejects_evaluation_data_and_undeclared_or_excessive_candidates(self):
        with self.assertRaises(ValueError):
            sweep_profile(self.profile, prepared=self.prepared, config=self.config, manifest=self.manifest,
                settings=self.options, output=self.root/'forbidden', strengths=[.5, 1.],
                registry=self.registry, client_factory=self.client)
        prepared, registry = self.calibration(strengths=False)
        with self.assertRaises(ValueError):
            sweep_profile(self.profile, prepared=prepared, config=self.config, manifest=self.manifest,
                settings=self.options, output=self.root/'undeclared', strengths=[.5, 1.],
                registry=registry, client_factory=self.client)
        metadata = read_json(self.profile/'profile.json'); metadata['strengths'] = [.5, 1.]
        (self.profile/'profile.json').write_text(json.dumps(metadata))
        for strengths in ([.5, .75], [.5, .5], [0., .5], [float('nan')], [.5]*33):
            with self.subTest(strengths=strengths), self.assertRaises(ValueError):
                sweep_profile(self.profile, prepared=prepared, config=self.config, manifest=self.manifest,
                    settings=self.options, output=self.root/'invalid-sweep', strengths=strengths,
                    registry=registry, client_factory=self.client)
        metadata['strengths'] = [index/100 for index in range(1, 101)]
        (self.profile/'profile.json').write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            sweep_profile(self.profile, prepared=prepared, config=self.config, manifest=self.manifest,
                settings=self.options, output=self.root/'too-many', strengths=metadata['strengths'][:33],
                registry=registry, client_factory=self.client)
        self.assertFalse(self.requests)

    def test_cli_dispatch_routes_evaluate_and_sweep(self):
        with mock.patch('flashalign.profile_evaluation.main', return_value=7) as entry:
            self.assertEqual(cli.main(['profile', 'evaluate', '/profile', '--prepared', '/corpus']), 7)
            entry.assert_called_with(['evaluate', '/profile', '--prepared', '/corpus'])
            self.assertEqual(cli.main(['profile', 'sweep', '/profile']), 7)
            entry.assert_called_with(['sweep', '/profile'])

    def test_cli_collection_only_output_is_explicit(self):
        for name, value in (('config', self.config), ('manifest', self.manifest), ('settings', self.options)):
            # The direct config fixture contains a Path key location.
            (self.root/(name+'.json')).write_text(json.dumps(value, default=str))
        out = io.StringIO()
        result = {'collection_only': True, 'collection_valid': True, 'behavioral_score': 'not_run',
                  'evidence_card_written': False}
        args = ['evaluate', str(self.profile)]
        for name in ('prepared', 'config', 'manifest', 'settings', 'output'):
            value = self.prepared if name == 'prepared' else self.root/'cli' if name == 'output' else self.root/(name+'.json')
            args += ['--'+name, str(value)]
        with mock.patch('flashalign.profile_evaluation.evaluate_profile', return_value=result), redirect_stdout(out):
            self.assertEqual(main(args), 0)
        self.assertEqual(json.loads(out.getvalue()), result)


if __name__ == '__main__': unittest.main()
