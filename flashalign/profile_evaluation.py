# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Evaluate a validated local profile through the existing paired protocol."""
import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys

import httpx

from . import Client
from .benchmark_io import atomic_json, canonical, loads, private_directory, read_bytes, read_json
from .benchmark_judge import SCORERS, judge_run, score_run, validate_judge_config
from .benchmark_protocol import (collect_calibration_sweep, collect_run, endpoint_fingerprint,
                                 validate_evaluation_manifest, validate_settings)
from .benchmark_reports import evidence_card, performance_summary, public_manifest
from .benchmark_suites import SUITES, load_prepared


def bind_profile(profile_path, manifest):
    """Bind recorded provenance to validated bytes, without importing model code."""
    from owner_mode.profiles import match_profile, profile_fingerprint, validate_profile
    checked = validate_profile(profile_path)
    if checked.get('artifact_valid') is not True:
        raise ValueError('Evaluation requires a complete validated profile payload')
    if not isinstance(manifest, dict) or not isinstance(manifest.get('model'), dict):
        raise ValueError('Evaluation needs pinned model provenance')
    matches, _ = match_profile(checked, manifest['model'])
    if not matches:
        raise ValueError('Manifest model or revision differs from the profile')
    manifest_bytes = read_bytes(checked['manifest'], limit=4*1024*1024)
    if hashlib.sha256(manifest_bytes).hexdigest() != checked['hashes']['profile.json']:
        raise ValueError('Profile manifest changed after validation')
    raw = loads(manifest_bytes)
    payload = checked.get('payload_file')
    if payload is None:
        payload = 'profile.safetensors' if 'profile.safetensors' in checked['hashes'] else None
    if payload is None:
        candidates = set(checked['hashes']) - {'profile.json'}
        if len(candidates) == 1: payload = candidates.pop()
    if payload not in checked['hashes']:
        raise ValueError('The artifact adapter must identify its primary payload')
    bound = deepcopy(manifest)
    target = bound.setdefault('profile', {})
    if not isinstance(target, dict):
        raise ValueError('Invalid manifest profile provenance')
    derived = {'name': checked['name'], 'artifact_sha256': checked['hashes'][payload],
               'content_digest': profile_fingerprint(checked)}
    if checked.get('format') is not None:
        derived['format'] = checked['format']
    elif 'format' in target:
        raise ValueError('Manifest format differs from the profile')
    strengths = raw.get('strengths')
    if strengths is not None:
        if (not isinstance(strengths, list) or not strengths
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1 for value in strengths)
                or len(set(strengths)) != len(strengths)):
            raise ValueError('Invalid artifact strength declaration')
        derived['allowed_strengths'] = strengths
    elif 'allowed_strengths' in target:
        raise ValueError('The profile artifact does not declare allowed strengths')
    for name, value in derived.items():
        if name in target and target[name] != value:
            raise ValueError('Manifest profile provenance differs from the validated artifact')
        target[name] = value
    validate_evaluation_manifest(bound)
    return checked, bound


def _preflight(prepared, config, manifest, settings, *, registry, purpose):
    suite, _, _ = load_prepared(prepared, registry=registry)
    if suite.purpose != purpose:
        raise ValueError('Use a registered '+purpose+' suite for this command')
    validate_settings(settings)
    if (not isinstance(config, dict) or config.get('mode', 'verified') != 'verified'
            or config.get('gateway', False)
            or config.get('model') != manifest['model'].get('served_id', manifest['model']['id'])):
        raise ValueError('Evaluation requires the manifest model on an isolated dual-mode endpoint')
    endpoint_fingerprint(config['base_url'])
    return suite


def _strength(manifest, scale):
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Evaluation strength must be in (0, 1]')
    allowed = manifest['profile'].get('allowed_strengths')
    if allowed is not None and scale not in allowed:
        raise ValueError('Requested strength is outside the artifact declaration')


def _publish(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Existing evaluation result differs from its validated receipts')
    else:
        atomic_json(path, value, replace=False)


def evaluate_profile(profile_path, *, prepared, config, manifest, settings, output, scale=1.,
                     judge_config=None, resume=False, registry=SUITES, scorers=SCORERS,
                     client_factory=Client, judge_client_factory=httpx.Client):
    _, bound = bind_profile(profile_path, manifest)
    suite = _preflight(prepared, config, bound, settings, registry=registry, purpose='evaluation')
    _strength(bound, scale)
    if judge_config is not None:
        scorer = scorers.get(suite.scorer)
        validate_judge_config(judge_config, endpoint_fingerprint(config['base_url']), scorer)
        public_manifest(bound)  # Reject unsafe export metadata before collecting.
    root = private_directory(output, create=not resume)
    collection = collect_run(prepared, config, bound, settings, root/'collection', scale=scale,
                             resume=resume, registry=registry, client_factory=client_factory)
    result = {'schema': 'flashalign.profile-evaluation.v1', 'collection_only': judge_config is None,
              'collection_valid': collection['collection_valid'], 'collection': collection,
              'behavioral_score': 'not_run', 'evidence_card_written': False}
    if judge_config is None:
        return result
    if not collection['collection_valid']:
        result['behavioral_score'] = 'not_run_incomplete_collection'
        return result
    judgments = root/'judgments'
    judge_run(root/'collection', judge_config, judgments, resume=judgments.exists(), registry=registry,
              scorers=scorers, client_factory=judge_client_factory)
    report = score_run(root/'collection', judgments, registry=registry, scorers=scorers)
    performance = performance_summary(root/'collection', registry=registry)
    _publish(root/'report.json', {**report, 'performance': performance})
    valid = report['final_behavioral_gate_valid']
    result.update(behavioral_score='complete' if valid else 'incomplete', final_behavioral_gate_valid=valid)
    if valid:
        _publish(root/'evidence-card.json', evidence_card(report, performance))
        result['evidence_card_written'] = True
    return result


def sweep_profile(profile_path, *, prepared, config, manifest, settings, output, strengths,
                  registry=SUITES, client_factory=Client):
    _, bound = bind_profile(profile_path, manifest)
    _preflight(prepared, config, bound, settings, registry=registry, purpose='calibration')
    if ('allowed_strengths' not in bound['profile'] or not isinstance(strengths, list)
            or not 1 <= len(strengths) <= 32):
        raise ValueError('Calibration requires up to 32 artifact-declared candidate strengths')
    for value in strengths: _strength(bound, value)
    if len(set(strengths)) != len(strengths):
        raise ValueError('Calibration strengths must be unique')
    return collect_calibration_sweep(prepared, config, bound, settings, output, strengths=strengths,
                                      registry=registry, client_factory=client_factory)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flashalign profile', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'sweep'):
        command = commands.add_parser(name)
        command.add_argument('profile', type=Path)
        for field in ('prepared', 'config', 'manifest', 'settings', 'output'):
            command.add_argument('--'+field, type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--scale', type=float, default=1.)
            command.add_argument('--judge-config', type=Path)
            command.add_argument('--resume', action='store_true')
        else:
            command.add_argument('--strengths', type=float, nargs='+', required=True)
    args = parser.parse_args(argv)
    try:
        options = {name: read_json(getattr(args, name)) for name in ('config', 'manifest', 'settings')}
        options.update(prepared=args.prepared, output=args.output)
        if args.command == 'evaluate':
            result = evaluate_profile(args.profile, **options, scale=args.scale, resume=args.resume,
                                      judge_config=read_json(args.judge_config) if args.judge_config else None)
            complete = result['collection_valid'] and (result['collection_only'] or result.get('final_behavioral_gate_valid', False))
        else:
            result = sweep_profile(args.profile, **options, strengths=args.strengths)
            complete = all(row['collection']['collection_valid'] for row in result['runs'])
        print(canonical(result))
        return 0 if complete else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('Profile evaluation stopped: '+type(error).__name__+'. Inspect private inputs and receipts.', file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
