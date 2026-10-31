# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Inspect local profile files without a model, GPU, network or deployment."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        'owner_mode', root/'__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

from owner_mode.profiles import ProfileWorkLimit, validate_profile
from owner_mode.terminal_ui import clean_text


DEFAULT_BASIS_WORK = 50_000_000


def inspect_profile(path, *, max_basis_work=DEFAULT_BASIS_WORK):
    checked = validate_profile(path, include_audit=True, max_basis_work=max_basis_work)
    audit = checked['audit']
    basis = [item['basis'] for item in audit['tensors'] if 'basis' in item]
    model = {field: checked['model'][field] for field in ('id', 'revision')
             if field in checked['model']}
    if isinstance(checked['model'].get('format'), str):
        model['format'] = checked['model']['format']
    compatible = [{field: row[field] for field in ('id', 'revision', 'model_type') if field in row}
                  for row in checked['compatible_models']]
    checksum_count = len(audit['supplied_checksums']['verified_files'])
    return {'schema': 'flashalign.profile-inspection.v1', 'artifact_valid': True,
            'inspection_complete': True,
            'name': checked['name'], 'model': model, 'compatible_models': compatible,
            'transform': checked['transform'], 'profile_schema': checked['schema'] or 'standard',
            'requires': checked['requires'],
            'hashes': checked['hashes'], **audit,
            'basis_count': len(basis), 'basis_rows': sum(row['rank'] for row in basis),
            'maximum_rank': max(row['rank'] for row in basis),
            'assessment': {'file_structure': 'passed', 'tensor_values': 'passed',
                           'supplied_checksums': ('verified' if checksum_count == 2 else
                                                  'partial' if checksum_count else 'not_supplied'),
                           'runtime_compatibility': 'not_assessed',
                           'model_quality': 'not_assessed', 'semantic_safety': 'not_assessed'}}


def _display(value):
    return ' '.join(clean_text(value).splitlines())[:300]


def render(report, details=False):
    if not report['artifact_valid']:
        return 'FlashAlign profile inspection incomplete: ' + _display(report['error'])
    model = report['model']
    rows = ['FlashAlign profile inspection',
            'Profile: ' + _display(report['name']),
            'Model: ' + _display(model['id']),
            'Revision: ' + _display(model['revision']),
            'Transform: ' + _display(report['transform']),
            f"Sites: {report['site_count']} | Bases: {report['basis_count']} | Maximum rank: {report['maximum_rank']}",
            f"Hidden width: {report['hidden_size']} | Tensor file: {report['tensor_file_bytes']:,} bytes",
            'Requires: ' + (', '.join(_display(value) for value in report['requires']) or 'standard projection'),
            'File and numerical validation: passed',
            'Supplied checksums: ' + report['assessment']['supplied_checksums'].replace('_', ' '),
            'Runtime compatibility, model quality and semantic safety: not assessed',
            'SHA-256 profile.json: ' + report['hashes']['profile.json'],
            'SHA-256 profile.safetensors: ' + report['hashes']['profile.safetensors']]
    if details:
        rows.append('Sites:')
        for site in report['sites']:
            operator = report['operators'].get(site.get('operator'), {})
            rank = site.get('rank', operator.get('rank'))
            strength = site.get('scale', operator.get('scale'))
            target = site.get('tensor', site.get('tensor_pattern', site.get('source_tensors', site.get('tensors', ''))))
            rows.append(f"  L{site['layer']} {_display(site['site'])}: rank {rank}, scale {strength}; {_display(target)}")
        rows.append('Tensor checks:')
        for tensor in report['tensors']:
            if 'basis' in tensor:
                basis = tensor['basis']
                rows.append(f"  {_display(tensor['name'])}: row norms {basis['row_norm_min']:.7g}..{basis['row_norm_max']:.7g}, max cross-dot {basis['max_abs_cross_dot']:.3g}")
            else:
                value = tensor['range']
                rows.append(f"  {_display(tensor['name'])}: range {value['minimum']:.7g}..{value['maximum']:.7g}, L2 norm {tensor['l2_norm']:.7g}")
    return '\n'.join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flashalign-profile', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('inspect', help='Validate and describe a local profile')
    command.add_argument('profile', type=Path, help='Profile directory or profile.json file')
    command.add_argument('--json', action='store_true', help='Write the full structured inspection')
    command.add_argument('--details', action='store_true', help='Include site and tensor checks in terminal output')
    command.add_argument('--max-basis-work', type=int, default=DEFAULT_BASIS_WORK,
                         help='Maximum scalar products for basis checks (default: 50000000)')
    args = parser.parse_args(argv)
    try:
        report = inspect_profile(args.profile, max_basis_work=args.max_basis_work)
    except ProfileWorkLimit as error:
        report = {'schema': 'flashalign.profile-inspection.v1', 'artifact_valid': None,
                  'inspection_complete': False, 'error_type': 'work_limit', 'error': str(error)}
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
        report = {'schema': 'flashalign.profile-inspection.v1', 'artifact_valid': False,
                  'inspection_complete': False, 'error_type': 'invalid_artifact', 'error': str(error)}
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) if args.json else render(report, args.details))
    return 0 if report['artifact_valid'] else 3 if report.get('error_type') == 'work_limit' else 2


if __name__ == '__main__':
    raise SystemExit(main())
