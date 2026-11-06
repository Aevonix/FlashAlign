# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify owned Docker resources after the managed launcher has been reaped."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from engine_config import IMAGE
from supervisor import private_directory, read_receipt, stop_owned


SCHEMA = 'flashalign.external-cleanup.v1'


def cleanup(state, deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError('External cleanup deadline expired')
    state = private_directory(state)
    deployment = json.loads(read_receipt(state/'deployment.json'))
    if (not isinstance(deployment, dict) or deployment.get('container_runtime') != 'docker'
            or deployment.get('image') != IMAGE):
        raise ValueError('Deployment does not use the pinned native image')
    runtime = state/'native-runtime'
    if not runtime.exists():
        return {'schema': SCHEMA, 'status': 'stopped', 'external_resources_stopped': True, 'resource_count': 0}
    private_directory(runtime)
    runs = runtime/'runs'
    if not runs.exists():
        return {'schema': SCHEMA, 'status': 'stopped', 'external_resources_stopped': True, 'resource_count': 0}
    private_directory(runs)
    count = 0
    for run in sorted(runs.iterdir()):
        if re.fullmatch(r'[0-9a-f]{24}', run.name) is None:
            raise ValueError('Unrecognized native container run')
        private_directory(run)
        intent = json.loads(read_receipt(run/'container-intent.json'))
        labels = {'io.aevonix.flashalign.deployment': hashlib.sha256(str(runtime).encode()).hexdigest(),
                  'io.aevonix.flashalign.run': run.name}
        if (not isinstance(intent, dict) or intent.get('schema') != 'flashalign.container-intent.v1'
                or intent.get('container_name') != 'flashalign-native-'+run.name
                or intent.get('image_id') != deployment.get('image_id')
                or intent.get('labels') != labels):
            raise ValueError('Container creation intent is not bound to this deployment')
        # A process may die after Docker receives create but before its ID is
        # recorded. An unresolved creation intent is never proof of absence.
        record = json.loads(read_receipt(run/'container.json'))
        fields = ('image_id', 'labels', 'daemon_id', 'parent_pid', 'parent_start',
                  'started_unix', 'profile_digest')
        if not isinstance(record, dict) or any(record.get(key) != intent.get(key) for key in fields):
            raise ValueError('Container record differs from its creation intent')
        stop_owned(record, deadline=deadline)
        count += 1
    return {'schema': SCHEMA, 'status': 'stopped', 'external_resources_stopped': True, 'resource_count': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--deadline-monotonic', type=float, required=True)
    args = parser.parse_args()
    try:
        result = cleanup(args.state_dir, args.deadline_monotonic)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        result = {'schema': SCHEMA, 'status': 'incomplete', 'external_resources_stopped': False}
    print(json.dumps(result))
    return 0 if result['external_resources_stopped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
