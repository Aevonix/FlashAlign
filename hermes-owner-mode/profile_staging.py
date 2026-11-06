# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Strict artifact staging for the generic profile lifecycle coordinator."""
from pathlib import Path
from contextlib import contextmanager
import secrets
import tempfile

from .profile_lifecycle import Artifact, LifecycleError
from .profiles import profile_fingerprint, stage_profile


def stage_validated_profile(source, destination):
    """Copy and revalidate complete profile bytes into an empty private slot.

    ``stage_profile`` checks the source, copies only declared artifact files
    and notices, validates the copied snapshot and verifies equal hashes. Its
    native reader streams the large payload rather than loading it whole.
    """
    destination = Path(destination)
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise LifecycleError("Profile staging requires an empty owned directory")
    destination.rmdir()
    checked = stage_profile(source, destination)
    return Artifact(checked["name"], profile_fingerprint(checked),
                    checked["model"]["id"], checked["model"]["revision"],
                    checked["transform"], destination,
                    tuple(sorted(checked["requires"])))


@contextmanager
def remote_profile_stage(source, target):
    """Transfer an immutable local snapshot through an operator's configured SSH host."""
    from . import native_cluster
    if (not isinstance(target, dict) or set(target) - {'host', 'staging_directory', 'ssh_config'}
            or not isinstance(target.get('host'), str) or not target['host'] or target['host'] == 'local'
            or not isinstance(target.get('staging_directory'), str)
            or not Path(target['staging_directory']).is_absolute()):
        raise LifecycleError('Invalid remote administration staging configuration')
    options = {'ssh_config': target.get('ssh_config')}
    identifier = secrets.token_hex(10)
    parent = target['staging_directory']
    destination = Path(parent)/identifier
    with tempfile.TemporaryDirectory(prefix='flashalign-upload-') as temporary:
        snapshot = Path(temporary)/'profile'
        snapshot.mkdir(mode=0o700)
        stage_validated_profile(source, snapshot)
        files = {'profile/'+path.name: (path, False) for path in snapshot.iterdir() if path.is_file()}
        receipt = native_cluster.stage_files(target['host'], files, parent=parent, id=identifier, **options)
        if receipt.get('state') != str(destination):
            raise LifecycleError('Remote profile staging returned an unexpected destination')
        try:
            yield str(destination/'profile')
        finally:
            result = native_cluster.rpc(target['host'], 'discard_artifact', parent=parent, id=identifier, **options)
            if result.get('discarded') is not True:
                raise LifecycleError('Remote temporary profile cleanup is incomplete')
