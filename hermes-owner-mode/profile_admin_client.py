# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Owner-credentialed Unix profile-control client shared by setup and the CLI."""
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import shutil
import httpx
from .profile_staging import stage_validated_profile


class LifecycleRequestError(RuntimeError):
    pass


class AdminClient:
    def __init__(self, socket_path, *, timeout=300., transport=None):
        self.path = Path(socket_path).expanduser()
        info = self.path.lstat()
        parent = self.path.parent.lstat()
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
                or stat.S_IMODE(parent.st_mode) != 0o700):
            raise ValueError('Use an owned mode-0600 administration socket in a private directory')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError('Invalid administration timeout')
        self.http = httpx.Client(base_url='http://localhost', timeout=timeout,
                                 transport=transport or httpx.HTTPTransport(uds=str(self.path)))

    def request(self, action=None, **fields):
        with self.http.stream('GET' if action is None else 'POST',
                              '/status' if action is None else '/operations',
                              json=None if action is None else {'action': action, **fields}) as response:
            parts, size = [], 0
            for part in response.iter_bytes():
                size += len(part)
                if size > 1024*1024:
                    raise LifecycleRequestError('Administration response exceeds the size limit')
                parts.append(part)
            try:
                value = json.loads(b''.join(parts))
            except (ValueError, UnicodeError):
                raise LifecycleRequestError('Invalid administration response') from None
            if not isinstance(value, dict) or type(value.get('ok')) is not bool:
                raise LifecycleRequestError('Invalid administration response envelope')
            if response.status_code != 200 or not value['ok']:
                error = value.get('error')
                code = error.get('code') if isinstance(error, dict) else None
                if not isinstance(code, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}', code):
                    code = 'AdministrationFailed'
                raise LifecycleRequestError('Profile operation failed: ' + code)
            return value.get('result')

    def close(self):
        self.http.close()

    def stage(self, source, *, staging_directory=None, remote=None):
        """Snapshot input into a configured shared mount for container backends."""
        source = Path(source).expanduser().absolute()
        if remote is not None:
            if staging_directory is not None:
                raise ValueError('Choose local or remote profile staging')
            from .profile_staging import remote_profile_stage
            with remote_profile_stage(source, remote) as staged:
                return self.request('stage', path=staged)
        if staging_directory is None:
            return self.request('stage', path=str(source))
        root = Path(staging_directory).expanduser()
        info = root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError('Profile staging requires an owned mode-0700 directory')
        temporary = Path(tempfile.mkdtemp(prefix='.flashalign-stage-', dir=root.absolute()))
        try:
            stage_validated_profile(source, temporary)
            return self.request('stage', path=str(temporary))
        finally:
            # This name was freshly created by this call under the checked root.
            shutil.rmtree(temporary)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
