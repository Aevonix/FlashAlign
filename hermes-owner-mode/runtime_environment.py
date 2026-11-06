# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Environment settings for commands run by a selected Python interpreter."""
import os
from pathlib import Path
import shutil

from .setup_runtime import SetupError


def interpreter_path(python, *, path=None):
    """Locate an interpreter without following its virtual-environment symlink."""
    executable = os.fspath(python)
    if not os.path.dirname(executable):
        executable = shutil.which(executable, path=path)
        if executable is None:
            return None
    return os.path.abspath(os.path.expanduser(executable))


def python_environment(python, environment=None, *, inherited=None):
    """Find interpreter-local tools while preserving the configured PATH.

    Keep the interpreter path lexical: a virtual environment's Python can be a
    symlink to a system interpreter, while its console scripts live beside the
    symlink. An explicit PATH, including an empty one, takes precedence over the
    inherited search path after this interpreter's directory.
    """
    if environment is not None and not isinstance(environment, dict):
        raise SetupError('Engine environment must contain string values')
    result = dict(environment or {})
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in result.items()):
        raise SetupError('Engine environment must contain string values')
    inherited = os.environ if inherited is None else inherited
    remainder = result['PATH'] if 'PATH' in result else inherited.get('PATH', os.defpath)
    if not isinstance(remainder, str):
        raise SetupError('Engine PATH must be a string')
    executable = interpreter_path(python, path=remainder)
    if executable is None:
        return result
    directory = str(Path(executable).parent)
    if remainder.split(os.pathsep, 1)[0] != directory:
        remainder = directory + os.pathsep + remainder
    result['PATH'] = remainder
    return result
