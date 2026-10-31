# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Stock weights with the isolated SM120 compatibility backport."""
import os
import sys

def main():
    os.environ['FLASHALIGN_DEEPSEEK_SM120_GEOMETRY']='64'
    os.environ['VLLM_PLUGINS']='flashalign_deepseek'
    from .plugin import register
    register()
    from vllm.entrypoints.cli.main import main as serve
    sys.argv=['vllm',*sys.argv[1:]]
    serve()


if __name__=='__main__':main()
