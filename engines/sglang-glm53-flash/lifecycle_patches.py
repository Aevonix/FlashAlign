# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Exact-preimage additions for the pinned SGLang control transport."""


def apply_lifecycle_patches(edit):
    usage = 'python/sglang/srt/entrypoints/openai/usage_processor.py'
    edit(usage, '''        """Return PromptTokensDetails only when count > 0 (keeps JSON slim)."""
        return PromptTokensDetails(cached_tokens=count) if count > 0 else None''', '''        """Return the measured count, including zero when reporting is enabled."""
        if type(count) is not int or count < 0:
            raise ValueError("Invalid measured cached-token count")
        return PromptTokensDetails(cached_tokens=count)''')
    io = 'python/sglang/srt/managers/io_struct.py'
    edit(io, 'class RpcReqInput(BaseReq, kw_only=True):', '''class FlashAlignControlReqInput(BaseReq, kw_only=True):
    payload: str


class FlashAlignControlReqOutput(BaseReq, kw_only=True):
    payload: str


class RpcReqInput(BaseReq, kw_only=True):''')
    tokenizer = 'python/sglang/srt/managers/tokenizer_manager.py'
    edit(tokenizer, '        self.init_communicators()\n', '''        self.init_communicators()
        from sglang.srt.flashalign.lifecycle_api import install_tokenizer_control
        install_tokenizer_control(self)
''')
    scheduler = 'python/sglang/srt/managers/scheduler.py'
    edit(scheduler, '    def init_request_dispatcher(self):\n        self._request_dispatcher', '''    def init_request_dispatcher(self):
        from sglang.srt.managers.io_struct import FlashAlignControlReqInput
        from sglang.srt.flashalign.lifecycle_worker import handle_control, dispatch_generate, dispatch_batch_generate, dispatch_abort
        self._request_dispatcher''')
    edit(scheduler, '                (TokenizedGenerateReqInput, self.handle_generate_request),', '''                (FlashAlignControlReqInput, lambda request: handle_control(self, request)),
                (TokenizedGenerateReqInput, lambda request: dispatch_generate(self, request)),''')
    edit(scheduler, '                (AbortReq, self.abort_request),',
         '                (AbortReq, lambda request: dispatch_abort(self, request)),')
    edit(scheduler, '                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),',
         '                (BatchTokenizedGenerateReqInput, lambda request: dispatch_batch_generate(self, request)),')
    runner = 'python/sglang/srt/model_executor/model_runner.py'
    edit(runner, '        install(self)\n', '''        install(self)
        from sglang.srt.flashalign.lifecycle_worker import register_runner
        register_runner(self)
''')
    edit(runner, '                mode_check()\n                and self.decode_cuda_graph_runner',
         '                mode_check()\n                and not getattr(forward_batch, "flashalign_force_eager", False)\n                and self.decode_cuda_graph_runner')
    edit(runner, '                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)\n                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)',
         '                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)\n                and not getattr(forward_batch, "flashalign_force_eager", False)\n                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)')
    http = 'python/sglang/srt/entrypoints/http_server.py'
    edit(http, '        # Start the HTTP server\n        yield\n    finally:', '''        # Start the HTTP server
        from sglang.srt.flashalign.lifecycle_api import start_api, stop_api
        await start_api(_global_state.tokenizer_manager)
        yield
    finally:
        from sglang.srt.flashalign.lifecycle_api import stop_api
        await stop_api()''')


def copy_lifecycle(destination):
    """Vendor canonical pure modules into a prepared isolated engine package."""
    import hashlib
    from pathlib import Path
    import shutil
    root = Path(__file__).resolve().parent
    local = root if root.name == 'common' else root / 'flashalign_sglang'
    prefix = 'sglang_' if root.name == 'common' else ''
    sources = {'flashalign_lifecycle.py': local / 'flashalign_lifecycle.py',
               'flashalign_admin_server.py': local / 'flashalign_admin_server.py',
               **{f'lifecycle_{name}.py': local / f'{prefix}lifecycle_{name}.py'
                  for name in ('api', 'worker', 'support')}}
    canonical = root.parents[1] / 'hermes-owner-mode' / 'profile_lifecycle.py'
    if canonical.exists() and canonical.read_bytes() != sources['flashalign_lifecycle.py'].read_bytes():
        raise ValueError('Generated lifecycle module differs from its canonical source')
    result = {}
    for name, source in sources.items():
        shutil.copyfile(source, destination / name)
        result[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result
