# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import hashlib
import hmac
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT.parents[1] / 'profiles/deepseek41-flash1017-v1'

from flashalign_native.core import load_profiles
from flashalign_native.native_profile import NativeProfile, validate_native_selection


def profile():
    return NativeProfile(PROFILE, allow_research=True)


def test_native_profile_is_explicit_and_separate():
    item = profile()
    loaded = load_profiles([item.directory], allow_research=True)
    assert isinstance(loaded[item.name], NativeProfile)
    assert len(item.digest) == 64
    assert item.operators == {}
    assert set(item.sites) == {(n, 'attn_out') for n in range(10, 36)}
    assert item.meta['transform'] == 'native_matrix_override_v1'


def test_native_public_profile_needs_no_private_research_flag():
    assert NativeProfile(PROFILE).name == 'deepseek41-flash1017-v1'


def test_native_manifest_is_pinned(tmp_path):
    raw = (PROFILE / 'profile.json').read_bytes()
    (tmp_path / 'profile.json').write_bytes(raw + b' ')
    (tmp_path / 'sha256.json').write_bytes((PROFILE / 'sha256.json').read_bytes())
    with pytest.raises(ValueError, match='digest'):
        NativeProfile(tmp_path, allow_research=True)


@pytest.mark.parametrize('scale', [0, 0.5, 1.1, True, False, '1', float('nan'), None])
def test_native_rejects_unimplemented_strength(scale):
    with pytest.raises(ValueError, match='exactly 1.0'):
        validate_native_selection(profile(), scale)


def test_native_accepts_exact_strength():
    validate_native_selection(profile(), 1.0)
    validate_native_selection(profile(), 1)


def overlay_fixture(tmp_path, monkeypatch):
    from flashalign_native import overlay
    engine = tmp_path / 'engine'
    package = tmp_path / 'package'
    engine.mkdir(); (package / 'overlays').mkdir(parents=True)
    original = engine / 'test_target.py'
    replacement = package / 'overlays/test_target.py'
    original.write_text('value = 0\n'); replacement.write_text('value = 1\n')
    (package / 'overlay.py').write_text('')
    (package / 'overlay-pins.json').write_text(json.dumps({'test_target.py': {
        'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'overlay_sha256': hashlib.sha256(replacement.read_bytes()).hexdigest(),
    }}))
    monkeypatch.setattr(overlay, '__file__', str(package / 'overlay.py'))
    monkeypatch.setattr(sys, 'meta_path', [x for x in sys.meta_path if not isinstance(x, overlay.OverlayFinder)])
    monkeypatch.delitem(sys.modules, 'vllm.test_target', raising=False)
    return overlay, engine, original, replacement


def test_overlay_reuses_only_exact_loaded_overlay(tmp_path, monkeypatch):
    overlay, engine, _, replacement = overlay_fixture(tmp_path, monkeypatch)
    finder = overlay.install(engine)
    monkeypatch.setitem(sys.modules, 'vllm.test_target', SimpleNamespace(
        __file__=str(replacement), __spec__=SimpleNamespace(origin=str(replacement))))
    assert overlay.install(engine) is finder
    assert sum(isinstance(x, overlay.OverlayFinder) for x in sys.meta_path) == 1


def test_overlay_rejects_preloaded_original(tmp_path, monkeypatch):
    overlay, engine, original, _ = overlay_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'vllm.test_target', SimpleNamespace(
        __file__=str(original), __spec__=SimpleNamespace(origin=str(original))))
    with pytest.raises(RuntimeError, match='outside the pinned overlay'):
        overlay.install(engine)


def test_overlay_rejects_wrong_origin(tmp_path, monkeypatch):
    overlay, engine, original, replacement = overlay_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'vllm.test_target', SimpleNamespace(
        __file__=str(replacement), __spec__=SimpleNamespace(origin=str(original))))
    with pytest.raises(RuntimeError, match='outside the pinned overlay'):
        overlay.install(engine)


def test_overlay_rejects_changed_loaded_file(tmp_path, monkeypatch):
    overlay, engine, _, replacement = overlay_fixture(tmp_path, monkeypatch)
    replacement.write_text('value = 2\n')
    monkeypatch.setitem(sys.modules, 'vllm.test_target', SimpleNamespace(
        __file__=str(replacement), __spec__=SimpleNamespace(origin=str(replacement))))
    with pytest.raises(RuntimeError, match='Changed FlashAlign model overlay'):
        overlay.install(engine)


def test_unknown_format_fails_without_projection_fallback(tmp_path):
    (tmp_path / 'profile.json').write_text('{"format":"unimplemented"}')
    with pytest.raises(ValueError, match='Unsupported profile format'):
        load_profiles([tmp_path], allow_research=True)


def test_repeat_setup_preserves_original_packed_object(monkeypatch):
    from flashalign_native import native_attach
    calls = []

    class Attention:
        def setup_wo_projection(self):
            calls.append('original')
            self.stock = object()

    def install(attention):
        calls.append('native')
        attention._flashalign_native_weights = object()

    monkeypatch.setattr(native_attach, 'ensure_native_weights', install)
    native_attach.wrap_native_setup(Attention)
    item = Attention()
    item._flashalign_native_runtime = object()
    item.setup_wo_projection()
    original = item.stock
    item.setup_wo_projection()
    assert item.stock is original
    assert calls == ['original', 'native', 'native']


def test_worker_receipt_exact_coverage_and_atomic_replacement(tmp_path, monkeypatch):
    from flashalign_native import native_attach
    tmp_path.chmod(0o700)
    monkeypatch.setenv('FLASHALIGN_NATIVE_RECEIPTS', str(tmp_path))
    runtime = SimpleNamespace(profile=profile())
    for layer in range(10, 36):
        native_attach._receipt(runtime, 0, {'layer': layer, 'proof': 'synthetic unit test'})
    paths = list(tmp_path.glob('rank-0-pid-*.json'))
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_text())
    assert receipt['complete'] and set(receipt['layers']) == {str(n) for n in range(10, 36)}
    assert receipt['profile_digest'] == runtime.profile.digest
    assert paths[0].stat().st_mode & 0o777 == 0o600
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match='binding changed'):
        native_attach._receipt(runtime, 0, {'layer': 10, 'proof': 'changed'})


def test_worker_receipt_rejects_public_directory(tmp_path, monkeypatch):
    from flashalign_native import native_attach
    tmp_path.chmod(0o755)
    monkeypatch.setenv('FLASHALIGN_NATIVE_RECEIPTS', str(tmp_path))
    with pytest.raises(ValueError, match='native receipt directory'):
        native_attach._receipt(SimpleNamespace(profile=profile()), 0, {'layer': 10})


@pytest.mark.parametrize('native_requested', [False, True])
def test_launcher_preserves_native_command_arguments(tmp_path, monkeypatch, native_requested):
    from flashalign_native import launch
    called = []
    item = profile()
    fake = SimpleNamespace(name=item.name, meta=item.meta, is_native_wo=native_requested,
                           validate_payload=lambda: called.append('validated'))
    monkeypatch.setattr(launch, 'load_key', lambda path: bytes(range(32)))
    monkeypatch.setattr(launch, 'load_profiles', lambda *args, **kwargs: {item.name: fake})
    plugin = ModuleType('flashalign_native.plugin')
    plugin.register = lambda: called.append('registered')
    cli = ModuleType('vllm.entrypoints.cli.main')
    cli.main = lambda: called.append(list(sys.argv))
    monkeypatch.setitem(sys.modules, 'flashalign_native.plugin', plugin)
    monkeypatch.setitem(sys.modules, 'vllm.entrypoints.cli.main', cli)
    monkeypatch.setattr(sys, 'argv', ['untouched'])
    for name in ('FLASHALIGN_NATIVE_RECEIPTS', 'FLASHALIGN_DEEPSEEK_PROFILES',
                 'FLASHALIGN_DEEPSEEK_KEY_FILE', 'FLASHALIGN_DEEPSEEK_MODEL_ID',
                 'FLASHALIGN_NATIVE_SERVED_MODELS', 'FLASHALIGN_DEEPSEEK_ALLOW_RESEARCH',
                 'FLASHALIGN_NATIVE_SECURE_LAUNCH', 'FLASHALIGN_REPLAY_STORE',
                 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_PLUGINS'):
        monkeypatch.delenv(name, raising=False)
    tmp_path.chmod(0o700)
    args = ['--profile', str(item.directory), '--key-file', str(tmp_path / 'key'),
            '--replay-file', str(tmp_path / 'replay'), '--canonical-model', launch.MODEL,
            ]
    if native_requested:
        args += ['--native-receipts', str(tmp_path)]
    args += ['--', 'serve', '/model', '--served-model-name', 'test-model', '--port', '1']
    launch.main(args)
    assert called[-1] == ['vllm', 'serve', '/model', '--served-model-name', 'test-model',
                          '--port', '1', '--middleware', 'flashalign_native.middleware.OwnerMiddleware']


@pytest.mark.parametrize('authenticated,scale,status,effective', [
    (True, 0.5, 400, 'off'), (True, 1.0, 200, 'native'), (False, 0.5, 200, 'off'),
])
def test_native_strength_is_checked_after_authentication(authenticated, scale, status, effective):
    from flashalign_native.middleware import OwnerMiddleware
    from flashalign_native.protocol import Gate
    item = profile()
    key = bytes(range(32))
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'fixture'}],
            'owner_mode': item.name, 'owner_mode_scale': scale}
    raw = json.dumps(body, separators=(',', ':')).encode()
    headers = []
    if authenticated:
        stamp, nonce = str(int(time.time())), 'a' * 32
        signed = (stamp + '\n' + nonce + '\n' + hashlib.sha256(raw).hexdigest()).encode()
        header = 'v1:' + stamp + ':' + nonce + ':' + hmac.new(key, signed, hashlib.sha256).hexdigest()
        headers.append((b'x-owner-mode-auth', header.encode()))
    sent = []

    async def app(scope, receive, send):
        forwarded = json.loads((await receive())['body'])
        extra = forwarded.get('vllm_xargs', {})
        if authenticated and scale == 1:
            assert extra['_flashalign_digest'] == item.digest
            assert extra['_flashalign_scale'] == 1
        else:
            assert '_flashalign_digest' not in extra
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'{}'})

    middleware = object.__new__(OwnerMiddleware)
    middleware.app = app
    middleware.profiles = {item.name: item}
    middleware.gate = Gate(middleware.profiles, key)
    middleware.model = item.meta['model']['id']
    middleware.served_models = ['test-model']

    async def receive():
        return {'type': 'http.request', 'body': raw, 'more_body': False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware({'type': 'http', 'method': 'POST',
                            'path': '/v1/chat/completions', 'headers': headers}, receive, send))
    start = next(message for message in sent if message['type'] == 'http.response.start')
    assert start['status'] == status
    mode = dict(start['headers'])[b'x-owner-mode-effective'].decode()
    assert mode == (item.name if effective == 'native' else 'off')


def exchange(body, *, sign=True, repeat=False):
    """Exercise the ASGI boundary with synthetic data and no engine import."""
    from flashalign_native.middleware import OwnerMiddleware
    from flashalign_native.protocol import Gate
    item = profile(); key = bytes(range(32)); sent = []; forwarded = []
    raw = json.dumps(body, separators=(',', ':')).encode()
    stamp, nonce = str(int(time.time())), 'z' * 32
    material = (stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()).encode()
    auth = 'v1:'+stamp+':'+nonce+':'+hmac.new(key, material, hashlib.sha256).hexdigest()
    async def app(scope, receive, send):
        forwarded.append((scope, json.loads((await receive())['body'])))
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'{}'})
    middleware = object.__new__(OwnerMiddleware)
    middleware.app=app; middleware.profiles={item.name:item}
    middleware.model=item.meta['model']['id']; middleware.served_models=['synthetic-alias']
    middleware.gate=Gate(middleware.profiles,key)
    async def receive():return {'type':'http.request','body':raw,'more_body':False}
    async def send(value):sent.append(value)
    headers=[(b'x-owner-mode-auth',auth.encode())] if sign else []
    scope={'type':'http','method':'POST','path':'/v1/chat/completions','headers':headers}
    asyncio.run(middleware(scope,receive,send))
    if repeat:asyncio.run(middleware(scope,receive,send))
    return forwarded,sent


def test_unsigned_metadata_is_removed_and_cache_namespace_cannot_be_forged():
    item=profile()
    body={'model':item.meta['model']['id'],'messages':[{'role':'user','content':'fixture'}],
          'owner_mode':item.name,'cache_salt':'caller-salt',
          'vllm_xargs':{'_flashalign_profile':item.name,'_flashalign_digest':item.digest}}
    unsigned,_=exchange(body,sign=False)
    signed,_=exchange(body)
    off=unsigned[0][1]; on=signed[0][1]
    assert off['model']==on['model']=='synthetic-alias'
    assert 'vllm_xargs' not in off
    assert on['vllm_xargs']['_flashalign_digest']==item.digest
    assert off['cache_salt']!=on['cache_salt']!='caller-salt'
    assert all(key!=b'x-owner-mode-auth' for key,_ in signed[0][0]['headers'])


def test_replayed_signature_falls_back_and_media_cache_stays_namespaced():
    item=profile()
    body={'model':item.meta['model']['id'],'owner_mode':item.name,
          'messages':[{'role':'user','content':[{'type':'image_url',
          'image_url':{'url':'data:image/png;base64,c3ludGhldGlj'},'uuid':'caller-value'}]}]}
    forwarded,sent=exchange(body,repeat=True)
    first,second=(row[1] for row in forwarded)
    assert first['messages'][0]['content'][0]['uuid']!=second['messages'][0]['content'][0]['uuid']
    assert first['vllm_xargs']['_flashalign_profile']==item.name
    assert 'vllm_xargs' not in second
    effective=[dict(row['headers'])[b'x-owner-mode-effective'] for row in sent
               if row['type']=='http.response.start']
    assert effective==[item.name.encode(),b'off']


def test_websocket_and_precomputed_embedding_routes_fail_closed():
    from flashalign_native.middleware import OwnerMiddleware
    middleware=object.__new__(OwnerMiddleware)
    sent=[]
    async def unused():raise AssertionError('websocket must not reach the engine')
    async def send(value):sent.append(value)
    asyncio.run(middleware({'type':'websocket'},unused,send))
    assert sent==[{'type':'websocket.close','code':1008}]
    item=profile()
    _,sent=exchange({'model':item.meta['model']['id'],'owner_mode':item.name,
                     'messages':[],'prompt_embeds':[0]})
    assert sent[0]['status']==400


def test_reasoning_none_overrides_both_native_thinking_aliases():
    from flashalign_native.compatibility import normalize_chat_request
    result=normalize_chat_request({'reasoning_effort':'none'})
    assert result['reasoning_effort']=='none'
    assert result['chat_template_kwargs']=={
        'thinking':False,'enable_thinking':False,'reasoning_effort':'none'}
    with pytest.raises(ValueError):
        normalize_chat_request({'reasoning_effort':'none','chat_template_kwargs':{'thinking':True}})
    with pytest.raises(ValueError):
        normalize_chat_request({'chat_template_kwargs':{'thinking':True,'enable_thinking':False}})
