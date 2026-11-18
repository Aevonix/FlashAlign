# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Low-rank profiles and profile sets on CPU. The GPU kernels need their own qualification."""
import asyncio
import hashlib
import hmac
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from flashalign_native import delta_profile
from flashalign_native.profile_sets import Applied, authorize_set, parse_set, set_cache_key
from flashalign_native.delta_runtime import parse_selection, request_rows
from flashalign_native.protocol import FieldError, Gate, Selection, cache_key

KEY = bytes(range(32))
MODEL = dict(delta_profile.MODEL)


def fake(name, fill):
    return SimpleNamespace(name=name, digest=fill * 64, meta={'model': MODEL})


PRIMARY = {'main': fake('main', '1')}
LOW_RANK = {'focus': fake('focus', '2'), 'careful': fake('careful', '3')}


def signed(body, nonce='a' * 32):
    raw = json.dumps(body, separators=(',', ':')).encode()
    stamp = str(int(time.time()))
    material = (stamp + '\n' + nonce + '\n' + hashlib.sha256(raw).hexdigest()).encode()
    return raw, ['v1:' + stamp + ':' + nonce + ':' + hmac.new(KEY, material, hashlib.sha256).hexdigest()]


def authorize(body, *, sign=True, gate=None, nonce='a' * 32):
    raw, headers = signed(body, nonce)
    return authorize_set(gate or Gate(PRIMARY, KEY), LOW_RANK, raw, body, headers if sign else [])


def test_a_set_applies_every_named_profile_in_request_order():
    applied = authorize({'owner_mode': ['focus', 'main', 'careful'], 'owner_mode_scale': [0.5, 1, 0.25]})
    assert applied.names == ('focus', 'main', 'careful')
    assert applied.primary == Selection('main', '1' * 64, 1.0)
    assert applied.low_rank == (Selection('focus', '2' * 64, 0.5), Selection('careful', '3' * 64, 0.25))


def test_a_set_may_hold_only_low_rank_profiles():
    applied = authorize({'owner_mode': ['careful', 'focus']})
    assert applied.primary is None
    assert [row.name for row in applied.low_rank] == ['careful', 'focus']
    assert [row.scale for row in applied.low_rank] == [1.0, 1.0]


def test_one_low_rank_profile_keeps_the_scalar_form():
    applied = authorize({'owner_mode': 'focus', 'owner_mode_scale': 0.75})
    assert applied == Applied(None, (Selection('focus', '2' * 64, 0.75),), ('focus',))


def test_one_matrix_profile_is_authorized_exactly_as_before():
    body = {'owner_mode': 'main'}
    raw, headers = signed(body)
    assert authorize(body) == Applied(Gate(PRIMARY, KEY).authorize(raw, body, headers), (), ('main',))
    assert authorize({'model': 'x'}) == Applied()


@pytest.mark.parametrize('body', [
    {'owner_mode': ['focus', 'careful']}, {'owner_mode': 'focus'}, {'owner_mode': ['main', 'focus']}])
def test_an_unsigned_request_gets_no_profile(body):
    assert authorize(body, sign=False) == Applied()


def test_a_replayed_set_signature_gets_no_profile():
    gate = Gate(PRIMARY, KEY)
    body = {'owner_mode': ['main', 'focus']}
    assert authorize(body, gate=gate).names == ('main', 'focus')
    assert authorize(body, gate=gate) == Applied()


def test_a_set_with_an_unknown_profile_is_rejected_whole():
    with pytest.raises(FieldError, match='owner_mode names no profile this server loaded'):
        authorize({'owner_mode': ['focus', 'absent']})


def test_a_set_holds_at_most_one_profile_that_replaces_an_operation():
    gate = Gate({**PRIMARY, 'other': fake('other', '4')}, KEY)
    with pytest.raises(FieldError, match='at most one'):
        authorize({'owner_mode': ['main', 'other']}, gate=gate)


@pytest.mark.parametrize('mode,scale', [
    (['focus'], None), ([], None), (['focus', 'focus'], None), (['focus', 'off'], None), (['focus', 7], None),
    (['focus'] * 9, None), (['focus', 'careful'], 0.5), (['focus', 'careful'], [1.0]),
    (['focus', 'careful'], [1.0, 0]), (['focus', 'careful'], [1.0, 1.5]), (['focus', 'careful'], [1.0, True]),
    (['focus', 'careful'], [1.0, float('nan')])])
def test_malformed_sets_are_rejected_before_authentication(mode, scale):
    body = {'owner_mode': mode}
    if scale is not None:
        body['owner_mode_scale'] = scale
    with pytest.raises(FieldError):
        parse_set(body)
    with pytest.raises(FieldError):
        authorize_set(Gate(PRIMARY, KEY), LOW_RANK, b'{}', body, [])


def test_the_cache_namespace_covers_the_whole_set():
    main = Selection('main', '1' * 64, 1.0)
    focus, careful = Selection('focus', '2' * 64, 1.0), Selection('careful', '3' * 64, 1.0)
    def key(applied, original=None):
        return set_cache_key(applied, PRIMARY, 'model', original)
    variants = [
        Applied(), Applied(main, (), ('main',)),
        Applied(None, (focus,), ('focus',)), Applied(None, (careful,), ('careful',)),
        Applied(None, (focus, careful), ('focus', 'careful')), Applied(None, (careful, focus), ('careful', 'focus')),
        Applied(main, (focus,), ('main', 'focus')), Applied(main, (focus,), ('focus', 'main')),
        Applied(main, (focus, careful), ('main', 'focus', 'careful')),
        Applied(None, (Selection('focus', '2' * 64, 0.5),), ('focus',)),
        Applied(None, (Selection('focus', '9' * 64, 1.0),), ('focus',)),
        Applied(Selection('main', '1' * 64, 1.0, 'generation-2'), (focus,), ('main', 'focus')),
    ]
    keys = [key(applied) for applied in variants]
    assert len(set(keys)) == len(keys)
    assert key(variants[4], 'salt') != key(variants[4])
    # Requests without a low-rank profile keep the key they had before.
    assert key(Applied()) == cache_key(None, PRIMARY, 'model', None)
    assert key(variants[1], 'salt') == cache_key(main, PRIMARY, 'model', 'salt')


def test_a_set_reports_the_generation_of_its_updatable_profile_at_that_position():
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'fixture'}],
            'owner_mode': ['careful', 'main', 'focus']}
    lifecycle = Lifecycle()
    status, headers, forwarded = exchange(body, lifecycle=lifecycle)
    assert status == 200 and lifecycle.open == 0
    assert header(headers, b'x-owner-mode-effective') == 'careful,main,focus'
    # The client reads these two headers position by position; a static profile's generation is empty.
    assert header(headers, b'x-flashalign-generation') == ',generation-7,'
    assert header(headers, b'x-flashalign-profile-digest') == ','.join(['3' * 64, '7' * 64, '2' * 64])
    assert forwarded['vllm_xargs']['_flashalign_generation'] == 'generation-7'


def test_a_name_an_update_loaded_a_second_time_is_rejected_not_guessed():
    gate = Gate({**PRIMARY, 'focus': PRIMARY['main']}, KEY)
    for index, mode in enumerate(('focus', ['focus', 'careful'], ['main', 'focus'])):
        with pytest.raises(FieldError, match='loaded twice'):
            authorize({'owner_mode': mode}, gate=gate, nonce=str(index) * 32)
    assert authorize({'owner_mode': ['main', 'careful']}, gate=gate, nonce='9' * 32).names == ('main', 'careful')


def test_no_caller_salt_reaches_a_set_namespace():
    applied = Applied(None, (Selection('focus', '2' * 64, 1.0),), ('focus',))
    target = set_cache_key(applied, PRIMARY, 'model', None)
    forged = ['flashalign.profile-set.v1', 'model', None, [['focus', '2' * 64, (1.0).hex()]], ['focus'], None]
    assert target.startswith('flashalign-set:')
    assert set_cache_key(Applied(), PRIMARY, 'model', forged) != target
    assert set_cache_key(Applied(), PRIMARY, 'model', target) != target


def exchange(body, *, sign=True, path='/v1/chat/completions', method='POST', lifecycle=None):
    from flashalign_native.middleware import OwnerMiddleware
    raw, auth = signed(body)
    forwarded, sent = [], []

    async def app(scope, receive, send):
        forwarded.append(json.loads((await receive())['body']))
        await send({'type': 'http.response.start', 'status': 200, 'headers': [
            (b'x-owner-mode-effective', b'forged'), (b'x-flashalign-profile-digest', b'forged')]})
        await send({'type': 'http.response.body', 'body': b'{}'})

    middleware = object.__new__(OwnerMiddleware)
    middleware.lifecycle = lifecycle
    middleware.app = app
    middleware.profiles = PRIMARY
    middleware.low_rank = LOW_RANK
    middleware.gate = Gate(PRIMARY, KEY)
    middleware.model = MODEL['id']
    middleware.served_models = ['test-model']

    async def receive():
        return {'type': 'http.request', 'body': raw, 'more_body': False}

    async def send(message):
        sent.append(message)

    headers = [(b'x-owner-mode-auth', auth[0].encode())] if sign else []
    asyncio.run(middleware({'type': 'http', 'method': method, 'path': path, 'headers': headers}, receive, send))
    start = next(message for message in sent if message['type'] == 'http.response.start')
    reply = b''.join(message.get('body', b'') for message in sent if message['type'] == 'http.response.body')
    return start['status'], start['headers'], forwarded[0] if forwarded else json.loads(reply)


class Lifecycle:
    """Pins every request to one generation of the profile it names."""
    def __init__(self):
        self.open = 0

    def pin(self, name):
        self.open += 1
        generation = SimpleNamespace(id='generation-7', artifact=SimpleNamespace(digest='7' * 64))
        return SimpleNamespace(generation=generation, close=self.close)

    def close(self):
        self.open -= 1


def header(headers, name):
    values = [value.decode() for field, value in headers if field == name]
    assert len(values) <= 1
    return values[0] if values else None


def test_the_boundary_forwards_a_signed_set_and_reports_it():
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'fixture'}],
            'owner_mode': ['careful', 'main', 'focus'], 'owner_mode_scale': [0.5, 1, 1],
            'vllm_xargs': {'_flashalign_low_rank': 'forged', 'kept': 1}}
    status, headers, forwarded = exchange(body)
    assert status == 200
    assert header(headers, b'x-owner-mode-effective') == 'careful,main,focus'
    assert header(headers, b'x-flashalign-profile-digest') == ','.join(['3' * 64, '1' * 64, '2' * 64])
    assert header(headers, b'x-flashalign-generation') is None
    extra = forwarded['vllm_xargs']
    assert json.loads(extra['_flashalign_low_rank']) == [['careful', '3' * 64, 0.5], ['focus', '2' * 64, 1.0]]
    assert (extra['_flashalign_profile'], extra['_flashalign_digest'], extra['kept']) == ('main', '1' * 64, 1)
    assert 'owner_mode' not in forwarded and 'owner_mode_scale' not in forwarded
    applied = Applied(Selection('main', '1' * 64, 1.0),
                      (Selection('careful', '3' * 64, 0.5), Selection('focus', '2' * 64, 1.0)),
                      ('careful', 'main', 'focus'))
    assert forwarded['cache_salt'] == set_cache_key(applied, PRIMARY, MODEL['id'], None)


def test_the_boundary_serves_an_unsigned_set_as_standard_and_strips_forged_metadata():
    body = {'model': 'test-model', 'messages': [], 'owner_mode': ['focus', 'careful'],
            'vllm_xargs': {'_flashalign_low_rank': json.dumps([['focus', '2' * 64, 1.0]])}}
    status, headers, forwarded = exchange(body, sign=False)
    assert status == 200
    assert header(headers, b'x-owner-mode-effective') == 'off'
    assert header(headers, b'x-flashalign-profile-digest') is None
    assert 'vllm_xargs' not in forwarded
    assert forwarded['cache_salt'] == cache_key(None, PRIMARY, MODEL['id'], None)


def test_the_boundary_rejects_a_set_with_an_unknown_profile_and_precomputed_inputs():
    status, headers, reply = exchange({'model': 'test-model', 'messages': [], 'owner_mode': ['focus', 'absent']})
    assert (status, header(headers, b'x-owner-mode-effective')) == (400, 'off')
    assert reply['error'] == {'code': 'invalid_owner_mode', 'type': 'invalid_request_error', 'param': 'owner_mode',
                              'message': 'owner_mode names no profile this server loaded'}
    status, _, reply = exchange({'model': 'test-model', 'prompt': 'x', 'prompt_embeds': 'x', 'owner_mode': 'focus'},
                                path='/v1/completions')
    assert status == 400 and 'precomputed' in reply['error']['message']


def test_discovery_lists_every_loaded_profile_in_engine_order():
    status, _, reply = exchange({}, path='/v1/owner-mode/profiles', method='GET')
    assert status == 200
    assert [row['name'] for row in reply['data']] == ['main', 'focus', 'careful']
    assert all(row['loaded'] is True and row['model'] == MODEL for row in reply['data'])


def test_the_worker_checks_low_rank_selections_against_its_own_profiles():
    raw = json.dumps([['careful', '3' * 64, 0.5], ['focus', '2' * 64, 1]])
    assert parse_selection(raw, LOW_RANK) == {'careful': 0.5, 'focus': 1.0}
    for bad in ('forged', '[]', '{}', json.dumps([['focus', '9' * 64, 1.0]]), json.dumps([['absent', '2' * 64, 1.0]]),
                json.dumps([['focus', '2' * 64, 1.0]] * 2), json.dumps([['focus', '2' * 64, 0]]),
                json.dumps([['focus', '2' * 64, 1.5]]), json.dumps([['focus', '2' * 64, True]]),
                json.dumps([['focus', '2' * 64]]), json.dumps([['focus', '2' * 64, 1.0]] * 3), None, 7):
        with pytest.raises(ValueError):
            parse_selection(bad, LOW_RANK)


def test_each_profile_gets_its_own_request_rows():
    selections = {'a': {'focus': 0.5, 'careful': 1.0}, 'b': {'careful': 0.25}}
    assert request_rows(['a', 'b', 'c'], selections, 'focus') == ([True, False, False], [0.5, 0.0, 0.0])
    assert request_rows(['a', 'b', 'c'], selections, 'careful') == ([True, True, False], [1.0, 0.25, 0.0])


def write_profile(directory, name='focus', sites=((23, 'attn_out', 2), (23, 'shared_expert_out', 3)),
                  manifest=None, tensors=None, seed=0):
    torch = pytest.importorskip('torch')
    save_file = pytest.importorskip('safetensors.torch').save_file
    generator = torch.Generator().manual_seed(seed)
    values, rows = {}, []
    for layer, site, rank in sites:
        tensor, width = delta_profile.SITES[site]
        q, _ = torch.linalg.qr(torch.randn(delta_profile.HIDDEN, rank, generator=generator))
        values[f'L{layer}.{site}.Q'] = q.T.contiguous().float()
        values[f'L{layer}.{site}.B'] = (torch.randn(rank, width, generator=generator) * 0.02).float()
        rows.append({'layer': layer, 'site': site, 'rank': rank, 'scale': 1.0, 'input_size': width,
                     'tensor': tensor.format(layer=layer)})
    meta = {'name': name, 'transform': delta_profile.TRANSFORM, 'requires': ['matrix_input'],
            'hidden_size': delta_profile.HIDDEN, 'model': MODEL, 'sites': rows}
    if manifest is not None:
        manifest(meta)
    if tensors is not None:
        tensors(values)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(values, str(directory / 'profile.safetensors'))
    (directory / 'profile.json').write_text(json.dumps(meta))
    (directory / 'sha256.json').write_text(json.dumps({
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in ('profile.json', 'profile.safetensors')}))
    return directory


def test_a_low_rank_profile_loads_with_its_identity(tmp_path):
    directory = write_profile(tmp_path / 'focus')
    profile = delta_profile.DeltaProfile(directory)
    name, digest, meta, _ = delta_profile.read_manifest(directory)
    assert (profile.name, profile.digest) == (name, digest) == ('focus', hashlib.sha256(
        (directory / 'profile.json').read_bytes() + (directory / 'profile.safetensors').read_bytes()).hexdigest())
    assert set(profile.sites) == {(23, 'attn_out'), (23, 'shared_expert_out')}
    q, b, scale = profile.sites[(23, 'shared_expert_out')]
    assert (tuple(q.shape), tuple(b.shape), scale) == ((3, 5120), (3, 2304), 1.0)
    assert delta_profile.is_delta_manifest(meta) and not delta_profile.is_delta_manifest({'transform': 'projection'})


def _drop(key):
    return lambda value: value.pop(key)


def _set(key, value):
    return lambda target: target.__setitem__(key, value)


def _site(key, value):
    return lambda meta: meta['sites'][0].__setitem__(key, value)


@pytest.mark.parametrize('manifest,tensors', [
    (_set('transform', 'projection'), None), (_set('requires', []), None), (_set('schema', 'x'), None),
    (_set('format', 'flashalign.native_wo_b.v1'), None), (_set('hidden_size', 4096), None),
    (_set('model', {'id': 'other', 'revision': 'x'}), None), (_set('name', 'off'), None), (_set('name', 'a b'), None),
    (_set('sites', []), None), (_site('layer', 40), None), (_site('site', 'moe_out'), None),
    (_site('tensor', 'layers.23.attn.wo_a.weight'), None), (_site('input_size', 2304), None),
    (_site('rank', 0), None), (_site('rank', 129), None), (_site('rank', 3), None), (_site('scale', float('inf')), None),
    (lambda meta: meta['sites'].append(dict(meta['sites'][0])), None),
    (None, _drop('L23.attn_out.B')), (None, _drop('L23.attn_out.Q')),
    (None, lambda values: values.__setitem__('L23.attn_out.Q', values['L23.attn_out.Q'] * 2)),
    (None, lambda values: values.__setitem__('L23.attn_out.B', values['L23.attn_out.B'][:, :2048].contiguous())),
    (None, lambda values: values.__setitem__('L23.attn_out.B', values['L23.attn_out.B'].double())),
    (None, lambda values: values.__setitem__('L23.attn_out.B', values['L23.attn_out.B'] * float('nan'))),
    (None, lambda values: values.__setitem__('extra', values['L23.attn_out.Q'].clone())),
])
def test_a_low_rank_profile_outside_the_contract_is_rejected(tmp_path, manifest, tensors):
    directory = write_profile(tmp_path / 'bad', manifest=manifest, tensors=tensors)
    with pytest.raises(ValueError):
        delta_profile.DeltaProfile(directory)


def test_a_changed_profile_file_fails_its_checksum(tmp_path):
    directory = write_profile(tmp_path / 'focus')
    (directory / 'profile.json').write_text((directory / 'profile.json').read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        delta_profile.read_manifest(directory)


def test_profiles_load_in_the_given_order_with_distinct_names(tmp_path):
    first = write_profile(tmp_path / 'a', name='careful', seed=1)
    second = write_profile(tmp_path / 'b', name='focus', seed=2)
    assert list(delta_profile.load_delta_profiles([str(first), str(second)])) == ['careful', 'focus']
    assert list(delta_profile.load_delta_profiles([str(second), str(first)])) == ['focus', 'careful']
    assert delta_profile.load_delta_profiles([]) == {}
    with pytest.raises(ValueError, match='Duplicate'):
        delta_profile.load_delta_profiles([str(first), str(first)])
    with pytest.raises(ValueError, match='seven'):
        delta_profile.load_delta_profiles([str(first)] * 8)


@pytest.mark.parametrize('width,per', [(8192, 2048), (2304, 576), (2304, 640)])
def test_the_tensor_parallel_shards_of_an_input_factor_sum_to_the_whole(width, per):
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(3)
    b = torch.randn(4, width, generator=generator)
    source = torch.randn(5, width, generator=generator)
    padded = torch.cat([source, torch.zeros(5, per * 4 - width)], dim=1)
    shards = [delta_profile.local_columns(b, rank, 4, per) for rank in range(4)]
    assert all(tuple(shard.shape) == (4, per) for shard in shards)
    assert torch.equal(torch.cat(shards, dim=1)[:, :width], b)
    assert not torch.cat(shards, dim=1)[:, width:].any()
    total = sum(padded[:, rank * per:(rank + 1) * per] @ shards[rank].T for rank in range(4))
    assert torch.allclose(total, source @ b.T, rtol=0, atol=1e-3)


@pytest.mark.parametrize('width,per,rank,size', [(2304, 512, 0, 4), (2304, 1024, 0, 4), (2304, 576, 4, 4),
                                                (2304, 576, -1, 4), (2304, 0, 0, 4)])
def test_an_input_factor_that_does_not_fit_the_writer_partition_is_rejected(width, per, rank, size):
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError):
        delta_profile.local_columns(torch.zeros(2, width), rank, size, per)


def test_the_reference_matches_a_merged_weight_and_leaves_other_rows_alone():
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(5)
    hidden, width, rank, rows = delta_profile.HIDDEN, 2304, 3, 6
    weight = torch.randn(hidden, width, generator=generator, dtype=torch.float64) * 0.02
    q, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=generator, dtype=torch.float64))
    q, b = q.T.contiguous(), torch.randn(rank, width, generator=generator, dtype=torch.float64) * 0.02
    source = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    state = [(True, 1.0), (False, 0.0), (True, 0.5), (False, 0.0), (True, 0.25), (False, 0.0)]
    stock = source @ weight.T
    poisoned = stock.clone()
    poisoned[[1, 3, 5]] = float('nan')
    result = delta_profile.reference_delta(poisoned, source, q, b, state, 0.8)
    assert torch.equal(result[[1, 3, 5]].view(torch.uint8), poisoned[[1, 3, 5]].view(torch.uint8))
    for index, (selected, strength) in enumerate(state):
        if selected:
            merged = source[index] @ (weight + strength * 0.8 * (q.T @ b)).T
            # The reference computes in FP32, like the kernel.
            assert torch.allclose(result[index], merged, rtol=0, atol=1e-5)


def test_profiles_on_one_site_add_up_in_any_order():
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(6)
    hidden, width = delta_profile.HIDDEN, 2304
    def operator(rank):
        q, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=generator, dtype=torch.float64))
        return q.T.contiguous(), torch.randn(rank, width, generator=generator, dtype=torch.float64) * 0.02
    (q1, b1), (q2, b2) = operator(2), operator(4)
    source = torch.randn(3, width, generator=generator, dtype=torch.float64)
    output = torch.randn(3, hidden, generator=generator, dtype=torch.float64)
    first, second = [(True, 1.0), (True, 0.5), (False, 0.0)], [(False, 0.0), (True, 1.0), (True, 1.0)]
    one = delta_profile.reference_delta(delta_profile.reference_delta(output, source, q1, b1, first, 1.0),
                                        source, q2, b2, second, 1.0)
    other = delta_profile.reference_delta(delta_profile.reference_delta(output, source, q2, b2, second, 1.0),
                                          source, q1, b1, first, 1.0)
    assert torch.allclose(one, other, rtol=0, atol=1e-5)
    assert torch.equal(one[0], delta_profile.reference_delta(output, source, q1, b1, first, 1.0)[0])


def test_the_launcher_mounts_and_passes_low_rank_profiles_in_order(tmp_path, monkeypatch):
    import launch
    first = write_profile(tmp_path / 'one', name='careful', seed=1)
    second = write_profile(tmp_path / 'two', name='focus', seed=2)
    low_rank = launch.low_rank_profiles([first, second], 'main')
    assert [(name, directory) for name, _, directory in low_rank] == [('careful', first), ('focus', second)]
    assert low_rank[0][1] == delta_profile.read_manifest(first)[1]
    with pytest.raises(ValueError, match='distinct'):
        launch.low_rank_profiles([first, first], 'main')
    with pytest.raises(ValueError, match='distinct'):
        launch.low_rank_profiles([first], 'careful')
    args = SimpleNamespace(host='127.0.0.1', port=8041, served_model_name='served', speculative_config=None,
                           arguments_json='[]', profile_admin_socket=None)
    model = tmp_path / 'model'
    model.mkdir()
    monkeypatch.setattr(launch, 'speculation', lambda *_: {})
    def command(low_rank):
        return launch.container_command(
            args, tmp_path / 'state' / 'runs' / 'run', name='fixture', labels={}, model=model,
            key=tmp_path / 'key', replay=tmp_path / 'replay' / 'replay.sqlite', profile=tmp_path / 'profile',
            devices=['0', '1', '2', '3'], low_rank=low_rank,
            model_layout={'source': model, 'target': '/model', 'model': '/model'})
    with_profiles, without = command(low_rank), command(())
    mounts = [with_profiles[index + 1] for index, value in enumerate(with_profiles) if value == '--mount']
    assert f'type=bind,src={first},dst=/fa-low-rank/0,readonly' in mounts
    assert f'type=bind,src={second},dst=/fa-low-rank/1,readonly' in mounts
    start = with_profiles.index('--profile')
    assert with_profiles[start:start + 6] == ['--profile', '/fa-profile', '--low-rank-profile', '/fa-low-rank/0',
                                              '--low-rank-profile', '/fa-low-rank/1']
    assert '--low-rank-profile' not in without and not any('/fa-low-rank/' in value for value in without)
    stripped, skip = [], False
    for index, value in enumerate(with_profiles):
        following = with_profiles[index + 1] if index + 1 < len(with_profiles) else ''
        if skip or (value in ('--mount', '--low-rank-profile') and '/fa-low-rank/' in following):
            skip = not skip
            continue
        stripped.append(value)
    assert stripped == without


def reference_kernels(torch):
    """The kernel contract in plain tensor code: per-row gate through the staged row map."""
    def stage_rows(row_map, x, indices):
        rows = x.shape[0]
        row_map[:rows] = torch.arange(rows, dtype=torch.int32) if indices is None else indices.to(torch.int32)

    def delta_inplace(x, source, b, q, owners, strengths, row_map, site_scale):
        flat = source if source.ndim == 2 else torch.cat([source[:, :, group] for group in range(source.shape[2])], dim=1)
        assert flat.shape[1] == b.shape[1] and x.shape[1] == q.shape[1]
        for row in range(x.shape[0]):
            origin = int(row_map[row])
            if 0 <= origin < owners.numel() and int(owners[origin]):
                change = float(strengths[origin]) * site_scale * ((flat[row].double() @ b.double().T) @ q.double())
                x[row] = (x[row].double() + change).to(x.dtype)
    return SimpleNamespace(stage_rows=stage_rows, delta_inplace=delta_inplace)


def runtimes(monkeypatch, profiles, capacity=16):
    torch = pytest.importorskip('torch')
    import flashalign_native
    from flashalign_native.delta_runtime import DeltaRuntime
    stub = reference_kernels(torch)
    monkeypatch.setitem(__import__('sys').modules, 'flashalign_native.delta_kernels', stub)
    monkeypatch.setattr(flashalign_native, 'delta_kernels', stub, raising=False)
    runner = SimpleNamespace(device='cpu', max_num_tokens=capacity, max_num_reqs=4)
    return torch, [DeltaRuntime(runner, profiles, tp_rank=rank, tp_size=4) for rank in range(4)]


@pytest.mark.parametrize('site,per,groups', [('attn_out', 2048, 2), ('shared_expert_out', 576, 1),
                                            ('shared_expert_out', 640, 1)])
def test_four_ranks_add_up_to_the_merged_weights_for_the_rows_that_selected_each_profile(
        tmp_path, monkeypatch, site, per, groups):
    first = write_profile(tmp_path / 'a', name='careful', sites=((23, site, 2),), seed=1)
    second = write_profile(tmp_path / 'b', name='focus', sites=((23, site, 3), (24, site, 1)), seed=2)
    profiles = delta_profile.load_delta_profiles([str(first), str(second)])
    torch, ranks = runtimes(monkeypatch, profiles)
    generator = torch.Generator().manual_seed(11)
    width = delta_profile.SITES[site][1]
    padded, rows = per * 4, 5
    weight = torch.randn(delta_profile.HIDDEN, padded, generator=generator, dtype=torch.float64) * 0.02
    weight[:, width:] = 0
    source = torch.randn(rows, padded, generator=generator, dtype=torch.float64)
    source[:, width:] = 0
    # Model rows 0..4 came from request rows 9, 2, 7, 0, 5. careful is on for 9 and 7, focus for 7 and 0.
    indices = torch.tensor([9, 2, 7, 0, 5])
    selected = {'careful': {9: 1.0, 7: 0.5}, 'focus': {7: 0.25, 0: 1.0}}
    total = torch.zeros(rows, delta_profile.HIDDEN, dtype=torch.float64)
    stock = torch.zeros_like(total)
    for rank, runtime in enumerate(ranks):
        assert runtime.names == ('careful', 'focus') and runtime.sites == [(23, site), (24, site)]
        for index, name in enumerate(runtime.names):
            for row, strength in selected[name].items():
                runtime.owner_rows[index, row], runtime.strength_rows[index, row] = 1, strength
        runtime.bind(23, site, per)
        local = source[:, rank * per:(rank + 1) * per]
        output = local @ weight[:, rank * per:(rank + 1) * per].T
        stock += output
        view = local
        if groups > 1:
            # The attention writer's input is stored group by group and viewed as [rows, inner, groups].
            inner = per // groups
            storage = torch.stack([local[:, group * inner:(group + 1) * inner] for group in range(groups)]).contiguous()
            view = storage.as_strided((rows, inner, groups), (inner, 1, rows * inner))
        if site == 'attn_out':
            runtime.attention(23, output[:, :, None], view, indices)
        else:
            # The attention operation of the layer stages the row map; it has no operator of its own here.
            attention = torch.ones(rows, delta_profile.HIDDEN, 1, dtype=torch.float64)
            runtime.attention(23, attention, None, indices)
            assert bool((attention == 1).all())
            runtime.shared_expert(23, output, view)
        total += output
    def merged(row, parts):
        change = sum(strength * (profiles[name].sites[(23, site)][0].double().T
                                 @ torch.cat([profiles[name].sites[(23, site)][1].double(),
                                              torch.zeros(profiles[name].sites[(23, site)][0].shape[0], padded - width,
                                                          dtype=torch.float64)], dim=1))
                     for name, strength in parts)
        return source[row] @ (weight + change).T
    # Rows that selected nothing hold exactly the sum of the stock shares.
    assert torch.equal(total[[1, 4]], stock[[1, 4]])
    assert torch.allclose(stock, source @ weight.T, rtol=0, atol=1e-9)
    assert torch.allclose(total[0], merged(0, [('careful', 1.0)]), rtol=0, atol=1e-9)
    assert torch.allclose(total[2], merged(2, [('careful', 0.5), ('focus', 0.25)]), rtol=0, atol=1e-9)
    assert torch.allclose(total[3], merged(3, [('focus', 1.0)]), rtol=0, atol=1e-9)
    assert not torch.allclose(total[0], stock[0], rtol=0, atol=1e-6)


def test_a_site_that_was_not_bound_or_does_not_fit_is_refused(tmp_path, monkeypatch):
    directory = write_profile(tmp_path / 'a', sites=((23, 'shared_expert_out', 2),))
    torch, ranks = runtimes(monkeypatch, delta_profile.load_delta_profiles([str(directory)]))
    with pytest.raises(ValueError, match='not bound'):
        ranks[0].shared_expert(23, torch.zeros(1, 5120), torch.zeros(1, 576))
    ranks[0].bind(23, 'shared_expert_out', 576)
    ranks[0].attention(23, torch.zeros(2, 5120, 1), None, None)
    with pytest.raises(ValueError, match='differ from the attention rows'):
        ranks[0].shared_expert(23, torch.zeros(1, 5120), torch.zeros(1, 576))
    with pytest.raises(ValueError, match='No low-rank operator'):
        ranks[0].bind(24, 'shared_expert_out', 576)
    with pytest.raises(ValueError, match='writer partition'):
        ranks[0].bind(23, 'shared_expert_out', 2048)


def fake_model(monkeypatch, torch, *, reduce_results=False, tp_rank=1):
    import sys
    from types import ModuleType
    distributed = ModuleType('vllm.distributed')
    distributed.get_tensor_model_parallel_rank = lambda: tp_rank
    distributed.get_tensor_model_parallel_world_size = lambda: 4
    package = ModuleType('vllm')
    package.distributed = distributed
    monkeypatch.setitem(sys.modules, 'vllm', package)
    monkeypatch.setitem(sys.modules, 'vllm.distributed', distributed)
    weight = torch.randn(5120, 576, dtype=torch.float64, generator=torch.Generator().manual_seed(21)) * 0.02

    class Down:
        input_is_parallel, input_size_per_partition = True, 576

        def __init__(self):
            self.reduce_results = reduce_results

        def forward(self, input_):
            return input_ @ weight.T, None

        def __call__(self, input_):
            return self.forward(input_)

    Base = type('DeepseekV4Model', (), {'__module__': 'vllm.models.deepseek_v4_1.nvidia.model'})
    base = Base()
    base.layers = [SimpleNamespace(attn=SimpleNamespace(layer_id=index, n_local_groups=2, o_lora_rank=1024),
                                   ffn=SimpleNamespace(shared_experts=SimpleNamespace(down_proj=Down())))
                   for index in range(40)]
    runner = SimpleNamespace(device='cpu', max_num_tokens=16, max_num_reqs=4, _flashalign_profiles={'main': None},
                             model=SimpleNamespace(language_model=SimpleNamespace(model=base)))
    return runner, base, weight


def test_attaching_binds_both_writers_of_the_targeted_layers_only(tmp_path, monkeypatch):
    directory = write_profile(tmp_path / 'a', sites=((23, 'attn_out', 2), (23, 'shared_expert_out', 2),
                                                     (30, 'shared_expert_out', 1)))
    profiles = delta_profile.load_delta_profiles([str(directory)])
    torch, _ = runtimes(monkeypatch, profiles)
    from flashalign_native.delta_runtime import attach_deltas
    runner, base, weight = fake_model(monkeypatch, torch)
    assert attach_deltas(runner, {}) is None and not hasattr(runner, '_flashalign_delta_runtime')
    runtime = attach_deltas(runner, profiles)
    assert runner._flashalign_delta_runtime is runtime and runner._flashalign_delta_selections == {}
    assert set(runtime.operators) == {(23, 'attn_out'), (23, 'shared_expert_out'), (30, 'shared_expert_out')}
    assert [index for index, layer in enumerate(base.layers) if hasattr(layer, '_flashalign_delta')] == [23, 30]
    # Every targeted layer stages its row map in the attention operation, also a layer with no attention operator.
    assert [index for index, layer in enumerate(base.layers) if hasattr(layer.attn, '_flashalign_delta')] == [23, 30]
    assert base.layers[23].attn._flashalign_delta is runtime and base.layers[30].attn._flashalign_delta_layer == 30
    # This rank holds columns 576..1151 of the shared expert's input.
    _, b, _ = profiles['focus'].sites[(23, 'shared_expert_out')]
    assert torch.equal(runtime.operators[(23, 'shared_expert_out')][0][2], b[:, 576:1152])
    source = torch.randn(3, 576, dtype=torch.float64, generator=torch.Generator().manual_seed(22))
    runtime.owner_rows[0, 1], runtime.strength_rows[0, 1] = 1, 1.0
    down = base.layers[23].ffn.shared_experts.down_proj
    runtime.attention(23, torch.zeros(3, 5120, 1, dtype=torch.float64), torch.zeros(3, 1024, 2), None)
    output, bias = down(source)
    stock = source @ weight.T
    assert bias is None and torch.equal(output[[0, 2]], stock[[0, 2]])
    q = profiles['focus'].sites[(23, 'shared_expert_out')][0].double()
    assert torch.allclose(output[1], stock[1] + (source[1] @ b[:, 576:1152].double().T) @ q, rtol=0, atol=1e-9)
    untouched, _ = base.layers[22].ffn.shared_experts.down_proj(source)
    assert torch.equal(untouched, stock)


def test_attaching_refuses_a_reduced_writer_output_a_foreign_model_and_a_taken_name(tmp_path, monkeypatch):
    directory = write_profile(tmp_path / 'a', sites=((23, 'shared_expert_out', 2),))
    profiles = delta_profile.load_delta_profiles([str(directory)])
    torch, _ = runtimes(monkeypatch, profiles)
    from flashalign_native.delta_runtime import attach_deltas
    runner, _, _ = fake_model(monkeypatch, torch, reduce_results=True)
    with pytest.raises(ValueError, match='before the reduction'):
        attach_deltas(runner, profiles)
    runner, base, _ = fake_model(monkeypatch, torch)
    runner._flashalign_profiles = {'focus': None}
    with pytest.raises(ValueError, match='Duplicate'):
        attach_deltas(runner, profiles)
    runner, base, _ = fake_model(monkeypatch, torch)
    runner.model.language_model.model = SimpleNamespace(layers=base.layers)
    with pytest.raises(ValueError, match='pinned native'):
        attach_deltas(runner, profiles)
