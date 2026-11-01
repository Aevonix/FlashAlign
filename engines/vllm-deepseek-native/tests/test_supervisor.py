# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the public deployment lifecycle, without Docker or a model."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine_config
import launch
import prepare
import supervisor


@pytest.mark.parametrize('field,value', [
    ('num_speculative_tokens', 5), ('enable_adaptive_verification', False),
    ('adaptive_verification_cost_scale', True), ('attention_backend', 'OTHER'),
    ('draft_tensor_parallel_size', 1), ('method', 'none'),
])
def test_preserve_supported_speculation_or_fail(field, value):
    config = dict(engine_config.DEFAULT_SPECULATION)
    config[field] = value
    with pytest.raises(ValueError):
        launch.speculation(json.dumps(config), Path('/model'))


def test_self_draft_has_no_added_model_and_rejects_other_draft():
    expected = engine_config.DEFAULT_SPECULATION
    assert launch.speculation(None, Path('/model')) == expected
    assert 'model' not in expected
    config = dict(expected, model='/model')
    assert launch.speculation(json.dumps(config), Path('/model'))['model'] == engine_config.MODEL_MOUNT
    config['model'] = '/different'
    with pytest.raises(ValueError):
        launch.speculation(json.dumps(config), Path('/model'))


@pytest.mark.parametrize('arguments', [
    ['--enforce-eager'], ['--speculative-config', '{}'], ['--kv-cache-dtype', 'bf16'],
    ['--enable-lora'], ['--max-model-len', '1048577'], ['--max-num-seqs', '9'],
    ['--gpu-memory-utilization', 'NaN'], ['--max-num-seqs', '2', '--max-num-seqs', '3'],
    ['--max-model', '100'], [2], {'flag':'value'},
])
def test_unqualified_or_malformed_arguments_fail(arguments):
    with pytest.raises(ValueError):
        launch.native_arguments(json.dumps(arguments))


def test_resource_overrides_retain_graphs_speculation_and_cache():
    result = launch.native_arguments('["--max-model-len=65536","--max-num-seqs","2"]')
    assert result[result.index('--max-model-len') + 1] == '65536'
    assert result[result.index('--max-num-seqs') + 1] == '2'
    assert '--enforce-eager' not in result
    assert '--enable-prefix-caching' in result
    assert '--no-enable-log-requests' in result
    assert '--disable-log-requests' not in result
    assert 'FULL_AND_PIECEWISE' in result[result.index('--compilation-config') + 1]
    assert result[result.index('--kv-cache-dtype') + 1] == 'fp8'


def test_container_preserves_host_uid_key_ownership_and_pinned_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('SECRET_TEST_TOKEN', 'must-not-cross')
    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    args = SimpleNamespace(host='127.0.0.1', port=8041, served_model_name='public/model',
        speculative_config=None, arguments_json='[]')
    command = launch.container_command(args, tmp_path/'runs'/'run', name='container',
        labels={'owned':'yes'}, model=tmp_path/'model', key=tmp_path/'original-key',
        replay=tmp_path/'replay'/'nonces.sqlite', profile=tmp_path/'profile', devices=['0','1','2','3'])
    assert command[command.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'
    assert command[command.index('--gpus') + 1] == '"device=0,1,2,3"'
    assert command[command.index('--publish') + 1] == '127.0.0.1:8041:8000'
    assert '--privileged' not in command and '--pid=host' not in command
    assert engine_config.IMAGE in command
    assert '--allow-research' not in command
    assert 'SECRET_TEST_TOKEN=must-not-cross' not in command
    assert 'OMP_NUM_THREADS=4' in command
    mounts = [command[n+1] for n, value in enumerate(command) if value == '--mount']
    assert any('original-key' in value and 'readonly' in value for value in mounts)
    assert any('profile' in value and 'readonly' in value for value in mounts)
    assert any('dst=/cache' in value and 'readonly' not in value for value in mounts)


def inventory():
    return [dict(index=str(n), uuid=f'GPU-{n}', capability='12.0',
                 total=96*1024**3, free=94*1024**3) for n in range(4)]


def test_resource_policy_fails_before_launch_and_permits_explicit_lower_memory():
    kwargs = dict(inventory=inventory(), memory=221*1024**3)
    assert supervisor.resources(['0','1','2','3'], **kwargs)['gpu_count'] == 4
    with pytest.raises(ValueError):
        supervisor.resources(['0','1','2','2'], **kwargs)
    with pytest.raises(ValueError):
        supervisor.resources(['0','1','2','3'], inventory=inventory(), memory=220*1024**3)
    assert supervisor.resources(['0','1','2','3'], inventory=inventory(), memory=100*1024**3,
        allow_untested_memory=True)['below_startup_policy']
    with pytest.raises(ValueError):
        supervisor.resources(['0','1','2','3'], inventory=inventory(), memory=23*1024**3,
            allow_untested_memory=True)


def test_receipts_are_private_bounded_and_not_symlinks(tmp_path):
    path = tmp_path/'receipt.json'
    supervisor.save(path, {'complete':True})
    raw = supervisor.read_receipt(path)
    assert json.loads(raw) == {'complete':True}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError): supervisor.read_receipt(path, limit=1)
    with pytest.raises(ValueError): supervisor.read_receipt(path, after=path.stat().st_mtime+1)
    path.chmod(0o644)
    with pytest.raises(ValueError): supervisor.read_receipt(path)
    link = tmp_path/'symlink'; link.symlink_to(path)
    with pytest.raises(OSError): supervisor.read_receipt(link)


def test_pinned_image_digest_and_platform(monkeypatch):
    def answer(data):
        return SimpleNamespace(returncode=0, stdout=json.dumps(data))
    valid = {'id':'sha256:'+'1'*64, 'digests':[engine_config.IMAGE], 'architecture':'amd64', 'os':'linux'}
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: answer(valid))
    assert prepare.docker_image()['id'] == valid['id']
    for changed in (dict(valid, digests=[]), dict(valid, architecture='arm64'), dict(valid, os='other')):
        monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: answer(changed))
        with pytest.raises(ValueError): prepare.docker_image()


def test_stop_never_signals_unowned_container(monkeypatch):
    record = {'container_id':'c'*64, 'image_id':'sha256:'+'1'*64, 'labels':{'owner':'one'}}
    foreign = {'id':record['container_id'], 'image':record['image_id'], 'labels':{'owner':'two'}, 'running':True}
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=json.dumps(foreign)))
    monkeypatch.setattr(supervisor, 'run', lambda *a, **k: pytest.fail('must not stop foreign container'))
    with pytest.raises(ValueError): supervisor.stop_owned(record)


def test_stop_is_bounded_and_targets_exact_container(monkeypatch):
    record = {'container_id':'a'*64, 'image_id':'sha256:'+'1'*64, 'labels':{'owner':'one'}}
    called=[]
    states=iter([{'running':True}, {'running':False}])
    monkeypatch.setattr(supervisor, 'owned_container', lambda _:next(states))
    monkeypatch.setattr(supervisor, 'run', lambda command, **kwargs:called.append((command,kwargs)))
    supervisor.stop_owned(record)
    assert called == [(['docker','stop','--time','10',record['container_id']], {'timeout':25})]


def test_overlay_hashes_are_consistent():
    package=Path(__file__).resolve().parents[1]/'flashalign_native'
    pins=json.loads((package/'overlay-pins.json').read_text())
    for name, entry in pins.items():
        assert hashlib.sha256((package/'overlays'/name).read_bytes()).hexdigest() == entry['overlay_sha256']


def test_mounted_distribution_exposes_worker_plugin():
    from importlib.metadata import distributions
    root=Path(__file__).resolve().parents[1]
    matches=[item for item in distributions(path=[str(root)])
             if item.metadata['Name']=='flashalign-native']
    assert len(matches)==1
    entries=[entry for entry in matches[0].entry_points if entry.group=='vllm.general_plugins']
    assert [(entry.name,entry.value) for entry in entries]==[
        ('flashalign_native','flashalign_native.plugin:register')]
    assert entries[0].load().__module__=='flashalign_native.plugin'


def snapshot_fixture(tmp_path):
    repository=tmp_path/('models--'+engine_config.MODEL_ID.replace('/','--'))
    snapshot=repository/'snapshots'/engine_config.MODEL_REVISION
    snapshot.mkdir(parents=True); (repository/'blobs').mkdir()
    payload=repository/'blobs'/('a'*64); payload.write_bytes(b'synthetic-model-payload')
    (snapshot/'model.safetensors').symlink_to('../../blobs/'+payload.name)
    (snapshot/'config.json').write_text('{}')
    return repository,snapshot,payload


def test_hf_snapshot_preserves_links_with_only_repository_mount(tmp_path):
    repository,snapshot,payload=snapshot_fixture(tmp_path)
    (snapshot/'nested').mkdir()
    (snapshot/'nested'/'tokenizer.json').symlink_to('../../../blobs/'+payload.name)
    plan=launch.model_mount_plan(snapshot)
    assert plan=={'source':repository,'target':'/hf-model',
                  'model':'/hf-model/snapshots/'+engine_config.MODEL_REVISION}
    assert (plan['source']/Path(plan['model']).relative_to(plan['target'])/'model.safetensors').read_bytes()==payload.read_bytes()
    args=SimpleNamespace(host='127.0.0.1',port=8041,served_model_name=engine_config.MODEL_ID,
        speculative_config=json.dumps(dict(engine_config.DEFAULT_SPECULATION,model=str(snapshot))),arguments_json='[]')
    command=launch.container_command(args,tmp_path/'run',name='fixture',labels={},
        model=snapshot,key=tmp_path/'key',replay=tmp_path/'replay'/'nonces',
        profile=tmp_path/'profile',devices=['0','1','2','3'],model_layout=plan)
    assert command[command.index('serve')+1]==plan['model']
    assert json.loads(command[command.index('--speculative-config')+1])['model']==plan['model']
    mounts=[command[index+1] for index,item in enumerate(command) if item=='--mount']
    assert f'type=bind,src={repository},dst=/hf-model,readonly' in mounts
    assert not any('src='+str(tmp_path)+',' in value for value in mounts)


def test_materialized_and_internal_relative_link_models_keep_original_mount(tmp_path):
    model=tmp_path/'materialized'; model.mkdir()
    (model/'weight').write_bytes(b'fixture'); (model/'alias').symlink_to('weight')
    assert launch.model_mount_plan(model)=={'source':model,'target':engine_config.MODEL_MOUNT,
                                           'model':engine_config.MODEL_MOUNT}


@pytest.mark.parametrize('kind',['absolute','escape','chain','directory','dangling','extra-repository-file'])
def test_unsupported_model_links_fail_before_container_create(tmp_path,kind):
    repository,snapshot,payload=snapshot_fixture(tmp_path)
    entry=snapshot/'model.safetensors'
    if kind=='absolute':
        entry.unlink(); entry.symlink_to(payload)
    elif kind=='escape':
        outside=tmp_path/'outside'; outside.write_bytes(b'private'); entry.unlink();entry.symlink_to('../../../outside')
    elif kind=='chain':
        payload.unlink();other=repository/'blobs'/('b'*64);other.write_bytes(b'fixture');payload.symlink_to(other.name)
    elif kind=='directory':
        (snapshot/'directory').symlink_to(repository/'blobs',target_is_directory=True)
    elif kind=='dangling':
        payload.unlink()
    elif kind=='extra-repository-file':
        (repository/'credentials').write_bytes(b'do-not-mount')
    with pytest.raises(ValueError):launch.model_mount_plan(snapshot)
