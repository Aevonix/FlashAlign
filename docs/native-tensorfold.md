# TensorFold setup

FlashAlign manages TensorFold 1.1.9 and 1.2.0 bundles on Linux x86_64 and
aarch64. A bundle must match the host's CPU and GPU architectures. Install the
NVIDIA driver before setup.

Setup stages the checked native executable, runtime libraries, matching source
archive, profile and unique deployment signing key.

The download catalog lists the available runtime bundles and their complete
checksums. When no matching bundle is listed, provide its local directory with
`--engine-bundle`. Profiles come from the local starter pack or an explicit
profile directory.

The DeepSeek starter runs on the [pinned native vLLM backend](../engines/vllm-deepseek-native/README.md)
or a matching TensorFold 1.2.0 native-matrix bundle. TensorFold uses the pinned
LibertAIDAI EXL3 serving base and the same unchanged native profile payload.
Strength is fixed at 1.0 and tensor parallelism at four SM120 GPUs. The bundle
must declare this operator and compatible base; older projection bundles
cannot load it. It supports chat and completion requests; `response_format`
and guided output are unsupported. See
[model downloads](model-downloads.md#deepseek-v41-flash-1017-on-tensorfold-120).

## One host

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path /models/my-model --engine-bundle /packages/native-runtime \
  --devices 0 --no-restart
```

The bundle declares the compatible model families, exact source revision,
profile operators, supported rank counts, context default and speculative
methods. Checkpoint fingerprints must match one complete profile declaration.
The installer rejects a changed binary, missing library, wrong ELF architecture,
or incompatible GPU target before starting the engine.

## Several Sparks

Use SSH config names or addresses that already work with host-key verification.
The controller needs Python 3.10+, SSH access, and the runtime bundle. Each model
host needs Python 3.10+ for the setup supervisor, the NVIDIA driver, and its local
model checkpoint. Python does not run inference.
Replace the example worker names, documentation address and `eth0` interface
below with the names, reachable address and network interface of your hosts.

```sh
bash ./install.sh --auto --backend tensorfold \
  --hosts worker-a,worker-b,worker-c,worker-d \
  --master-address 192.0.2.10 --network-interface eth0 \
  --model-path /models/glm --engine-bundle /packages/native-aarch64 \
  --no-restart
```

The same model path may be used on every host. An agent plan can specify different
paths or devices:

```json
{
  "engine_only": true,
  "model": "publisher/model",
  "profile": "glm53-full-v1",
  "engine": {
    "kind": "tensorfold",
    "bundle": "/packages/native-aarch64",
    "master_address": "192.0.2.10",
    "master_port": 29723,
    "network_interface": "eth0",
    "hosts": [
      {"host": "worker-a", "model_path": "/models/glm", "devices": ["0"]},
      {"host": "worker-b", "model_path": "/models/glm", "devices": ["0"]},
      {"host": "worker-c", "model_path": "/models/glm", "devices": ["0"]},
      {"host": "worker-d", "model_path": "/models/glm", "devices": ["0"]}
    ]
  }
}
```

Preparation checks the model metadata and GPU capability on every host. It
stages profiles and required files through encrypted SSH input into private
rank directories. Only commands that need a signing key receive one. Keys
never appear in arguments or setup output. The leader authenticates API
requests; every rank loads the same profile.
GLM Flash also receives the key-file path during startup. Preparation does not
start a model or restart a harness.

Activation starts followers, then the leader and its SSH forward. Startup
allows 30 minutes by default; bundles can set `startup_timeout` from 60 to
7,200 seconds. The DeepSeek 1.2.0 bundle allows 15 minutes. Set
`engine.startup_timeout` to an integer in the plan to override this value.
DeepSeek ranks use the same allowance to connect to their peers.

Each rank has a process supervisor and a private receipt. Setup verifies
standard and signed profile requests before changing harness routing. If
activation fails, it stops only those recorded ranks. An occupied service is
never stopped to make room.

For remote hosts, setup keeps an owned SSH forward running on the controller and
configures clients with its localhost URL. The forward uses the same verified SSH
host and is supervised and stopped with the deployment. It must remain running
while a harness uses that URL. Set `api_transport` to `direct` in an agent plan
only when the controller can reach the leader directly. The rendezvous address
must be reachable from every rank. The selected network interface must exist on
every host; configuring drivers, SSH access and the cluster network remains an
operator task.

A host entry may set `state_root` to a writable, persistent private directory
when its home partition has insufficient space. Runtime files on a temporary
filesystem need preparation again after reboot. Model files may be on a shared
read-only mount.

Native replay protection persists outside the versioned runtime directory.
Setup selects a stable private directory for the deployment key and passes it
with `--owner-replay-dir`. Keep this state across engine and profile upgrades;
stopping a deployment does not remove it. See the [security model](security.md)
before resetting state or moving a deployment key.

## Speculative decoding

The bundle declares the supported speculative method. Setup preserves a matching
active external drafter, MTP depth and parallel setting. An incompatible active
configuration stops setup; the installer does not disable speculation. Native
GLM uses its MTP weights. Each bundle declares whether MTP depth is adjustable.
Flash supports up to three drafts; Full defaults to four. The Full preset uses
an 8,192-token context, two concurrent sequences and an 8 GiB host prompt cache.
It supports automatic tool calls; forced tool selection, JSON Schema and image
input are unsupported. Setup rejects unsupported speculative settings before
staging. Qwen may require an external DFlash checkpoint. DeepSeek uses its
declared DSpark assets. See [validation](validation.md) for runtime test results.

## DeepSeek Engram tables

DeepSeek needs its original Engram tables on every rank, separately from the
EXL3 model. Setup detects the pinned original checkpoint in the Hugging Face
cache, or accepts `--engram-source /models/deepseek-engram` (`engram_source` in
a plan). It checks config and index hashes, tensor encodings and shapes, and
complete shard sizes before staging files. A prepared per-rank table directory
can instead use `--engram-dir` (`engram_dir` in a plan); its headers and sizes
must match the model and rank geometry.

Only the original config, index, and two Engram shards are needed:

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  config.json model.safetensors.index.json \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /models/deepseek-engram
```

These two shards total about 203 GB. They remain on the model host or its shared
read-only storage; they are not copied into the runtime bundle. Retain the pinned
download's payload-integrity record; setup checks headers and sizes.

### Legacy projection bundles

Older TensorFold bundles require a matching output-projection profile and
cannot load the current native starter. Their DeepSeek preset fixes the context
at 65,536 tokens, pool at 131,072 tokens, one live slot and four ranks. Changing
that geometry requires matching generated assets; the wizard preserves the
declared context and speculative settings.

## Bundle contract

A bundle uses manifest version 2 with `engine: "tensorfold-native"`,
`runtime_version: "1.1.9"` or `"1.2.0"`, an exact `source_revision`, `platform`, and
`cuda_architectures`. `durable_owner_replay: true` is required; the installer
rejects bundles without persistent replay protection before staging them.
It includes a complete SHA-256 and size inventory of its
`bin/`, `lib/`, `assets/`, `sources/`, and `licenses/` files plus `LICENSE` and
`NOTICE.md`. `source_archive` identifies the matching modified engine source.
Profiles and NVIDIA drivers cannot appear in the runtime archive.

Native-matrix bundles require TensorFold 1.2.0 and explicitly declare
`profile_schemas`, `supported_profile_requires`, `supported_profile_strengths`,
`profile_target_model`, and `compatible_profile_bases`. The profile keeps its
original target pin; compatible serving bases are declared by the runtime.

`binary` names the primary executable. `rank_transport` is `cli`, `glm_env`, or
`dsv41_env`, according to the native family interface. DeepSeek additionally
names its `follower_binary` and `assets_directory`.

Maintainers build a reproducible runtime archive with
[build-native-bundle.py](../tools/build-native-bundle.py). Its source archive comes
from the exact Git commit. `--release-tag` together with `--catalog-entry` writes a
complete download entry. Include each bundled library's license or notice and
any required source. Auxiliary runtime kernel sources and notices appear in
the manifest alongside the matching engine source.
