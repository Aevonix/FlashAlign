# Managed engine setup

Default setup prepares the engine, loads a profile, creates a signing key and
checks activation. It can also configure Hermes or OpenClaw.

Download the checkpoint and any required draft model before setup. The
[model download guide](model-downloads.md) provides pinned commands for each
starter model and its supported engine formats.

```sh
bash ./install.sh --auto
```

Run this on the model host. Add `--engine-only` for direct API access. The wizard
looks for local Hugging Face snapshots and asks for a model directory if needed.
For Verified contacts mode, it reuses existing approvals or asks which contact
to approve.

To supply the model location directly:

```sh
bash ./install.sh --auto --model-path /models/my-model
```

Use `--yes` to accept the final setup prompt. For example, this prepares a
standalone Qwen engine and stops before starting it:

```sh
bash ./install.sh --engine-only --backend vllm \
  --model-path /models/qwen38-27b --yes --no-restart
```

## Engine selection

`--backend auto` preserves a matching active backend, uses a supplied TensorFold
package when present, or selects an included integration for the local model
and profile. It checks declared checkpoint identities and formats before choosing
a backend. GGUF files select llama.cpp. Choose another engine explicitly with
`--backend`.

| Engine | Included integration | Model and configuration |
| --- | --- | --- |
| vLLM | Pinned 0.32.0 package and FlashAlign plugin | Qwen3.8-27B BF16, one GPU, eager execution, ngram speculation |
| SGLang | Pinned 0.5.20 source patch | Qwen3.8-27B BF16, one GPU, eager execution, DFlash2 |
| SGLang | Pinned 0.5.20 normalized-output integration | GLM 5.3 Flash FP8, four local SM120 GPUs, native EAGLE, BF16 KV |
| SGLang | Pinned 0.5.20 writer-group integration | Qwen Flash Next FP8, two GPUs, native MTP, PLE tables in host RAM; requires an operator-supplied profile |
| vLLM | Pinned 0.32.0 projection integration | Gemma 4 26B A4B FP8, one GPU, ngram |
| SGLang | [Pinned 0.5.20 projection integration](../engines/sglang-gemma4/README.md) | Gemma 4 26B A4B FP8, one GPU, NGRAM with three draft tokens |
| vLLM | [Pinned native container and tensor operator](../engines/vllm-deepseek-native/README.md) | DeepSeek V4.1 Flash 1017 official FP8, four SM120 GPUs, adaptive DSpark7, FULL decode graphs and eager prefill |
| llama.cpp | Upstream 0.8.0, commit `8642541`, source patch and profile converter | Qwen3.8-27B and Gemma 4 26B A4B Q8_0, one GPU; Qwen `draft-simple`, Gemma ngram or its official MTP draft |
| TensorFold | Compiled 1.1.9 and 1.2.0 bundles for Linux x86_64 and aarch64 | Model, GPU, operator and draft requirements come from the selected bundle; see [TensorFold setup](native-tensorfold.md) |

Each integration declares its supported model family, operator and serving
format. See [validation](validation.md) for runtime test results and known
model-quality limits.

Setup carries the engine's declared thinking format and supported effort levels
into OpenClaw. An explicit `compat` object on the matching existing model takes
precedence. Packages without a declaration retain OpenClaw's existing behavior.

The pinned GLM Full and Flash templates support low, high and max reasoning.
They do not support disabling thinking. Setup rejects a saved OpenClaw off
setting for these models.

```sh
bash ./install.sh --engine-only --backend sglang \
  --model-path /models/qwen38-27b --drafter /models/qwen38-dflash2

bash ./install.sh --engine-only --backend llamacpp \
  --model-path /models/qwen38-27b-q8_0.gguf --drafter /models/draft.gguf

bash ./install.sh --engine-only --backend sglang \
  --model-path /models/gemma4-26b-a4b-fp8
```

Gemma's SGLang integration uses NGRAM with three draft tokens and no external
draft model. Its tested configuration uses eager execution, non-overlapping
scheduling and two concurrent requests. Existing speculative settings must
match the integration's declared capabilities.

## Included profiles and engine packages

GLM 5.3 Full and Flash, Qwen3.8-27B and Gemma 4 26B A4B compact starter
profiles are included in the checkout. DeepSeek V4.1 Flash 1017 includes its
native manifest and a pinned public download descriptor. Setup retrieves the
1.09 GB native payload when needed, verifies its size and SHA-256, and caches it
outside the checkout.
For offline setup, place the verified payload beside its manifest before
preparation. See [the DeepSeek profile](../profiles/deepseek41-flash1017-v1/README.md)
for the exact file and source terms.

For TensorFold, managed setup can retrieve a matching runtime package from
FlashAlign's public release assets over HTTPS. The catalog pins each asset's
size, SHA-256 digest and complete file inventory. Setup verifies those checks
before staging and rechecks cached files
before reuse. These TensorFold downloads contain engine code, separately from
the DeepSeek native profile asset.

For offline setup or a separate compatible package, use `--engine-bundle`.
If no asset matches the model and platform, setup asks for a local package.
TensorFold bundles target Linux x86_64 or aarch64 and declare their GPU
architectures. The catalog contains the combinations listed in
[compatibility](compatibility.md). Multi-host Spark preparation and activation
use explicit owned rank directories and SSH host-key verification. See
[TensorFold setup](native-tensorfold.md).

You can also supply local packages:

```sh
bash ./install.sh --auto --model-path /models/my-model \
  --profile /private/my-profile --engine-bundle /private/my-engine-bundle
```

TensorFold setup checks CPU architecture, CUDA targets, source revision and the
checkpoint's configuration, weight-index and quantization fingerprints. DeepSeek
native-matrix profiles require a matching 1.2.0 bundle and its declared EXL3 base.

TensorFold packages may declare `input_modalities`, such as `["text"]` or
`["text", "image"]`. This describes the serving path, rather than the original
checkpoint's architecture. Setup validates the declaration and uses it for
Hermes and OpenClaw image configuration. A text-only declaration clears a
previous model's native vision flag while preserving an explicit auxiliary
image route. A missing declaration remains unverified.

Included engine integrations can declare different capabilities with
`input_modalities_by_model_type`. The standard SGLang integration declares
`qwen3_5` as `["text", "image"]` and `qwen3_5_text` as `["text"]`. A typed
declaration applies only to its listed model types; unlisted types remain
unverified even if a general declaration is also present. Setup does not infer
image support from an architecture name.

The [starter profiles](../profiles/README.md) are experimental.

## What setup does

1. Checks the model metadata, profile files and engine compatibility.
2. Copies the integration into private FlashAlign state and prepares a separate
   engine installation. Source patches use a checked upstream revision; model
   files remain unchanged.
3. Prepares the declared runtime, using a managed Python environment or the
   pinned native container, and installs required dependencies.
4. Creates or reuses a deployment signing key and stages the profile.
5. Starts the model workers and checks standard and signed profile requests.
6. Applies the harness configuration and requests the selected gateway restarts.

Use `--devices 0` to select a GPU explicitly. A TensorFold package may support
more GPUs, such as `--devices 0,1,2,3`. Automatic selection uses
visible GPUs or devices with less than 5% of their memory occupied. A machine
still needs suitable NVIDIA hardware and drivers. Integrations built from source
also need their declared compilers and CUDA build tools; prebuilt TensorFold
bundles do not require a CUDA build environment. The native DeepSeek backend
requires Docker Engine and NVIDIA Container Toolkit. Its pinned image is
`ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1b10ec7b853a495e66b7dbf15ff54f62741e444dea0b44e16736a10a9`.
Setup can pull that image when dependency installation is enabled. It does not
download a base model checkpoint.

Setup checks backend-specific interpreter requirements before preparation.
The Gemma and GLM Flash SGLang integrations require Python 3.12 or newer;
FlashAlign setup itself supports Python 3.10. Without an
explicit `--engine-python` or `engine.python`, setup can select a compatible
interpreter already installed locally. An incompatible explicit choice stops
setup before dependency installation.
It does not install a system Python.

Triton also needs matching Python development headers and a C compiler. Setup
reuses installed headers or an explicit `engine.environment.CPATH`. With default
dependency installation on an APT-based host, it can download the matching
header package and unpack it inside the deployment directory. It does not
install that package into the operating system. On other hosts, install the
matching development headers before activation.

Setup preserves a matching active engine's speculative configuration. It
rejects an unsupported method or a conflicting requested draft. New vLLM
Qwen deployments use ngram speculation. DeepSeek V4.1 Flash on
the native vLLM backend uses the stock model's adaptive DSpark with seven
draft tokens. Qwen3.8-27B on SGLang needs a
local DFlash2 model; Qwen Flash Next uses its own native MTP weights. llama.cpp
uses `draft-simple` for a local Qwen GGUF draft. Gemma defaults to
`ngram-simple`, which needs no separate draft file. Supplying the official
Gemma MTP GGUF selects `draft-mtp`; an existing supported method is preserved.
TensorFold
requirements come from its package. Use `--drafter /models/my-draft` or
`engine.draft_model` in a plan where applicable. No model or draft weights are
downloaded by setup.

An explicitly selected GPU can run a separate engine using the same checkpoint.
Setup treats another engine as independent only when its declared GPU visibility
and the driver's process ownership agree and do not overlap the selected GPUs.
Unknown placement keeps the speculative-configuration check in place.

Setup reports the selected engine context and checks existing Hermes and
OpenClaw context and output limits before installing or changing their routes.
The supported Hermes core also requires at least 64,000 tokens when its
configuration does not specify a context limit. Managed setup checks that
minimum before installation. Direct engine setup has no Hermes requirement.
An existing endpoint that omits context metadata is reported as unverified;
setup never guesses its capacity from the model name.
Setup stops if the engine cannot meet the harness's existing limits. An explicit
limit requires a known engine context. OpenClaw receives the selected engine
context in its model descriptor.
The Qwen3.8-27B llama.cpp default is 65,536 tokens per sequence, using 131,072
total context across two parallel slots. Its context and parallel overrides are
checked together so setup reports the same per-sequence cap that is launched.

An engine that supports omitted output budgets can declare an OpenClaw request
policy in its metadata:

```json
"openclaw_request_policy_by_model_type": {
  "example_model_type": {
    "output_budget": "remaining_context",
    "supports_thinking_off": false
  }
}
```

Each field is optional. The declaration applies only to the selected model
type. `remaining_context` leaves the request's output budget absent unless a
user or task supplied one. Catalog capacity stays available to OpenClaw; it is
not sent as a default output request. Explicit task budgets and actual reported
model limits remain distinct. Existing integrations without a declaration keep
their prior behavior.

`supports_thinking_off: false` rejects a saved off selection before setup and
rejects a later explicit off request before native transport. It does not pick
a reasoning level when the user omitted one or change the model's template.
Supported reasoning levels come from the model's separate compatibility
declaration.

These controls cover OpenClaw's regular provider stream. Its dedicated device
worker-local inference path does not use that stream wrapper and forces a
catalog output allowance. Setup rejects a required worker profile using that
path when `remaining_context` is declared. Optional worker placements need
their own integration before using these controls.

For vLLM, use complete option names in engine arguments. Hyphen and underscore
spellings are equivalent; values and nested JSON field names are preserved.
Shortened context or protected options are rejected before launch.

For a serving lane with declared image support, setup configures the selected
Hermes model's vision capability and a default automatic image route, and adds
image input to OpenClaw's model descriptor. Explicit text or auxiliary-image
routes are preserved and reported. A previous model's capability flag is updated
to match the selected lane. Text-only lanes, including this llama.cpp integration,
do not acquire image support from the base model's architecture. Missing endpoint
modality metadata is reported as unverified.

Hermes setup selects a dedicated local FlashAlign provider so an existing named
provider cannot override the new proxy URL. The previous provider definition is
preserved. Ordinary generation settings carry across; upstream credentials stay
in the protected FlashAlign route configuration. Repeating setup reuses its
managed provider, while an unrelated provider with the same name is preserved.

## Prompt reuse

The included engines reuse matching conversation prefixes in memory. Owner
entries are separated by loaded profile and strength; normal-mode entries remain
separate. Owner entries are never written to the engine's disk cache. Reloading
the model or profiles starts a new engine with an empty in-memory cache.

Prompt reuse consumes the engine's cache memory budget, independently of
profile storage. The DeepSeek native profile also needs GPU storage for its
alternate matrices and graph workspaces. Speculative decoding stays enabled.

## Process supervision

Managed activation requires Linux pidfd and child-subreaper support. Each
engine command runs under a supervisor that tracks its workers by PID, user and
process-start identity, including workers whose parent exits.

If startup or verification fails, setup cleans up its verified workers and keeps
the existing harness configuration. Incomplete cleanup preserves `running.json`
and is reported for review. Setup does not stop unrelated processes or overwrite
an existing engine. Logs and supervisor receipts stay in the private deployment
directory.

## Agent preparation and activation

An agent can prepare everything without starting a model or restarting a gateway:

```sh
bash ./install.sh --config /private/setup-plan.json --apply
```

Example plan:

```json
{
  "harnesses": ["hermes"],
  "model": "publisher/model",
  "profile": "/private/profile",
  "contacts": [
    {"channel": "whatsapp", "account": "default", "contact": "+15551234567"}
  ],
  "tools_default": true,
  "engine": {
    "kind": "tensorfold",
    "model_path": "/models/my-model",
    "draft_model": "/models/my-draft",
    "bundle": "/private/engine-bundle",
    "devices": ["0", "1", "2", "3"],
    "install_dependencies": true
  }
}
```

For another backend, set `engine.kind` to `vllm`, `sglang` or `llamacpp` and
omit `bundle`. vLLM accepts a `speculative_config` object. SGLang accepts a local
`draft_model`; llama.cpp accepts a GGUF `model_path` and optional GGUF
`draft_model`. Native tuning flags go in an `arguments` string array. Setup
reserves flags controlling model identity, listeners and speculation so a
native argument cannot override them. `devices` must match the selected variant:
Qwen Flash Next on SGLang requires two GPUs. DeepSeek on vLLM and GLM Flash on
SGLang require four GPUs; Qwen3.8-27B and Gemma use one.
TensorFold's required ranks and host layout come from the selected bundle;
see the [hardware table](compatibility.md#engines-and-models).

For TensorFold, `engine.arguments` accepts only one positive `--context` cap,
for example `["--context", "8192"]` or `["--context=8192"]`. The same cap is
checked during preparation and passed to the engine where the bundle permits
it. Fixed geometry still applies to a selected bundle, including the legacy
DeepSeek projection bundle's 65,536-token context, 131,072-token pool and four
ranks. That bundle does not support the native DeepSeek starter. Other
TensorFold overrides are rejected before setup writes files;
model, profile, listener, parallelism and speculative settings remain managed.
TensorFold setup uses a verified native runtime bundle.

Variants with a pinned quantized model verify configuration, quantization and
index or single-shard header fingerprints. These checks identify the tested
layout; they do not replace the model distributor's complete download hashes.
The wizard preserves supplied profile license notices when staging it.

Native vLLM DeepSeek preparation verifies every pinned checkpoint payload and keeps
a resumable receipt outside the read-only model folder. The first pass reads
about 510 GB; it does not download, copy or convert the base weights. Normal
Hugging Face snapshot symlinks are supported. The native profile is a separate
verified public asset. Its operator keeps the stock and edited WO-B bindings
immutable and selects between them per request or row. Strength is fixed at
1.0; projection-only integrations cannot load it.

The native vLLM launcher preserves adaptive DSpark7 and the declared graph
settings. Its default context is 1,048,576 tokens. See the
[DeepSeek integration](../engines/vllm-deepseek-native/README.md) for resource
requirements and arguments, and [validation](validation.md) for tested workloads.
Its Hermes setup preserves disabled thinking through DeepSeek's
`enable_thinking=false` option.

Existing DeepSeek output-projection profiles use the
[legacy vLLM integration](../engines/vllm-deepseek/README.md) or a matching
TensorFold bundle. These integrations cannot load the native starter.

An existing prepared installation can be reused without installing packages:
set `engine.python` to the compatible environment and leave
`install_dependencies` false. SGLang also needs a clean pinned `source`
checkout; llama.cpp needs the path to its prepared `installation.json` in
`engine.installation`.
The native DeepSeek starter requires its pinned container and overlay.

The result includes the activation command. When starting the engine and
restarting the gateways is authorized:

```sh
bash ./install.sh --config /private/setup-plan.json --activate
```

`--activate --no-restart` starts and verifies the engine and saves the harness
configuration, while leaving the gateway restarts for later.

To stop a managed engine, use the deployment file from the setup result:

```sh
bash ./install.sh --stop-engine /private/deployment.json
```

The stop operation verifies recorded process identities and only signals
workers created by this installer.
