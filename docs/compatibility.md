# Compatibility

## Product and runtime names

FlashAlign is the product; Owner Mode is the profile-selection feature. The
native plugin/provider ID is `owner-mode`, configuration lives under
`~/.config/owner-mode/`, and the engine protocol uses `owner_mode`.

In Verified contacts mode, Aevonix FlashAlign uses the harness's native channel
and run metadata. The adapter must identify an incoming contact and bind it to
a specific turn. Missing or ambiguous attribution uses the standard model.

Always On applies the configured profile to every request routed through
FlashAlign, including unverified, group and synthetic inputs, CLI and scheduled
work, all tools and all subagents. It overrides per-tool alignment flags and
persists across restarts until the operator changes the mode. The host still
controls message admission and tool execution permissions. Callers do not
become verified.

Custom harnesses and ordinary OpenAI clients can use the
[Always On gateway](direct-engine.md#always-on). All intended model requests must
reach that gateway or a configured FlashAlign route. Requests sent to another
provider or directly to the raw upstream engine are outside its policy.

## Harnesses

| Harness | Interface for Verified contacts and inherited access |
| --- | --- |
| Hermes | Native transport reference, ingress authorization, session route, current message ID, request and tool-execution hooks; native render/merge seams for media and grouped input |
| OpenClaw | Native `message_received` and `before_dispatch`, protected sender/account/channel fields in model hooks, isolated direct-message sessions, tool lifecycle hooks and provider `onPayload` |

The OpenClaw integration targets the 2026.10.8 plugin interface. Its runtime
requires Node 24.22.1+ on the 24.x line or 26.12.1+ on the 26.x line. Python 3.10+
is required for the shared core and setup wizard.

After setup, check message routing, profile selection, revocation and delegation
through your installed harness. See [validation](validation.md) for test coverage.

Verified contacts mode requires OpenClaw to isolate direct messages with
`per-account-channel-peer` or `per-channel-peer`. Setup preserves those scopes.
It backs up the configuration before upgrading a shared scope to
`per-account-channel-peer`. A session name or a CLI routing option alone does
not authorize Owner Mode.

## Channels

Verified contacts mode uses an installed adapter registry. WhatsApp and RCS
direct messages are enabled by default. SMS and Telegram require explicit
opt-in; Telegram uses native numeric user IDs. Each requires the harness's
trusted transport metadata and an installed messaging transport. Group chats
and synthetic events use the standard model in this mode.

Explicitly authorized local terminal sessions use a separate adapter with
expiring, revocable grants. A CLI label, session name or prompt cannot create
that authorization. See [channel and terminal setup](channels.md) for the
commands and extension interface.

On supported messaging transports, the adapters handle text, voice notes,
photos and other attachments, forwarded messages, quoted replies, and queued
message groups. Hermes observes
the host's native rendering and merge operations. OpenClaw binds the prepared
request in the provider payload hook. Neither adapter infers identity from a
transcript, a quoted author or model-generated content.
The local terminal adapter is text-only in V1.

For composed input in Verified contacts mode, Hermes and the message-based
bridge require each member to have a distinct native message ID and the same
verified channel, account and contact. Groups are limited to 64 members.
OpenClaw's native session path requires protected sender metadata for each
independently admitted run.
In Verified contacts mode, missing, mixed or replayed identities use the standard
model. A host that cannot provide a trusted binding for rendered messages can
still authenticate unmodified text. Transformed media and composed groups use
the standard model on that host.

Multi-account use requires matching native account IDs at ingress and turn
dispatch; a harness that omits account attribution supports its default account
only. See [message binding](message-bindings.md) for adapter details.

## Tools and child work

In Verified contacts mode, Hermes scopes synchronous and asynchronous tool
execution. OpenClaw binds native `sessions_spawn` children at the awaited
`subagent_spawned` event when exactly one parent tool matches. Those
children can continue after the creating tool returns. Typed `inter_session`
origins also work while their creating tool is active. Unattributed or ambiguous
children use the standard model.

When OpenClaw runs concurrent tools in a Verified contacts turn, model calls
use the standard model while any active tool's alignment flag is off. Tool
flags affect model selection; the harness still decides whether a tool may
execute.

Always On selects the profile for every routed tool and child request, including
unattributed work and tools whose alignment flag is off. The host's tool
execution permissions still apply.

## Engines and models

The supported API is OpenAI Chat Completions plus read-only `/models` discovery.
The engine must implement the [Owner Mode API](engine-contract.md), validate
the signing key and load the correct profile for the exact model revision and
serving format. The [starter pack](../profiles/README.md) includes five runtime
profiles; additional profiles can be supplied by the operator.

Qwen3.8-27B and Gemma 4 26B A4B have paths on TensorFold, vLLM, SGLang and
llama.cpp. The [starter pack](../profiles/README.md) lists each model's supported
engines; [validation](validation.md) records the test results.

An engine can reuse a profile when it supports that profile's model revision,
serving format, operator and output sites. Compact profiles also require a
supported basis rank; native tensor profiles require their declared storage
format and parallel layout.

For compact profiles, setup checks each backend's `maximum_rank` before
installation. The current GLM Flash and Qwen Flash Next SGLang adapters accept
rank one; the other packaged vLLM, SGLang and llama.cpp adapters accept ranks up
to 128. Site coverage and operator support differ by adapter.

The llama.cpp integration uses upstream version 0.8.0 at commit
`8642541b255bfb5dc39cd351a178be09013a55d6` and the declared Q8_0 GGUF
checkpoints. Gemma setup defaults to ngram speculation and can preserve its
official MTP draft. Other GGUF files need a matching compatibility declaration.
See the [llama.cpp guide](../engines/llamacpp/README.md).

The four compact starters have TensorFold 1.1.9 paths, with model-specific
builds and presets. The DeepSeek native starter has pinned vLLM and TensorFold
1.2.0 paths. TensorFold bundles support Linux x86_64 or aarch64 and declare the
required GPU architecture and rank count. Setup checks those requirements.
The following hardware was used to test the bundled configurations:

| Model | Checkpoint format | Tested hardware |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017, TensorFold 1.2.0 | Declared LibertAIDAI EXL3 3.5-bpw base | Four RTX PRO 6000 GPUs, Linux x86_64, SM120 |
| Qwen3.8-27B | MLX-format 4-bit | One DGX Spark, Linux aarch64, SM121 |
| Qwen3.8-27B | EXL3 2.0-bpw | One RTX PRO 6000, Linux x86_64, SM120 |
| Gemma 4 26B A4B | MLX-format 4-bit | One DGX Spark, Linux aarch64, SM121 |
| GLM 5.3 Flash | NVIDIA NVFP4 | Four DGX Sparks, Linux aarch64, SM121 |
| GLM 5.3 Full | EXL3 2.75-bpw | Four DGX Sparks, Linux aarch64, SM121 |

MLX-format describes the checkpoint files; TensorFold runs them on NVIDIA GPUs
under Linux. Use the
[model download guide](model-downloads.md) to select the pinned files for your
model and backend.

The [pinned native vLLM backend](../engines/vllm-deepseek-native/README.md)
for DeepSeek's `deepseek41-flash1017-v1` starter requires
the official FP8 checkpoint at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, Linux x86_64 and four SM120 GPUs.
It uses adaptive DSpark with seven draft tokens, FULL decode graphs and eager
prefill. The native operator selects 26 WO-B matrices and 26 scale tensors at
strength 1.0. Setup retrieves and verifies the 1.09 GB profile payload from a
public release asset.

TensorFold 1.2.0 uses the same native profile payload with the declared
LibertAIDAI EXL3 base at revision
`04a492fc3b4d25ddbab8f2d2f07f235ec77c9010`, strength 1.0 and four ranks on
SM120 GPUs. Its base quantization and arithmetic differ from the FP8 vLLM path.
Both current DeepSeek packages enforce this four-GPU layout.

The DeepSeek TensorFold preset has a 65,536-token context and a 131,072-token
pool. JSON Schema and guided output are unsupported. Older output-projection
profiles use their matching TensorFold or vLLM packages; those packages cannot
load the native starter. The older TensorFold preset supports automatic DSML
tool calls; forced tool selection and image input are unsupported.

GLM Full's TensorFold preset uses an 8,192-token context, two concurrent
sequences and native MTP depth 4. It supports automatic tool calls. Forced tool
selection, JSON Schema and image input are unsupported on this path.

Before changing harness routing, the wizard checks model and profile
compatibility and verifies signed profile and standard-mode requests. See
[TensorFold setup](native-tensorfold.md) for remote ranks, SSH forwarding and
DeepSeek Engram prerequisites.

Transcription, file decoding and vision depend on the configured model and
harness capabilities. Advertise image input only for a model that supports it.
Text-only models can receive the harness's prepared transcription or attachment
context.

Always On also covers custom tools that send model requests through the supported
Chat Completions route. Responses API, Anthropic Messages and other model
transports need an adapter to that route; they are outside the V1 API interface.

Managed inference supports Linux with NVIDIA GPUs. macOS clients can connect to
a remote engine; local macOS inference is not supported.
