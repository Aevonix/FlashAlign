# Use FlashAlign directly

Hermes and OpenClaw are optional. A trusted application can call a compatible
FlashAlign engine using its deployment signing key.

Prepare the engine and local profile without configuring a harness:

```sh
bash ./install.sh --engine-only --backend vllm --model-path /models/qwen38-27b
```

The wizard provisions the profile and a unique deployment key, starts the engine
and verifies standard and signed requests. Use `--no-restart` to prepare the
deployment and stop before starting it. An agent can use a noninteractive plan:

```json
{
  "engine_only": true,
  "model": "Qwen/Qwen3.8-27B",
  "profile": "qwen38-27b-v1",
  "engine": {
    "kind": "vllm",
    "model_path": "/models/qwen38-27b",
    "devices": ["0"],
    "install_dependencies": true
  }
}
```

Save it as an owned `0600` file. Run `--config /private/plan.json --apply` to
prepare, then `--config /private/plan.json --activate` when ready to start it.
This path does not configure contacts or restart a chat gateway. Successful
activation writes paths and model settings to `~/.config/flashalign/client.json`.
The signing key stays in its separate file.

See [engine selection](engine-setup.md#engine-selection) for SGLang, llama.cpp
and TensorFold plans. `--yes` accepts the final prompt when every required
choice is already supplied. `--config ... --apply` remains the noninteractive
agent path that stops before starting the engine.

## Python client

Install the client from this checkout:

```sh
python -m pip install .
```

```python
from pathlib import Path
from flashalign import Client

with Client.from_config(Path.home()/".config/flashalign/client.json") as client:
    normal = client.chat([{"role": "user", "content": "Explain this code."}])
    aligned = client.chat(
        [{"role": "user", "content": "Explain this code."}],
        profile="your-installed-profile",
        scale=1.0,
    )
```

In the default `verified` mode, direct calls use standard mode unless your trusted
application explicitly selects a profile. The client signs the exact bytes sent
and checks the engine's effective profile response. It raises `ProfileNotActive`
if the requested profile was not activated. `stream_chat` provides the same check
before exposing streamed events.

### Always On

Use `--mode always_on` with engine-only setup, or set `"mode": "always_on"` in
the private setup plan. Setup prepares a local FlashAlign gateway that applies
the configured profile to every request at its advertised OpenAI-compatible URL.
Ordinary OpenAI clients, tools and subagents can use that URL without signing
requests or using the FlashAlign SDK:

```sh
bash ./install.sh --engine-only --backend vllm --model-path /models/qwen38-27b --mode always_on
```

`--apply` stages the gateway without starting it. `--activate` starts the owned
gateway and verifies an ordinary unsigned request before writing
`~/.config/flashalign/client.json`. That file points to the gateway and contains
`"gateway": true`; it contains no engine signing key or upstream API credential.
Both `client.chat(...)` and `client.stream_chat(...)` also work with it.

Always On persists across restarts. Every routed tool and subagent request uses
the profile, including tools whose alignment flags are off. The host still
decides whether a tool may execute.

Use the reported `base_url` and configured model ID for every caller. The gateway
advertises only that model and rejects other model IDs. The internal engine URL
is retained only in private deployment state. Direct requests to that URL are not
covered by gateway policy. The gateway listens on loopback; this managed path
requires Linux process supervision. Existing remote SDK access in Verified
contacts mode remains available on other platforms.

A caller cannot override the profile with a per-request stock or tool-off hint.
Missing keys or policy state, an unavailable profile, or an unconfirmed effective
profile cause the request to fail without retrying in stock mode. The gateway
does not make callers verified or grant host tool permissions.

The result's `gateway` object records the approval-store path, private deployment
file, and explicit activation and stop commands. Stopping this gateway leaves
the upstream engine running. Repeated activation verifies and reuses the same
owned gateway. A validated replacement retires the previous owned gateway; a
failed replacement preserves its client configuration. Use the wizard with
`"mode": "verified"` to restore direct per-request profile selection. Changing only
the SDK's mode cannot override a running gateway's Always On policy.

For a direct terminal request, pipe text to `flashalign-client`. In default
Verified contacts mode, add `--owner` to select the configured profile. With
Always On configuration, the command uses that profile even without `--owner`.
The signing key authorizes requests for that profile. Keep it in the trusted
host process, outside prompts, browser code and model-accessible tools. For
verified channel/contact authorization and tool inheritance, use the
[custom harness guide](custom-harness.md).
