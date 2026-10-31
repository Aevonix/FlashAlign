# Setting up FlashAlign for a user

Read [the agent setup guide](docs/agent-setup.md) before configuring a deployment.
For standalone setup without a harness or contacts, use an `engine_only` plan
from [direct engine setup](docs/direct-engine.md).
Use the noninteractive setup plan for discovery, profile checks and installation.
`--config ... --apply` prepares the selected engine, profile and signing key.
It stops before starting the engine or restarting a gateway.

Preserve active speculative decoding and its draft model. Never disable it to
make a profile pass setup. If the active speculative configuration is unsupported,
report that incompatibility and stop before changing the deployment.

Default to Verified contacts unless the user selects Always On. In Verified
contacts mode, use only contacts and channels approved by the user. Always On
applies the configured profile to every request routed through FlashAlign,
including all tool and subagent requests, regardless of per-tool alignment
flags or sender verification. The mode persists until the operator changes it;
it does not grant tool execution permissions or mark a sender as verified.
For custom harnesses and ordinary OpenAI clients, use the advertised Always On
gateway URL. Requests sent directly to another provider or the raw upstream
engine are outside that gateway's policy.

Only the included starter profiles belong in `profiles/`. Keep additional
profiles, signing keys, API keys and setup plans outside this repository. Never
print key contents or put them in a command argument, prompt, commit or report.

After setup, report the preparation or check result and pending activation
command. Stop unless the user has explicitly authorized restarting this
deployment. Existing authorization still applies; do not request it again.
For local terminal access, report the explicit interactive authorization command
instead of starting a gateway. A terminal label alone grants no access.
File validation does not prove that the engine activated the profile or that
its behavior was tested.

For development, use temporary homes, synthetic profiles and test gateways.
Do not restart the user's real Hermes, OpenClaw or inference service while testing.
Do not delegate to or contact other agents unless the user requests it.
