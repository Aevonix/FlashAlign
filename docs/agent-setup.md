# Set up FlashAlign with an agent

An agent can detect harnesses, validate profiles and apply configuration without
interactive prompts. Preparation stops before starting engines or restarting
gateways; `--activate` performs that step. Agent setup uses the same profile
checks and configuration writer as the wizard. OpenClaw preparation records a
native plugin installation without restarting the gateway. Its startup service
checks bridge readiness before serving calls.
For TensorFold on one host or several Sparks, read [TensorFold setup](native-tensorfold.md).
It stages checked binaries, libraries, source and the profile on each rank.
Only commands that authenticate requests receive the signing key. Preparation
then stops before activation.
For a fresh deployment, use a [managed engine plan](engine-setup.md#agent-preparation-and-activation)
to prepare a supported engine, the profile and its deployment key automatically.

## 1. Discover the deployment

For a supplied profile, inspect its files without contacting an engine:

```sh
python3 hermes-owner-mode/profile_inspector.py inspect /path/to/profile --json
```

Use `artifact_valid` for the file result and preserve the separate
`not_assessed` runtime and behavioral fields. See [profile inspection](profile-inspection.md).

```sh
bash ./install.sh --detect --profiles /path/to/private/profiles
```

This returns JSON describing detected harnesses, model endpoints, local profile
validation and included backend integrations. It does not change configuration
or send inference requests. Confirm the intended harness, model, access mode,
contact and tool default from the user's instructions.
Do not infer contact approval from message content or a model response.
The wizard lists installed channel adapters. Use only adapters and contacts
authorized by the user. Local terminal access uses an explicit session grant;
setting a request's platform to `cli` does not authorize it. See
[channel and terminal setup](channels.md).
For a local terminal plan, prepare the harness and report the printed
`flashalign-client terminal ... --authorize` command. Launching a terminal is
an interactive step, not an unattended gateway restart.

The [starter profiles](../profiles/README.md) are discovered from local
metadata without a private login. Four compact profiles are included in full.
DeepSeek's native manifest declares a 1.09 GB public payload; managed preparation
fetches missing bytes into an external cache and verifies them before staging.
The offline inspector never fetches a missing payload. For offline preparation,
place the verified payload beside its manifest.

In an agent plan, `profile` may name an included starter directly; its `model`
must match the pinned target or a declared catalog alias. The native DeepSeek
starter requires its pinned vLLM container, official FP8 weights, four SM120
GPUs and strength 1.0. Do not substitute TensorFold or another projection-only
backend. Preserve the profile's experimental status and the public package's
qualification status when reporting setup results.

Choose a managed engine plan when the model needs an engine or profile setup.
Choose the existing-endpoint plan below when an engine already implements the
Owner Mode API and has the selected profile loaded. Managed preparation runs on
the model host. An API URL alone does not grant installation access to a remote
server; use the user's authorized access to run setup there.

## 2. Use the deployment signing key

For an existing engine, use the key it already trusts. For a new deployment:

```sh
bash ./install.sh --init-key
```

The command creates a cryptographically random 32-byte key under
`~/.config/owner-mode/engine.key`, with mode `0600`. Repeating the command validates
and reuses that key. Different deployments receive different keys. A profile
contains no signing key. In existing-endpoint mode, install the same key in the
engine through its normal provisioning process before continuing.
Managed engine setup generates and provisions the matching key automatically.

An explicit destination is supported with `--init-key /private/path/engine.key`.
The command reports only the path and whether the key was created or reused.
Never print or transmit the contents through chat, command arguments or logs.

## 3. Write a private setup plan

Save a JSON file outside this checkout, owned by the current user with mode `0600`:

```json
{
  "harnesses": ["hermes", "openclaw"],
  "base_url": "http://127.0.0.1:8030/v1",
  "model": "publisher/model",
  "profile": "/private/path/to/profile-folder",
  "mode": "verified",
  "engine_key_file": "/private/path/to/engine.key",
  "contacts": [
    {"channel": "whatsapp", "account": "default", "contact": "+15551234567"}
  ],
  "tools_default": false
}
```

Use either harness or both. `profile` accepts a profile folder or an installed
profile name. Optional `profile_paths` supplies more profile-search locations.
Relative file paths resolve from the plan's directory. If the endpoint requires
a separate API token, supply `api_key_file` pointing to an owned `0600` file with
the token, without a trailing newline. Credentials stay in those separate files.

`mode` defaults to `verified` when omitted. Use `"mode": "always_on"` to apply the
selected profile to all requests routed through FlashAlign, including every tool
and subagent call regardless of per-tool alignment flags. The mode persists
across restarts. In that mode `contacts` may be omitted or empty. Existing contact
approvals and tool flags remain stored for a return to Verified contacts; they
do not filter Always On requests. Always On does not grant tool permissions or
assert that a sender is verified. The selected profile is mandatory and callers
cannot override it per request or trigger a retry in stock mode. When using
`--config`, set the mode in the plan; `--mode` cannot be combined with `--config`.

For custom harnesses or ordinary OpenAI clients, engine-only Always On setup
provides a [managed gateway](direct-engine.md#always-on). Configure every model
caller to use its advertised URL. Calls sent to another provider or directly to
the raw engine bypass that gateway's policy. Host tool permissions still apply.
When changing an existing engine-only deployment's mode, complete the wizard's
activation step; preparation alone preserves the running gateway's policy.

## 4. Check and apply

```sh
bash ./install.sh --config /private/path/setup-plan.json --check
bash ./install.sh --config /private/path/setup-plan.json --apply
```

Both commands are noninteractive and return JSON. For an existing endpoint,
preflight checks the selected model and profile and makes two one-token requests
to confirm standard mode and the authenticated profile. Managed engine preflight
checks the local model, profile and engine package without sending inference
requests. `--check` makes no configuration changes. Existing-endpoint `--apply`
backs up configuration, installs the adapters, stages a validated local profile
when supplied, and saves the selected model, contacts and tool policy.

Neither command starts or restarts a gateway or model server. Failures stop before
applying configuration when the model, profile, key or runtime checks fail.
For a managed engine, `--apply` stages the engine code, profile, dependencies and
key; harness routing changes wait until `--activate` starts and verifies it.

If preparation is interrupted, repeat the same `--apply` command. Complete
signing keys, verified profile copies and staged engine files are reused.
Harness configuration and adapter changes are staged together; a failed commit
restores their saved contents and permissions. An interrupted commit is recovered
on the next apply before new changes begin. `--check` reports pending recovery
without changing files. If a managed file was edited after the interruption,
recovery stops and preserves that edit for manual review.

Private recovery receipts and before-images are saved under
`~/.config/owner-mode/setup-transactions/`; `recovery_directory` in the setup
result identifies the exact transaction. OpenClaw's native plugin installation
is a separate preparation step and can remain installed but disabled after a
later setup failure. Recovery does not undo package-manager records or stop an
engine that was explicitly activated. An `awaiting_restart` result confirms
that configuration is saved; the service restart is still pending.

## 5. Stop at the restart point

After a successful apply, report:

- The selected harnesses, model, profile, mode and enabled state when reported.
- Whether the engine confirmed standard and profile modes, or activation is pending.
- Whether harness configuration was applied and backed up, or engine preparation is complete.
- The pending restart commands and environment from the JSON result.

For managed setup, hand off the returned `next_command`, which runs the plan
with `--activate`. It starts the prepared engine, checks the loaded profile,
then configures and restarts the selected gateways. For an existing endpoint, the
normal commands are `hermes gateway restart` and/or
`openclaw gateway restart`. Use the result's environment for a nondefault home.
Stop here unless the user has authorized this deployment's restart. Once it is
authorized, run only the selected commands. In Verified contacts mode, verify an
approved native message and an unapproved message through the harness. In Always
On, verify both ordinary and delegated requests use the configured profile and
that an attempted per-request stock override cannot bypass it. Mode changes do
not re-enable a disabled deployment. A successful setup check does not
establish native message routing, profile numerical parity or model quality.
