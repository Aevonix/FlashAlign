# Set up Aevonix FlashAlign

The wizard enables Hermes through its native plugin allowlist. For OpenClaw it
registers the package with the native plugin manager and enables the required
conversation hooks. Engine-only setup skips both integrations; see
[direct access](direct-engine.md).

On an interactive terminal, the wizard shows the FlashAlign pixel logo and
keyboard selection panels. Use arrow keys and Enter, or press a number key,
to choose an option. Existing selections remain the default. The startup
animation runs once and takes less than a second.

Add `--no-animation` to skip motion or `--no-color` for plain text. `NO_ANIM`
and `NO_COLOR` provide the same controls through the environment. Noninteractive
plans and redirected output keep their existing text or JSON output.

## Before you start

Have the model weights available locally. Install Hermes or OpenClaw for a chat
integration, or select `--engine-only` for direct access. Use
`bash ./install.sh --auto` for [managed engine setup](engine-setup.md): it prepares
a compatible engine package, stages the profile, provisions the key and
checks activation before connecting your harness.

To connect to an existing server, run `bash ./install.sh`. The server must
implement the Owner Mode API. Obtain its signing-key file and profile name or
profile folder from the operator.

The signing key is 32 raw bytes in an owned file with mode `0600`. Managed setup
installs the matching key and profile itself. Existing-endpoint mode uses the
key and profile already loaded by that server. Keep key contents out of messages
and terminal history. If the endpoint also needs an API key, the wizard prompts
without echoing it and stores it in a separate private file.

## Run the wizard

From the repository directory:

```sh
bash ./install.sh
```

The launcher creates a local Python environment and installs setup dependencies.
If you installed a release wheel, use `flashalign setup` instead. It includes the
same engine kits, starter profiles and OpenClaw adapter; no source checkout is
needed. For example, `flashalign setup --detect` inspects available integrations
and `flashalign setup --auto` starts the managed setup wizard.

The wizard guides you through:

1. **Harness:** Hermes, OpenClaw, or both. Detected installations are suggested.
2. **Model endpoint:** an existing API base URL and a model from its model list.
3. **Profile:** choose a matching local or engine-listed profile, or enter an
   installed profile name. Supply the engine signing-key file. Two one-token
   requests check standard mode and the selected profile.
4. **Access mode:** Verified contacts (the default), or Always On. Verified
   contacts offers installed adapters and asks for the exact contact ID and
   channel account. SMS and Telegram are opt-in; local terminal access also requires an
   explicitly authorized session. See [channels and terminal access](channels.md).
   Always On requires no contact.
5. **Tool profiles:** choose whether tools inherit the profile by default in
   Verified contacts mode. You can change individual tools afterward. Always On
   includes all routed tool and subagent requests and retains these settings
   for a later return to Verified contacts.
6. **Review and activate:** confirm configuration changes and any engine or
   gateway starts. Local terminal setup prints its launch command instead.

Setup stores Owner Mode configuration under `~/.config/owner-mode/`, installs
the selected adapters, and backs up changed harness files while preserving
unrelated configuration. For chat integrations, after applying a verified
profile it runs
`hermes gateway restart` and/or `openclaw gateway restart` for the selected
harnesses. A restart failure is reported separately; saved configuration and
backups remain available.

YAML configuration stays YAML, with mapping key order and unrelated values
preserved. Writing YAML regenerates its formatting: comments, quoting style,
layout and original anchor names are not preserved. JSON5 is patched only at
properties changed by setup, including nested properties. Unchanged text,
comments, quoting and key order remain intact, except that CRLF line endings
become LF. Configuration files are read and written as UTF-8. New properties
are appended; changed arrays and scalar values use JSON formatting, so comments inside a
replaced value or removed property are lost. Comments outside those spans
remain. Backups retain the original file bytes.

Use `--no-restart` to save configuration for a later restart. `--no-color` or
the `NO_COLOR` environment variable disables the Aevonix terminal colors.

When you select a local terminal identity, setup prepares that harness without
restarting its gateway and prints the explicit terminal launch command. The
native terminal needs an interactive TTY and exclusive access to its harness
state. Other selected chat harnesses keep their normal restart behavior.

For one-command managed setup with Always On:

```sh
bash ./install.sh --auto --mode always_on
```

Always On persists across restarts and uses the selected profile for every request
routed through FlashAlign, including tool and subagent requests whose alignment
flags are off. Point every intended caller at the advertised FlashAlign route or
gateway URL. Requests sent directly to another endpoint are unaffected.
Always On grants no host tool permissions or verified status. Requests fail if
the profile or key is missing, the route is disabled, or the effective profile
is unconfirmed. Callers cannot select stock mode for an individual request, and
FlashAlign never retries it in stock mode.

## Find profiles

Setup discovers the [starter profiles](../profiles/README.md) in `profiles/`.
Four payloads ship in the checkout; DeepSeek's native payload is fetched from
the public release and verified before use.
To add your own profile folders:

```sh
bash ./install.sh --profiles /path/to/profiles
```

Repeat `--profiles` to search more than one location. A location may be a profile
folder containing `profile.json` and `profile.safetensors`, a directory of those
folders, or a `catalog.json` with `models` entries containing `path` and optional
`model_ids` aliases. The wizard also searches `~/.config/owner-mode/profiles/`,
the checkout's `profiles/` folder, and paths in `FLASHALIGN_PROFILE_PATH`.

Local checks cover the model revision, transform requirements, tensor names and
supplied checksums. Compact profiles also require valid FP32 shapes, finite
values, positive gain/norm vectors and orthonormal bases, with ranks up to 128
and tensor files up to 256 MiB. Native DeepSeek payloads use a separate streaming
validator. Invalid profiles are excluded. Matching uses explicit model IDs or
catalog aliases; revision and format conflicts are rejected when the engine
supplies that metadata.

If the engine does not provide a [profile inventory](engine-contract.md#optional-profile-discovery),
enter an installed profile name directly. Model names are not limited to the
included starter profiles.

A selected profile is copied into private FlashAlign state. Managed setup also
loads it into the selected engine. Existing-endpoint mode verifies the profile
already loaded by that server. If activation fails, the wizard stops before
modifying or restarting the harness. Prepare remote engines on their host;
an API URL alone grants no installation access.

For a read-only report of detected harnesses and models, use:

```sh
bash ./install.sh --detect
```

Use `bash ./install.sh --version` to check the package version.

For a new deployment, `bash ./install.sh --init-key` creates or reuses its private
signing key. The engine must receive the same key before profile verification.
For an existing deployment, use its current key.

Agents can use `--config /private/setup-plan.json --check` and then `--apply` to
prepare setup without prompts. This mode stops before engine or gateway starts.
Use `--activate` when the user authorizes the pending activation step.
See [agent-assisted setup](agent-setup.md) for the plan format and restart handoff.

## Activate and verify

The wizard confirms engine activation before saving configuration. Chat
integrations restart their selected gateways; local terminal setup prints
the explicit launch command instead. In Verified contacts mode,
send a message through the approved direct channel.
Check that the engine response reports the installed profile in
`X-Owner-Mode-Effective`, including when the HTTP status is 200.

Read the latest routing decisions without displaying message content:

```sh
flashalign why
flashalign why --limit 20
flashalign why --log /path/to/owner-mode.log
```

`why` prints one JSON decision per line, oldest first, with the most recent ten
by default. It includes the channel (`platform`), reason (`decision` and
`identity_reason`), requested and effective profile, and timing/status fields
when available. It never prints request bodies, messages, contact IDs, session
IDs, credentials or unknown log fields. No rows means no recorded decisions
were found; it does not prove that a turn used the profile.

By default it reads Hermes's `~/.hermes/logs/owner-mode.log`, the bridge's
`~/.config/owner-mode/owner-mode.log`, and direct gateway logs beneath
`~/.local/share/flashalign/gateways/`. Use `--home /path/to/home` for another
home or repeat `--log` for custom locations. Missing or unreadable logs and
malformed JSON lines are ignored; readable decision rows are still printed.
A changed request body records `request_changed`; a
bridge request with no run capability records `no_run_context`.

In Verified contacts mode, check an unapproved contact too: its effective mode
should be `off`. Try one enabled and one disabled tool, then revoke a test
contact and confirm that subsequent requests use the standard model.

In Always On, check a normal request and a tool/subagent request through the
configured FlashAlign route. Both must report the configured profile. A caller's
request for stock mode must not bypass this setting. Unapproved contacts and
disabled tool-inheritance flags still use the configured profile in Always On.

If you use voice notes, attachments or queued messages, verify those paths too.
They need a supported native rendering binding and the appropriate transcription
or media support in your harness. See [message bindings](message-bindings.md).

## Manage contacts and tools

Pause or resume profile selection without changing the saved mode, contacts,
channels or tool settings:

```sh
flashalign pause
flashalign resume
flashalign pause --store /path/to/approvals.json
```

These commands set the approval store's existing `enabled` flag. They take
effect on subsequent requests without a restart and do not cancel requests
already running. Pausing makes routed requests use stock mode, including in
Always On. Resuming restores the saved policy; it does not add any approvals.
While paused, no turn counts as the owner's, so other plugins see a non-owner
and no tool inherits owner status; resume restores that status.
The default store is `~/.config/owner-mode/approvals.json`. For a direct gateway
or another installation, pass that deployment's approval-store path.

Use [channel administration](channels.md) to inspect installed adapters or
change the enabled channel selection. New transports can register their own
identity format and capabilities in trusted adapter code.

Use the main command for local access changes. `approve`, `revoke`, `mode`,
`channels` and `tool` call the same operations as `manage.py`. Each accepts
`--store /path/to/approvals.json`; without it they use
`~/.config/owner-mode/approvals.json`.

```sh
flashalign approve rcs +10000000000 --account default
flashalign revoke rcs +10000000000 --account default
flashalign channels --set whatsapp rcs telegram
flashalign tool sessions_spawn off
flashalign mode always_on
flashalign mode verified
```

List the full stored state with the original administration entry point:

```sh
python3 hermes-owner-mode/manage.py --store ~/.config/owner-mode/approvals.json list
```

Use `revoke` with the same channel, account and contact to remove contact approval
in Verified contacts mode. It does not switch off Always On.
Updates apply to subsequent requests without restarting the harness. The `list`
command displays your local approvals, mode and enabled state; keep that output
within your account. Mode is stored durably and reread on each request. Switching
modes does not re-enable a disabled approval store. Existing stores without a
mode retain Verified contacts behavior. Direct-engine client mode is saved in
`~/.config/flashalign/client.json`; see [direct access](direct-engine.md).

## Troubleshooting

| Result | Check |
| --- | --- |
| No models found | API URL, endpoint reachability and model-list permissions |
| Runtime requirement fails | Python or Node version in the compatibility report |
| Approved contact uses standard mode in Verified contacts | Native account/contact IDs, direct-message support and signing key |
| Engine reports profiles unavailable | Load a compatible profile through local administration; restart only if the engine lacks hot loading or needs a configuration change |
| Child work uses standard mode in Verified contacts | Creating tool's flag and native child attribution |
| Always On request fails | Configured profile, signing key, route state and effective-profile confirmation |
| Configuration was interrupted | Restore the timestamped backups, then rerun setup |

The setup result records profile validation, the effective-mode check and each
gateway restart separately. Native-channel tests and numerical or behavioral
profile evaluation remain separate checks. For adapter requirements,
see [compatibility](compatibility.md).
