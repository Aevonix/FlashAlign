# Aevonix FlashAlign for Hermes

The [setup wizard](../docs/setup.md) installs the plugin and configures the
loopback model provider. Native channel/contact verification is included.

Verified contacts is the default. Always On applies the configured profile to
every request routed through this provider, including all tools and subagents
regardless of per-tool alignment flags. It persists across restarts until the
operator changes the mode. It does not mark a sender as verified or grant tool
execution permissions. Model requests sent to another provider are outside this
route's policy. See [setup](../docs/setup.md) for switching modes.

## Configuration

Equivalent manual settings:

Setup writes YAML with mapping key order preserved. It regenerates layout,
quoting and anchor names and does not retain YAML comments; the backup keeps
the original bytes. See [configuration preservation](../docs/setup.md#run-the-wizard).

```yaml
plugins:
  entries:
    owner-mode:
      enabled: true
      settings:
        serve_argv: [gateway, run]
        port: 18034
        approvals_file: /absolute/path/to/approvals.json
        key_file: /absolute/path/to/engine.key
        trust:
          system_prompt: host
          tools: host
          turn_context: host
          history: verified_owner
        routes:
          head:
            upstream: http://127.0.0.1:8030/v1
            profile: YOUR_INSTALLED_PROFILE
            owner_mode_scale: 1.0
            timeout: 1800
```

Set the model base URL to `http://127.0.0.1:18034/head/v1`. Use the profile name
installed for that engine and model. The request scale is configuration-owned
and lies in `(0, 1]`. Client-supplied selection overrides are stripped.

The signing key contains 32 raw bytes in an owned `0600` regular file. The
approval store also requires an owned `0600` regular file. The wizard and
administration command create the store. Authorization rereads its mode,
approvals and tool flags. Contact revocation takes effect without a restart in
Verified contacts mode; it does not disable Always On.
Use `flashalign pause` and `flashalign resume` to suspend and restore profile
selection for either mode. Pass `--store` for a nondefault approval store.
These commands preserve the saved contacts, channels, mode and tool settings.

Saved channel selections also take effect without a restart. The plugin admits
installed channel adapters and the policy checks the approval store on each
call. An older configuration with `allowed_channels` still limits the plugin
to that list; remove the key and reload the plugin, or rerun setup, to use the
live selection for every installed adapter.

Plugin managers in the same process share the first active registration's
policy, settings and proxy. Each manager receives the guarded hooks and
middleware. Unloading one manager keeps that proxy open until the last
registration unloads, including when a reload registers before unloading.

## Verified contacts

The adapter checks the registered transport, host access list, direct-message
status, account, contact, route and message ID. It binds native message text to
one turn. Contact approvals have no expiry; each incoming turn still needs
native attribution.

Voice, attachments, forwarded/quoted messages and queued groups retain their
native attribution through the host's rendering step. The adapter observes
the renderer and queue merger without changing their return values, content
or exceptions. Every member of a group must belong to the same approved
channel/account/contact. See [message binding](../docs/message-bindings.md).

An optional `identity_adapter` may name an operator-installed module defining
`Authority(proof_type=...)` or
`build_authority(toolkit, proof_type=..., settings=...)`. This is trusted host
code and must verify native ingress. Do not create proofs from model text.
If the adapter depends on another host plugin's identity capture, declare and
test that deployment-specific ordering. See
[custom identity provider ordering](../docs/custom-identity-ordering.md).

`owner_turn(session_id, turn_id, tool_name=None)` reports a personally verified
turn, optionally subject to a tool flag. Delegated work follows its own
inherited authority path. Tool middleware preserves arguments, results and
exceptions while applying model-selection scope.
This API reports identity; it does not report whether Always On applies to the
provider's next request.

Request tickets bind the whole request body except transport streaming flags.
If later middleware changes that body, the request uses stock mode and the
decision log records `request_changed`. It does not reuse the ticket's identity
metadata for the changed request.

The plugin also emits one warning and a decision entry when a request route
does not match (`route_mismatch`), request middleware raises
(`request_middleware_exception`), the process does not match `serve_argv`
(`serving_process_mismatch`), or native ingress observation fails
(`ingress_observation_exception`). These diagnostics contain fixed reason
codes, not message content or exception text. They do not change fallback or
authorization behavior.

Use `flashalign why` to read the recent decision trail, or
`flashalign why --log /path/to/owner-mode.log` when `audit_log` names a custom
location. Hermes defaults to `logs/owner-mode.log` beneath its home. The
standalone bridge and direct gateway default to `owner-mode.log` beside their
approval store; both accept `audit_log` in their configuration. The bridge
also records control failures and returns a fixed `reason` alongside its error.
See [setup diagnostics](../docs/setup.md#activate-and-verify) for the reader's
fields and options.

## Trust settings

`host` accepts framing and tools supplied by the harness. `pinned` requires
configured hashes. Absent or invalid settings use restrictive defaults.
`history: verified_owner` accepts validated history structure; it does not
authenticate every message in that history. Harness permissions still govern
tool execution.

For channel and version requirements, see [compatibility](../docs/compatibility.md).

Licensed under [AGPL-3.0-only](LICENSE), with a separate commercial option.
The installed plugin includes the full terms and attribution notice.
