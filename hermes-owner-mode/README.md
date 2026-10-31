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
        allowed_channels: [whatsapp, rcs]
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

## Trust settings

`host` accepts framing and tools supplied by the harness. `pinned` requires
configured hashes. Absent or invalid settings use restrictive defaults.
`history: verified_owner` accepts validated history structure; it does not
authenticate every message in that history. Harness permissions still govern
tool execution.

For channel and version requirements, see [compatibility](../docs/compatibility.md).

Licensed under [AGPL-3.0-only](LICENSE), with a separate commercial option.
The installed plugin includes the full terms and attribution notice.
