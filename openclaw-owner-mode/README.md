# Aevonix FlashAlign for OpenClaw

Run the [setup wizard](../docs/setup.md) and select OpenClaw. It installs the
extension, configures its model provider and registers the Python bridge as a
harness-managed service.

Verified contacts is the default. Always On applies the configured profile to
every request routed through the FlashAlign provider, including all tools and
subagents even when their alignment flags are off. It persists across restarts
until the operator changes the mode. OpenClaw still controls message admission
and tool execution permissions. Always On does not make a sender verified, and
requests sent to a different provider are outside this route's policy.

## Verified contacts

The adapter uses native contact and run metadata to authorize a turn. An opaque
capability accompanies model requests in transport headers. Credentials and
capabilities are never added to prompts.

Native ingress first creates a pending claim. The provider's `onPayload` hook
then binds the final text or multimedia content to that claim before dispatch.
An unbound claim never enables Owner Mode. This supports the host's rendered
text and attachments. Composed-message integrations require distinct native
member IDs and a single verified channel, account and contact.

Native direct messages require `per-account-channel-peer` or
`per-channel-peer` session isolation. The wizard preserves either and upgrades
shared scopes with a configuration backup. Each admitted run must carry the
verified native channel, account and sender; a CLI request directed at an
existing conversation does not inherit its authorization. Overlapping runs get
independent capabilities.

Native `sessions_spawn` children inherit when the awaited `subagent_spawned`
event identifies one active, enabled parent tool. The child can continue after
that tool and parent run finish; contact revocation and tool flags still apply.
The typed `inter_session` path remains available for other supported host
flows. Ambiguous or unattributed children use the standard model. Tool
execution permissions remain with OpenClaw.

## Shared runtime

The bridge listens on loopback and uses a separate protected control key. The
service verifies the authenticated process and unique startup instance before
accepting readiness. An existing listener from another deployment cannot stand
in for its bridge. The service owns only the bridge process it starts. In
Verified contacts mode, a tool scope that cannot close stays in standard mode
until that run ends.

The integration targets the OpenClaw 2026.10.8 plugin interface. See
[compatibility](../docs/compatibility.md) for runtime requirements and activation
checks.

FlashAlign preserves explicit output limits and supports omitting them when the
engine allows it. Model-specific reasoning restrictions are checked before
dispatch; an omitted preference stays unchanged. See the policy format in
[engine setup](../docs/engine-setup.md). These controls require the regular
provider stream. Device worker-local inference is unsupported.

Licensed under [AGPL-3.0-only](LICENSE), with a separate commercial option.
The extension includes the full terms and attribution notice.
