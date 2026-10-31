# Channels and terminal access

The built-in Verified contact flow checks three things: an installed adapter,
the operator's enabled channel selection, and an active approval for the exact
channel, account and contact. A channel name in a request never supplies any of
those permissions. Existing custom identity authorities verify their own
evidence through the separate authority interface.

## Built-in adapters

| Adapter | Identity | Activation |
| --- | --- | --- |
| `whatsapp` | Native phone or linked-device identity | Enabled by default; approve each contact |
| `rcs` | E.164 phone number from trusted transport metadata | Enabled by default; approve each contact |
| `sms` | E.164 phone number from trusted transport metadata | Opt in after checking provider identity guarantees, then approve each contact |
| `telegram` | Positive numeric native user ID | Opt in, then approve the user for the bot/account |
| `local-terminal` | Local operating-system UID and harness | Explicit, expiring session authorization |

Phone-number attribution is only as strong as the configured transport. Verify
its sender authentication before treating an SMS or RCS number as an owner
identity. Telegram usernames, display names and group IDs are not user IDs.

The legacy `voice` and `companion` entries require an installed custom identity
authority. Their names do not make them native chat transports.

List installed adapters and their enabled state:

```sh
python3 hermes-owner-mode/manage.py --store ~/.config/owner-mode/approvals.json channels
```

The wizard offers installed adapters. To change an existing selection, supply
the complete list you want to keep enabled, then approve the exact contact:

```sh
python3 hermes-owner-mode/manage.py --store ~/.config/owner-mode/approvals.json \
  channels --set whatsapp rcs telegram
python3 hermes-owner-mode/manage.py --store ~/.config/owner-mode/approvals.json \
  approve telegram 123456789 --account default
```

Use the account ID supplied by your harness. Disable a channel by leaving it
out of `channels --set`. Existing capabilities from that channel become invalid.
SMS is also opt-in. After verifying the provider's sender authentication, add
`sms` to the complete channel selection you want to keep, then approve the
contact. Saved explicit channel selections are preserved across upgrades.
These controls apply to Verified mode. Always On uses its configured profile
for every routed request without claiming a verified identity.

## Interactive terminals

For direct multi-turn chat with a configured engine:

```sh
flashalign-client chat --owner
```

This uses the direct client's deployment credentials. It has no agent tools.
`/reset` clears history and `/exit` ends the chat.

Setup and direct chat share the FlashAlign terminal theme: a sharp pixel logo,
amber and teal controls, and a brief startup sweep. Interactive setup supports
arrow keys or number keys for its selection panels. Direct chat displays the
configured model and requested profile; a successful reply confirms the
effective profile through the engine contract.

Use `--no-animation` to skip the startup sweep or `--no-color` for plain output.
These options work with the setup wizard, `flashalign-client chat` and
`flashalign-client terminal`. Setting `NO_ANIM` or `NO_COLOR` also disables the
corresponding effect, including when the variable is empty. Redirected output
uses plain text. Narrow terminals use a compact layout.

To use the real Hermes or OpenClaw interactive terminal with its tools, start an
explicit local session from an interactive TTY:

```sh
flashalign-client terminal --harness hermes --authorize
flashalign-client terminal --harness openclaw --authorize
```

The launcher uses the installed harness and its configured model route. It
binds each native text submission to a fresh turn and applies normal
tool-inheritance settings. Terminal images, voice and attachments are not
covered by this text binding and use standard mode. It does not authorize
unrelated CLI, cron or API traffic. The local
grant lasts one hour by default, expires when the launcher exits, and can be
bounded with `--duration SECONDS` from 60 to 86400 seconds.

The command displays a session ID for revocation. It is not a signing key.
From another terminal:

```sh
flashalign-client sessions list
flashalign-client sessions revoke SESSION_ID
```

Revocation or expiry removes profile authorization from subsequent model calls,
including inherited work. A request already running is not cancelled. Under
Always On, profile selection follows that global setting instead of a local
session grant; the session launcher does not misrepresent this as revocable
per-session access.

## Add a transport adapter

Register adapters in trusted application code, not configuration strings or
request bodies. Use a stable ID, an idempotent identity normalizer and only the
capabilities the adapter actually implements:

```python
from owner_mode.channels import ChannelAdapter, DEFAULT_REGISTRY, opaque_id
from owner_mode.contacts import ApprovalStore

registry = DEFAULT_REGISTRY.with_adapter(ChannelAdapter(
    id="enterprise-chat",
    normalize_contact=opaque_id,
    capabilities=frozenset({"native_message"}),
    label="Enterprise chat",
    identity_version="1",
))
store = ApprovalStore("/private/approvals.json", registry=registry)
store.set_channels(["enterprise-chat"])
store.set_contact("enterprise-chat", "workspace-id", "native-user-id", True)
```

Pass the same registry and store to the trusted authority and bridge described
in [custom harness integration](custom-harness.md). The stock `serve.py` uses the
built-in registry; an additional adapter needs a trusted launcher that constructs
the bridge with its extended registry. Enabled-channel strings cannot import
code or create an adapter. The separate custom identity-authority configuration
can load trusted installed code and must be controlled by the operator.

`native_message` binds individual transport messages. `native_session` supports
the protected native session path. `local_interactive` requires a local session
authority; it cannot be claimed through ordinary message metadata.
`custom_authority` requires the application's own verifier.

Bind identities and rendered content from the host's authenticated transport.
Keep them outside prompts. Test unknown users, account mismatches, replay,
media, revocation, tools and delegated work before enabling an adapter.

Increment `identity_version` if identity normalization or the meaning of an
account/contact changes. Existing grants then require explicit renewal. Removing
an adapter leaves its stored records intact but inactive, without disabling
unrelated channels. Reinstalling the same version restores eligibility for
its otherwise active approvals; change the version when the identity contract
changes.
