# Native message bindings

In Verified contacts mode, FlashAlign verifies the contact who sent a message.
Transcription, attachment rendering and queue composition must preserve the link
to the original native message. A transcript, quoted author, filename or sender
ID in prompt text is not identity evidence.

This guide describes that verification workflow. Always On applies the profile
to every request routed through FlashAlign, including unverified messages,
groups, tools and subagents, regardless of these identity checks or per-tool
alignment flags. It does not label a sender as verified or add media support.
The host still controls which inputs and tools it accepts. See the
[Always On gateway](direct-engine.md#always-on) for ordinary clients.

## Shared contract

In Verified contacts mode, eligibility comes from a native channel, account,
contact and conversation. The contact must be approved and admitted by the
harness transport. The message-based path also requires native message IDs;
OpenClaw can use the verified-session path described below. Group chats, synthetic input and
own-account echoes do not become owner turns.

The harness then binds the rendered model input to the verified message or
message group. Text and Chat Completions content arrays are supported, including
image, audio, file and video parts. Media parts are hashed for comparison;
attachment payloads are not retained in the authorization ledger. Replacing a
bound attachment before the first request makes the request ineligible.

On Hermes and the bridge's message-based `claim` path, an ordered queued group
must contain distinct native messages from one channel, account, contact and
conversation. Every member is checked and consumed together. Mixed contacts,
a missing member, duplicate IDs and replay fail
closed. Groups are limited to 64 messages; receipt tables are bounded in memory.
OpenClaw's native verified-session path gives overlapping runs independent
claims and requires the same protected sender tuple on each admitted run.

The first model request must match the bound content. Configured host context
may add text; it cannot substitute or append unbound media to that first request.
The established turn can then continue its normal tool loop, subject to the
configured tool policy and contact revocation.

## Hermes adapter

`NativeAuthority.hear(event=..., gateway=...)` observes native ingress.
`HermesObserver` observes these host integration points when present:

- `gateway.platforms.base.merge_pending_message_event`: records ordered members
  while preserving the host's merge behavior.
- `GatewayRunner._prepare_profile_scoped_inbound_message_text`: records the
  rendered message returned by the host, including media context or transcript.
- `GatewayRunner._hmwa_apply_message_timestamp`: follows the host's final
  timestamp formatting without reinterpreting it as user identity.

Observers preserve arguments, return values and exceptions. They restore their
own wrappers on unload and do not replace a wrapper installed by someone else.
The original native route and message IDs bind the subsequent `pre_llm_call`.

For a host with a different rendering API, its trusted adapter can call
`NativeAuthority.record_render(event=..., gateway=..., content=...)` after
preparing the content. It must preserve the complete native composition and
verified transport metadata. Never call this from a user-controlled HTTP body.
For a custom composition, pass `members=[native_event_1, native_event_2, ...]`
and retain one member's native message ID as the composite anchor. Every member
is independently checked; caller-supplied ID strings are not accepted as events.
Without a rendering seam, direct text can still use its original content;
transformed media and groups stay in standard mode when using Verified contacts.

## OpenClaw and other non-Python adapters

OpenClaw uses native `message_received` and `before_dispatch` events to bind an
isolated direct-message session to a verified channel, account and contact.
Each model run must carry the same protected native sender fields. CLI routing
options and the spelling of a session key do not establish that identity.
Concurrent runs receive separate pending capabilities. The provider's
`onPayload` callback binds the final rendered content before dispatch; existing
payload callbacks run first. The direct `inbound_claim` contract remains
available to integrations that receive that targeted native hook.

The authenticated loopback bridge exposes the same two-step interface to other
trusted adapters:

| Command | Input | Result |
| --- | --- | --- |
| `claim` | `session`, `run`, and `messages` containing native `id`, `channel`, `account`, `contact` | Pending opaque capability, after checking all approvals |
| `claim_session` | `session`, `run`, `channel`, `account`, `contact` from an isolated, verified native session | Independent pending capability for that run, after checking its contact approval |
| `bind` | `capability`, `content` prepared by the host | Activates that claim for matching model requests |
| `end` | `capability` | Removes the run's authority |

Only the local adapter may call these commands with the protected bridge key.
The model receives neither that key nor the capability. Claiming a message
alone grants no model access. Native IDs cannot be claimed by another
run while retained in the bounded replay table.

Use `claim_session` only when the trusted host establishes session isolation
and carries protected sender metadata into every admitted run. It does not
authorize deriving a contact from session-key text, history or a user-supplied
request. Both claim forms require content binding and ongoing approval checks.

These bindings control model selection. They do not grant tool permission,
verify the factual accuracy of a transcript, or add media capabilities to an
engine. Configure transcription, attachment handling and model modalities in
the harness itself.
