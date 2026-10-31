# Security model

FlashAlign controls which model requests receive an alignment profile. The
harness controls who can submit requests and which tools may run. A profile
does not grant tool permissions or filter tool output.

## Identities and channels

Verified contacts mode trusts identity metadata supplied by the configured
harness adapter. Text claiming to be an approved sender does not establish
identity. Custom adapters must bind the transport, account and sender to the
actual incoming event; see [channel adapters](channels.md).

Sender-number authenticity depends on the messaging provider. Some SMS
gateways allow spoofed sender numbers. Enable SMS only when the provider's
identity guarantees meet your needs. SMS and Telegram are disabled by default
and require explicit opt-in. An existing saved channel selection is preserved.

Always On applies the profile to every request routed through FlashAlign,
including tools and subagents. Use it on a public bot only when that profile is
intended for every caller. It does not make callers verified.

The standalone Always On gateway binds to `127.0.0.1` and accepts ordinary
local API requests without a per-caller password. Programs that can connect
on that host can use it; loopback binding does not isolate other local users.
The gateway rejects foreign browser origins and non-loopback HTTP hostnames.
Remote access needs a separately authenticated tunnel or gateway.

## Signed requests and restart protection

Keep each deployment's signing key private and use encrypted transport between
hosts. HMAC signatures bind the exact request body; they do not encrypt it.
The engine rejects replay throughout the accepted timestamp window, including
after a restart. Missing or invalid authentication leaves the model in stock
mode, and the client checks the effective-profile response.
The Python engine wrappers close WebSocket connections before model dispatch;
V1 defines signed HTTP routes, not a WebSocket authentication protocol.

TensorFold uses private, authenticated nonce snapshots. The wizard keeps
them outside versioned runtime directories so an engine or profile upgrade
does not discard them. The snapshots contain nonce digests, expiry times and
integrity metadata. They contain no prompts, responses or model cache data.
The other engine integrations use their documented persistent replay stores.

Preserve the replay store whenever you retain a signing key. If you intentionally
reset or lose that state, generate a new key and update its clients.
Storage errors, corruption or another writer holding the native store lock
must not permit owner authorization.

## Untrusted content and model caches

Restricting abliteration to approved callers reduces exposure to malicious
injection from other callers. An approved request can still contain untrusted
documents or tool results. The harness must enforce its own tool approvals and
content-handling rules throughout that request.

Stock and profile requests use separate model-cache identities, including the
profile strength. Authentication replay state is independent of those caches.
Clearing a model cache does not require clearing replay protection.

Profiles are experimental model transforms. They do not guarantee a particular
answer, zero refusals, or protection from every injection attempt. See
[validation](validation.md) for tested behavior and limits.
