# Custom identity provider ordering

This guide applies to Verified contacts mode. Always On selects the configured
profile for every request routed through FlashAlign, including tools and
subagents, without requiring an identity provider. It does not turn those
requests into verified identities. See [direct access](direct-engine.md#always-on).

The standard Hermes integration includes its own native contact verification.
An additional plugin dependency is needed only when a custom `identity_adapter`
uses identity state produced by another host plugin.

That producer must register and run before FlashAlign verifies the turn.
Installing both plugins or listing them in a particular configuration order
does not establish the required hook order.

## Declare the dependency

Keep a deployment-owned overlay of the installed FlashAlign `plugin.yaml`.
Preserve the existing manifest fields and add the provider's actual plugin ID:

```yaml
requires_plugins:
  - verified-identity-provider
```

`verified-identity-provider` is an example ID, not an included plugin. The host
must support plugin dependencies, and that provider must be installed and
enabled. Reapply and verify this deployment-specific overlay when updating
FlashAlign.

A dependency changes load order. It does not prove a contact's identity or
grant permission to produce owner claims. The adapter must still validate
native transport evidence for each turn.

## Preserve the complete order

For each top-level turn, the integration must establish this sequence:

1. The trusted transport admits the native message and its identity.
2. The host prepares the rendered text or media and preserves its binding to
   that native message.
3. The custom identity producer makes the current turn's evidence available.
4. FlashAlign's `pre_llm_call` hook verifies that evidence for the exact session
   and turn.
5. The request middleware authorizes the final request body, and the proxy signs
   the bytes sent to the engine.

Some providers collect evidence at ingress; others also need an earlier
`pre_llm_call` hook to bind it to the current turn. Validate the actual host
callbacks used by the adapter. A dependency declaration cannot fix a producer
that runs asynchronously after verification.

The adapter must return no proof when current evidence is missing, stale,
revoked or inconsistent. A prior owner turn, a sender ID inside prompt text, or
successful model output is not a substitute for current verification.

## Test the installed integration

Exercise the selected host's real manifest discovery, dependency ordering and
hook dispatcher with synthetic transport events and a local test engine. Include:

- A new approved session and another turn in an existing approved session.
- An unapproved contact and a missing identity producer.
- Revocation, changed native message identity, and transformed media when the
  adapter supports them.
- The configured reasoning settings, including nested template fields, through
  request middleware and the HTTP proxy.

Check that approved requests reach the test engine with the expected effective
profile and that unapproved requests remain off. Test the final serialized body
and replay handling as well as adapter construction. Calling the identity
producer manually before verification, or signing a direct engine request,
does not test the host's installed callback order.

After updating a deployment, verify the installed files and restart or reload
the affected gateway using the host's supported mechanism. A model-server
restart is unnecessary when only the host integration changed.

## Diagnose using mode metadata

The proxy's mode audit can distinguish common integration failures without
recording conversation content. The identity-related statuses below describe
Verified contacts mode; a missing identity does not prevent alignment in Always
On when the request reaches its configured route.

| Result | What it establishes |
| --- | --- |
| `owner_unverified` | The gateway did not have a valid active proof for this turn. Check adapter loading, current native evidence and callback order. |
| `no_middleware_context` | The request reached the proxy without its expected middleware context. Check the configured provider route and host request hook. |
| `bad_chat_template_kwargs` | The authorized request contains unsupported template fields or values. Check the host's final serialization and the selected engine's compatibility declaration. |
| Matching requested and effective profile | The engine reports the selected profile active. |

If the authority cannot be constructed at startup, FlashAlign emits
`FlashAlign identity authority unavailable` followed by the exception class.
The exception message is omitted because it can contain private paths or
credentials. Owner authorization remains unavailable until the adapter or its
dependency is repaired; the warning does not enable a fallback identity source.

Do not copy message text, contact IDs, authentication headers or signing keys
into diagnostics. Keep identity verification enabled while resolving failures.

See [native message bindings](message-bindings.md),
[custom harness integration](custom-harness.md), and the
[engine contract](engine-contract.md) for the remaining boundaries.
