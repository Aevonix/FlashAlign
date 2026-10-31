# Integrate another harness

For Always On, a custom harness can use the ordinary OpenAI-compatible
[FlashAlign gateway](direct-engine.md#always-on). Point every model caller,
including tools and subagents, at its advertised URL. The gateway applies the
configured profile to every routed request, regardless of sender attribution
or per-tool alignment flags. It persists until the operator changes the mode.
Host admission and tool execution permissions remain unchanged. Requests sent
to another provider or the raw upstream engine are outside the gateway policy.

For per-request selection, a trusted service can use the Python
[engine client](direct-engine.md) after performing its own authorization.
A chat harness can use Verified contacts mode with FlashAlign's contact
approvals, native-message binding and tool-inheritance policy through the bridge.

The identity and inheritance workflow below describes Verified contacts mode.
An Always On bridge selects the profile for every routed request without a
verified claim. It does not mark callers as verified.

The bridge is independent of Hermes and OpenClaw. Start
`python hermes-owner-mode/serve.py --config /private/bridge.json` with a private
configuration containing `approvals_file`, `control_key_file`, `key_file`, `port`
and `routes`. The control and engine keys are distinct 32-byte `0600` files.
The engine key signs upstream requests; the control key authenticates native
harness events over loopback. See the [OpenClaw adapter](../openclaw-owner-mode/index.mjs)
for a working non-Python integration.

Python adapters can use `HarnessClient` instead of building control requests:

```python
from flashalign import HarnessClient

with HarnessClient("http://127.0.0.1:8765", "/private/control.key") as bridge:
    # Populate these records from your authenticated native transport.
    claim = bridge.claim_native(session=session_id, run=run_id, messages=native_records)
    capability = claim.get("capability", "")
    try:
        if capability:
            bridge.bind_content(capability, rendered_content)
        response = bridge.chat_response(capability, model_messages, model=model_id)
        result = response.json()
    finally:
        if capability:
            bridge.end(capability)
```

`native_records` contains `id`, `channel`, `account` and `contact` for each message.
Use `tool_scope` around tool execution and `adopt` for a child created by that
tool. In Verified contacts mode, missing authorization returns no capability,
so ordinary requests remain standard. The bridge also checks the rendered
content on the first model call. The control key is used only to authenticate
control requests and is never forwarded to the engine.

Use these hooks in the host's trusted adapter code:

1. At native ingress, attest the actual channel, account, sender and message ID.
   Call the bridge's `claim` operation with the native message records. Never
   build this attestation from a prompt or a model response.
2. After rendering text, images, voice or attachments, call `bind` with the
   returned capability and the exact content the model will receive. Every
   member of a composed group must belong to the same approved contact.
3. Pass the opaque capability in the model request's transport metadata. The
   bridge decides whether to sign the upstream request; the capability never
   belongs in a prompt.
4. Open and close tool scopes through the bridge, propagate verified child-run
   origins and end each run. Contact revocation is rechecked on subsequent calls.

The control operations, field names and authentication handling are
implemented in [bridge.py](../hermes-owner-mode/bridge.py). The bridge's tests in
[test_portable.py](../hermes-owner-mode/tests/test_portable.py) are runnable contract
examples for native ingress, media binding, tool scopes and child inheritance.

Before shipping a Verified contacts adapter, verify approved and unapproved
contacts, forged sender fields, replayed message IDs, media transformations,
mixed message groups, disabled tools, child runs and revocation. Missing or ambiguous native evidence
must leave the request in standard mode. The installed registry separates
identity normalization, transport capabilities and enabled channels. WhatsApp
and RCS are enabled by default; SMS and Telegram are opt-in. A new transport can
register an adapter in trusted application code and use the same approval and
binding flow. See [the adapter interface](channels.md#add-a-transport-adapter).
Other authenticated services can also use the direct client with their own
authorization policy.

For Always On, also verify an ordinary request without identity metadata, a tool
with its alignment flag off, and a subagent request. Each must report the
configured profile. An explicit per-request `off` must not bypass it. Check that
missing profile/key or effective-profile confirmation fails without a stock retry.

For a custom Hermes identity adapter that consumes another plugin's evidence,
follow [custom identity provider ordering](custom-identity-ordering.md). Test
the installed manifest and hook order as well as the adapter's verification code.

Engine support is separate from harness support. Any harness may use an engine
that implements the [same signed-request contract](engine-contract.md); an
OpenAI-compatible endpoint alone does not implement FlashAlign transforms.
