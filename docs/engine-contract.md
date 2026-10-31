# Owner Mode engine API

The adapter selects an installed profile by its opaque `owner_mode` name and
sets `owner_mode_scale` in `(0, 1]`. The inference engine applies that selection
only after authenticating the request. Profile files are installed separately
and scoped to a model revision and serving format.

This is the raw engine contract. An Always On FlashAlign gateway selects and
signs every routed request, including all tools and subagents, before it reaches
this boundary. A caller that sends requests directly to the raw engine is
outside that gateway's policy. See [direct access](direct-engine.md#always-on).

## Optional profile discovery

The setup wizard can read `GET <model-base-url>/owner-mode/profiles`. An engine
that implements this optional extension returns:

```json
{
  "data": [
    {
      "name": "example-profile",
      "model": {"id": "publisher/model", "revision": "pinned-revision", "format": "FP8"},
      "loaded": true
    }
  ]
}
```

Only loaded profiles are offered. Explicit `model.aliases` can identify the
same served model under additional IDs. The `/models` response may include
`revision` and `format` fields so the wizard can reject mismatches, and `input`,
`contextWindow` and `maxTokens` to configure the model descriptor.

This is a discovery extension, not an upload API or an existing-engine support
claim. A 404/405 falls back to local profiles or an operator-entered profile
name. The wizard independently checks standard and selected modes with two
one-token Chat Completions calls, including the required signed owner request.
It never assumes a listed or structurally valid profile is active.

## Request authentication

The engine and adapter share a protected 32-byte signing key. The header is:

```text
X-Owner-Mode-Auth: v1:<unix_seconds>:<random_nonce>:<hex_hmac_sha256>
```

Sign `timestamp + "\n" + nonce + "\n" + sha256(raw_body)` with HMAC-SHA256.
The engine enforces a 60-second timestamp window and rejects nonce replay
throughout that window. Replay-store failure must fail closed. Keep keys and
authentication headers out of logs.

The bundled Python engine integrations persist keyed nonce hashes and expiry
times across API restarts. vLLM and SGLang use
`$XDG_STATE_HOME/flashalign/replay/nonces.sqlite3`, or
`~/.local/state/flashalign/replay/nonces.sqlite3` when that variable is unset.
`FLASHALIGN_REPLAY_STORE` can select a stable private path. The llama.cpp
launcher keeps the store in its managed state directory. Keep this state while
reusing the deployment key. Its directory must be owned by the serving user
with mode `0700`, and its database must have mode `0600`.

TensorFold uses a private replay directory selected with
`--owner-replay-dir`; direct launches default to a sidecar beside the key file.
The wizard chooses a stable per-key directory outside versioned runtime
folders. The store holds an exclusive writer lock and durably saves accepted
nonce digests before authorizing a request. Preserve it across restarts and
upgrades. A new signing key is required when deliberately resetting this state.

The stores contain nonce hashes, expiration times and integrity metadata. A full,
unavailable or replaced store denies Owner Mode. Ordinary requests continue in
standard mode. Model prompt and KV-cache policy is configured separately.

Missing or invalid authentication uses standard mode. Valid authentication
selects the installed profile; an authenticated unknown profile is an error.
Return `X-Owner-Mode-Effective` with the actual selection, including `off`.

## Errors

The adapter recognizes these exact HTTP 400 messages for `invalid_owner_mode`:

| Case | Message |
| --- | --- |
| Profiles unavailable | `this server was started without owner-mode profiles: owner_mode must be "off"` |
| Unknown profile | `owner_mode names no profile this server loaded` |
| Invalid type | `owner_mode must be "off" or an owner-mode profile name` |
| Invalid scale | `owner_mode_scale must be a number in (0, 2]` |

The engine error string for scale is matched verbatim. This adapter accepts
only `(0, 1]`. In Verified contacts mode, only the profiles-unavailable response
causes one retry in standard mode. Always On returns the failure without a
stock retry. Other errors are returned to the caller. Recheck this contract
when upgrading an engine.

## Isolation and profile loading

Isolate caches by every effective profile, scale and model-revision dimension,
or bypass profile-mode cache reads and writes. Standard requests must not
reuse profile-mode cache entries or transforms. The engine must validate the
profile format and reject unsupported extensions before advertising support.

An in-memory profile index is a sufficient cache discriminator only while the
loaded model and profile set stay immutable. Reloading either must discard those
entries, or use new keys that include the changed model/profile content. Keys
must also distinguish the effective strength. A profile name alone is insufficient.

An effective-profile header reports selection; it does not itself establish
numerical correctness or cache isolation. Engine implementations must test
those properties independently. Model profiles never grant tool permissions.

## Optional weighted output projection

A profile may declare `transform: "weighted_output_projection_v1"` and
`requires: ["per_direction_strength"]`. Each site must have `scale: 1` and
these tensors in `profile.safetensors`:

| Tensor key | Type and shape | Validation |
| --- | --- | --- |
| `L{layer}.{site}.Q` | FP32 `[rank, hidden_size]` | Finite, orthonormal rows |
| `L{layer}.{site}.strengths` | FP32 `[rank]` | Finite, strictly positive values |

For a row batch `y` of the named matrix's unmodified outputs, apply the operator
before residual addition. With request strength `alpha = owner_mode_scale`:

```python
y_new = y - alpha * ((y @ Q.T) * strengths) @ Q
```

Each direction keeps its own strength. A loader must reject a non-unit site
scale, missing or invalid tensors, or unsupported `transform` and `requires`
values. Engines must explicitly support this weighted operator; accepting the
file schema does not establish runtime support. Do not average the strengths or
silently use an ordinary unweighted projection. Standard-mode rows bypass all
transform arithmetic, and mixed batches modify only the authenticated rows.
