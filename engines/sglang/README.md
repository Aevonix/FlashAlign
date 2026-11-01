# FlashAlign SGLang integration

This bridge applies a saved FlashAlign projection inside SGLang's target-model
forward pass.

The integration is pinned to SGLang **v0.5.20**, commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`. It supports the Qwen model below.

## Prepare and launch

Use an isolated environment. This does not patch an existing service or install
packages globally:

```sh
python3 -m venv /private/flashalign-sglang/venv
/private/flashalign-sglang/venv/bin/python -m pip install -r requirements.txt
git clone --depth 1 --branch v0.5.20 https://github.com/sgl-project/sglang.git /private/sglang-upstream
python3 prepare.py --source /private/sglang-upstream --output /private/flashalign-sglang/source

/private/flashalign-sglang/venv/bin/python launch.py \
  --source /private/flashalign-sglang/source \
  --model /models/qwen38-27b \
  --profile /profiles/qwen38-27b-v1 \
  --key-file /private/deployment.key \
  --draft-model /models/qwen38-dflash2 \
  --host 127.0.0.1 --port 8030 \
  -- --context-length 65536 --enable-cache-report
```

The default context is 65,536 tokens, tested with serial requests on one
95 GiB RTX PRO 6000 Blackwell Max-Q GPU with DFlash2. See
[validation](../../docs/validation.md) for setup, long-input and harness results.

The model, profile and draft model must already exist. The key must be a private
32-byte regular file. The launcher accepts repeated `--profile` arguments and
sets `FLASHALIGN_PROFILES` and `FLASHALIGN_KEY_FILE`. The equivalent TensorFold
environment names, `TF_OWNER_PROFILES` and `TF_OWNER_MODE_KEY_FILE`, are accepted
for provisioning compatibility.

The launcher keeps speculative decoding enabled when a draft model is supplied.
Additional SGLang arguments are preserved after `--`. The validated
integration uses DFlash2. An unsupported active algorithm is rejected; no
fallback disables it.

## Scope

- One GPU and one HTTP worker; TP, PP, DP and CP size one.
- Dense Qwen3.5-family target model classes. The starter used for qualification
  is Qwen3.8-27B at the revision declared in its profile.
- Standard projection profiles with embedding, attention and dense MLP output
  sites. Unsupported tensor sites or transforms stop startup.
- Eager execution, with both prefill and decode CUDA graph backends disabled.
  This restriction leaves DFlash2 speculation active. CUDA graphs and
  `torch.compile` require a separate implementation and validation.
- Existing in-memory radix-cache behavior with a server-created namespace
  containing the profile digest and exact strength. Owner and standard requests
  cannot share a namespace. LoRA composition, session-cache reuse, HiCache and
  distributed/disaggregated execution are rejected by this integration.

The supplied draft model is left unchanged and provides proposals. Only the target forward
pass applies the selected profile; it verifies the proposals against that target.
Main-model profile execution does not imply reproducing changes to an edited
checkpoint's native MTP weights.

## Direct API

Use `/v1/chat/completions` with the FlashAlign `owner_mode` and
`owner_mode_scale` fields, plus `X-Owner-Mode-Auth` signed over the original raw
body. The response carries `X-Owner-Mode-Effective`. The same gate covers the
native `/generate` and legacy `/v1/completions` routes. No harness is required.

The read-only discovery endpoint is `/v1/owner-mode/profiles`. Missing, expired,
invalid or replayed authentication serves standard mode. A valid unknown profile
returns the pinned `invalid_owner_mode` error. Client-supplied cache salts are
wrapped, including on standard requests, so they cannot impersonate an owner
namespace. Keys and authentication header values are never logged or forwarded
to SGLang's generation handlers. Request-content logging must remain disabled.

The HTTP server is the supported authentication boundary. In-process Python
code with access to the deployment process and key is trusted operator code.
Profile loading requires the authenticated HTTP startup path; separate ungated
transports are refused. Other POST APIs are closed while profiles are loaded,
apart from clearing the cache and SGLang's startup garbage-collector freeze. This prevents alternative or nested request
shapes from entering the scheduler without the authentication boundary.

## Validation

Run the component tests with an environment containing Torch and safetensors:

```sh
PYTHONPATH=. python -m unittest discover -s tests -v
```

Tests cover raw-body signatures, replay, tampering, cache namespaces, stock-row
identity, mixed prefill and speculative verification rows, strict loader behavior,
and actual PyTorch module hooks using the pinned full-attention naming pattern.

See [validation](../../docs/validation.md) for live protocol, cache and answer
checks, and [measured results](../../docs/benchmark-results.md) for timing samples.

Upstream SGLang source retains its Apache 2.0 notices. FlashAlign runtime additions
follow the surrounding FlashAlign repository's license.
