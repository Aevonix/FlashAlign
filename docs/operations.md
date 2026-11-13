# Inspect a deployment

The setup environment includes the FlashAlign client and operating commands:

```sh
bash ./flashalign.sh status
bash ./flashalign.sh doctor
bash ./flashalign.sh dashboard
```

After `python -m pip install .`, the same commands are available as `flashalign`.
Use `--config /path/to/client.json` for another deployment. Managed setup reports
its client configuration path.

`status` and `doctor` read model, profile and runtime metadata. They do not
submit a prompt. To send a small synthetic activation check explicitly:

```sh
bash ./flashalign.sh doctor --probe
```

The probe checks the effective response mode. It does not test refusal behavior
or prove that every model layer executed the intended transform.

## Dashboard and JSON

```sh
bash ./flashalign.sh status --json
bash ./flashalign.sh dashboard --interval 2
bash ./flashalign.sh dashboard --json --samples 5
```

The dashboard uses the existing FlashAlign terminal styling and supports
`--no-color`. JSON output is suitable for scripts. It excludes credentials,
prompts and responses.

The dashboard displays graph mode, generation, expected, installed and exercised
sites, active requests and GPU memory when supplied by the engine. Missing
values display as unavailable, while measured zeros remain zero. A listed or
loaded profile is not a behavioral benchmark. The dashboard does not display
an execution attestation: no engine currently supplies one.

Memory observations describe the engine's reported usage, not the profile
admission estimate. Allocated, reserved and available memory are distinct
measurements; unsupported measurements stay unavailable. Profile payload
budgets do not cap all engine or allocator memory.

Output rates use counter changes between samples from the same engine instance
and generation. They describe the observation window, not a maximum-throughput
benchmark. TTFT and draft acceptance are unavailable when the engine does not
expose the corresponding measurements. vLLM and SGLang reuse their existing
in-process metrics collectors; metrics must be enabled in the engine. A missing
collector or sample remains unavailable. SGLang reads generation tokens from
one tensor-parallel leader, avoiding duplicated counts. llama.cpp currently
leaves output-token and TTFT counters unavailable; the legacy vLLM DeepSeek kit
does not expose runtime status. No shipped adapter currently reports draft
acceptance counters. See the [status contract](engine-contract.md#optional-runtime-status)
for the shared counter names and scope.

See [profile management](profile-lifecycle.md) for loading, activation, rollback
and recovery, and [benchmarks](benchmarks.md) for controlled comparisons.

## Startup budget

Managed preparation records `startup_timeout` in seconds for every engine kind.
An explicit engine setting overrides the kit's `startup_timeout`; SGLang's Qwen4
and GLM Flash kits declare 1,800 seconds. Without a declaration, Python engine
kits use 900 seconds, or 3,600 seconds for native Docker launches. TensorFold
keeps its bundle default of 1,800 seconds. Activation uses the recorded budget.
Use only `startup_timeout` in kit manifests and setup plans.
The timeout must be an integer from 60 to 7200 seconds for every engine kind.
Deployments prepared before this release keep their recorded budget (900 seconds
for Python kits) until they are prepared again.

## Configured profiles on restart

SGLang, vLLM and llama.cpp recover the lifecycle journal before checking the
configured profile list on every start. A configured name without an alias
record in the journal is staged and activated. Existing active aliases keep
their generation; changing a file under the same name does not replace it.
Use the profile admin commands to replace an active generation.
A configured profile that cannot be staged or activated stops the start.
Earlier releases activated configured profiles only on the first start, so names
added later are activated at the next restart.

A profile deactivated through the profile admin commands stays deactivated
across restarts, even when it is still in the startup configuration and its old
generation has been retired. Activate it again with the profile admin commands.
Removing a configured path alone does not deactivate an alias already in the
journal.

## Native startup failures

The native Docker launcher reports the last failing readiness step and its
cause when startup times out or repeated readiness checks fail. The steps are
engine health and profile discovery, native worker coverage, and native worker
liveness. A container exit also includes the last readiness failure when one
was observed. Failed helper commands report their operation, exit status and
standard error. Container stdout logging remains disabled; these diagnostics
come from the launcher's checks, not an inferred model execution result.
