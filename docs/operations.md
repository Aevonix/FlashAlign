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

Configured selection, engine-reported state and numerical attestation are
separate facts. A listed or loaded profile is not a behavioral benchmark.
Counters and attestation are displayed only when an engine supplies them;
unavailable values are not reported as zero.

Memory observations describe the engine's reported usage, not the profile
admission estimate. Allocated, reserved and available memory are distinct
measurements; unsupported measurements stay unavailable. Profile payload
budgets do not cap all engine or allocator memory.

Output rates use counter changes between samples from the same engine instance
and generation. They describe the observation window, not a maximum-throughput
benchmark. TTFT and draft acceptance are unavailable when the engine does not
expose the corresponding measurements.

See [profile management](profile-lifecycle.md) for loading, activation, rollback
and recovery, and [benchmarks](benchmarks.md) for controlled comparisons.

## Startup budget

Managed preparation records `startup_timeout` in seconds for every engine kind.
An explicit engine setting overrides the kit's `startup_timeout`; SGLang's Qwen4
and GLM Flash kits declare 1,800 seconds. Without a declaration, Python engine
kits use 900 seconds, or 3,600 seconds for native Docker launches. TensorFold
keeps its bundle default of 1,800 seconds. Activation uses the recorded budget.
Use only `startup_timeout` in kit manifests and setup plans.
