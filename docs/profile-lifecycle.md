# Load and switch profiles

A compatible engine can load another profile while serving the same model.
Existing requests keep the profile generation they started with. New requests
use the updated alias after activation succeeds.

This requires an engine that reports `hot_loading: true`. A model's existing
engine support does not by itself establish hot-loading support. The profile
must match the loaded checkpoint and an operator the engine implements.

## Local administration

Managed setup writes the administration socket into the client configuration.
The socket accepts only the operating-system user running the deployment. An
inference signing key cannot authorize profile administration, and the public
HTTP API has no profile upload endpoint.

After setup:

```sh
bash ./flashalign.sh profile list
bash ./flashalign.sh profile stage /profiles/my-profile --json
```

Staging validates an immutable copy and returns its generation ID. It does not
change the active alias. Copy that ID into the activation command; use the
current alias revision shown by `profile list`:

```sh
bash ./flashalign.sh profile activate personal \
  --generation GENERATION_ID --expected-revision 0
```

Revision `0` creates a new alias. Updating an existing alias requires its actual
revision. If another administrator changes it first, activation fails instead
of overwriting that change.

A compatible container deployment uses the private shared staging directory
recorded by setup. The client copies the profile there automatically.

For a TensorFold deployment on another host, setup records its SSH administration
connection. The same commands transfer a validated profile to that host and
remove the temporary upload after staging.

## Roll back or remove a selection

Rollback selects a retained generation and advances the alias revision:

```sh
bash ./flashalign.sh profile rollback personal \
  --generation PREVIOUS_GENERATION_ID --expected-revision CURRENT_REVISION
bash ./flashalign.sh profile deactivate personal --expected-revision CURRENT_REVISION
```

Deactivation prevents future requests from selecting that alias. Requests
already admitted keep their generation. It does not change contact approvals,
Always On settings or tool permissions.

A generation can be retired only after aliases, requests, caches and graphs no
longer reference it. Retirement returns an error while it is still in use:

```sh
bash ./flashalign.sh profile retire GENERATION_ID
```

`profile evict-cache` asks the idle engine to evict cached profile generations.
Ordinary profile switching preserves caching and speculative decoding. Use
eviction when you want to release cached generations.

## Recovery and resource limits

Loading keeps the old and new profile resources available together. For vLLM,
SGLang and llama.cpp, profile budgets limit declared payload and reservation
estimates. They are not hard limits on physical host or GPU memory. Allocator
rounding, retained allocation blocks and engine caches can use additional memory.
An estimate that exceeds the configured budget is rejected before activation;
an allocation failure also leaves the active profile unchanged. A native matrix
profile can require substantially more space than a projection. The model
weights are not reloaded during profile activation.

If a worker or administration operation fails after a durable decision, new
profile requests remain fenced until recovery completes:

```sh
bash ./flashalign.sh profile recover
```

Recovery reconciles workers with the saved decision. It does not choose another
profile or turn a failed owner request into a standard request. Keep the private
lifecycle state directory across engine restarts.

Changing the model, serving quantization, operator support or worker topology
still requires a compatible engine setup. Hot loading cannot make an unsupported
profile compatible, and it does not make fixed-strength operators tunable.
