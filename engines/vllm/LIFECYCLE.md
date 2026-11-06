# Profile updates

The managed launcher starts a local profile administration socket alongside the
inference API. Its default location is `flashalign-runtime/admin.sock` beside
the deployment key. `--profile-state-dir` and `--profile-admin-socket` select
other locations. The socket requires an owned, mode-0700 directory and accepts
only processes running as the same operating-system user. The inference
signing key does not authorize profile administration.

Staging validates and copies a local artifact before loading it on workers.
Activation changes a versioned alias after every worker prepares the new
generation. Requests already using the previous generation keep their tensors
until completion. Cache keys include generation, profile digest and strength.
Retiring an old generation requires its requests, caches and graphs to drain;
cache eviction is an explicit administration operation.

The standard Qwen and Gemma adapters use the same lifecycle implementation.
New projection sites are installed during preparation. The native DeepSeek
adapter stages exact matrix bindings and their graph captures together. Its
checkpoint, operator, strength and topology requirements still apply.

`/v1/flashalign/status` exposes measured runtime metadata without request text.
`/v1/flashalign/capabilities` reports available administration features. Missing
performance counters remain missing; they are not estimates.

The managed native container exports `admin_socket` and
`admin_staging_directory` in its private deployment record. The staging
directory is mounted at the same path inside the container. The setup client
can copy an artifact there before staging it. Container and host administration
must use the same UID; the model, original profile and package mounts remain
read-only.

The lifecycle tests run on CPU. Native CUDA graph capture, mixed-generation GPU
parity, memory pressure and throughput require the engine qualification run.
