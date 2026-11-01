# Inspect a profile

Inspect a local profile before setting up an engine:

```sh
flashalign-profile inspect /path/to/profile
flashalign-profile inspect /path/to/profile --details
flashalign-profile inspect /path/to/profile --json
```

The same command is available as `flashalign-client profile inspect`. From a
repository checkout, it also runs with Python's standard library alone:

```sh
python3 hermes-owner-mode/profile_inspector.py inspect profiles/qwen38-27b-v1 --json
```

Inspection reads the supplied files. It does not download weights, load a model,
contact an engine, read deployment credentials, install a profile or restart a
service.

The report includes the model and revision, operator family, required features,
site inventory, file sizes and SHA-256 hashes. For compact profiles it reports
basis ranks and checks finite tensor values, positive gains where required, and
orthonormal basis rows. For the native DeepSeek format it streams the payload
and validates native encodings, tensor hashes and every declared tensor-parallel
partition. Native matrices have no basis rank. JSON and detailed terminal
output include the applicable aggregate checks. Vector entries
and freeform source/status metadata are omitted. Status labels in a manifest
are not treated as independently verified evidence.

Computed hashes identify the files that were inspected. The report separately
states whether a supplied checksum record verified both files, only some files,
or was absent. A computed hash alone does not authenticate a publisher.
Compact-profile validation and hashes use the same in-memory snapshots. Native
validation uses the bounded streaming reader and declared payload, tensor and
partition hashes. If a file changes later, its new contents require a new
inspection.

Inspection validates the artifact. Use engine activation checks and model
evaluations to assess runtime behavior.

The command exits with status `0` for a valid artifact, `2` for invalid or
unsupported files, and `3` when the configured work budget stops inspection.
In that last case, `artifact_valid` is `null`; it is not an invalid-file verdict.
Basis checks default to at most 50 million scalar products. For a larger trusted
compact profile, set an explicit higher limit with `--max-basis-work`. Compact
payloads remain limited to 256 MiB and are held in memory while validating;
inspection can use more memory than the file size. The native DeepSeek payload
uses a separate streaming path, so its 1.09 GB file is not loaded as one matrix
in memory. A missing native payload is an incomplete local artifact and fails
offline inspection. Setup can retrieve its declared public asset; the inspector
does not download it.

JSON uses schema `flashalign.profile-inspection.v1` and keeps
the artifact result separate from unassessed runtime and behavioral properties.
See [profile compatibility](compatibility.md#engines-and-models) and the
[engine contract](engine-contract.md) before deployment.
