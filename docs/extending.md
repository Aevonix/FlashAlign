# Extend FlashAlign

Runtime extensions come from trusted installed code. Profile manifests, model
requests and remote metadata cannot name Python modules to import.

| Extension | Interface | Responsibility |
| --- | --- | --- |
| Engine | `owner_mode.engine_adapters` | Compatibility checks, preparation, selection and probes |
| Model | `owner_mode.model_adapters` | Model IDs and supported request/reasoning codecs |
| Artifact or operator | `owner_mode.profile_adapters` | Bounded parsing, tensor layout and semantic validation |
| Channel | `owner_mode.channel_plugins` | Explicitly enabled installed identity adapters |
| Evaluation suite | `flashalign.benchmark_suites` | Pinned corpus identity, row validation and purpose |
| Scorer | `flashalign.benchmark_judge` | Rubric, parameters and verdict mapping |
| Runtime lifecycle | `owner_mode.profile_lifecycle` | Immutable generations, worker transactions, leases and recovery |

Installed distributions can declare these Python entry-point groups:

| Group | Value or zero-argument factory result |
| --- | --- |
| `flashalign.engine_adapters` | `EngineAdapter` |
| `flashalign.model_adapters` | `ModelAdapter` |
| `flashalign.profile_adapters` | `OperatorAdapter` or `ArtifactFormatAdapter` |

For example, a distribution can expose a model factory in its package metadata:

```toml
[project.entry-points."flashalign.model_adapters"]
example-model = "example_extension:build_model_adapter"
```

The registry loads installed entries on first access, using the same discovery,
distribution-metadata checks and typed loader as channel plugins. Entry-point
names use the channel loader's lowercase adapter-ID syntax. These three groups
load all installed entries; they do not use the channel approval store.
Channel identity extensions still require explicit local selection and matching
distribution/version pins. Installing other adapter types never enables a
channel or approves a contact.

Import errors, factory exceptions and wrong result types propagate to the
caller, as they do for a selected channel plugin. A failed load does not publish
a partial registry. Registries reject duplicate or ambiguous identities,
including collisions with built-ins. Successful discovery is cached for the
process; restart to pick up changed installed distributions.

Trusted code that is already imported can also call `register_engine_adapter`,
`register_model_adapter`, `register_operator` or `register_artifact_format`.
Merely placing a module on disk does not import or register it. A model can
reuse an existing request codec; a new codec must implement its validation and
request conversion before it is advertised. Registration alone does not add
engine launch behavior: engine callbacks must implement the supported setup
contract for their backend.

Artifact-format adapters receive a bounded reading context. Operator adapters
declare their tensor shapes, basis and positivity requirements. Reject unknown
requirements and unsupported layouts before allocating runtime resources.
The included native-matrix format retains its streaming validator.

A serving adapter implements the lifecycle backend contract: validate, estimate,
reserve, prepare, fence, commit, resume, abort, reference accounting, retirement
and reconciliation. Workers acknowledge the same transaction and immutable
generation. Public names are aliases, never cache identities. Every cache and
captured graph must retain the generation it actually uses.

New adapters should test mixed standard/profile batches, old and new generations
in flight, failed worker preparation, replay rejection, recovery, cache release,
and active speculative decoding. A registry test proves an extension can be
registered; it does not establish hardware or model support. Publish compatible
layouts separately from configurations actually exercised on hardware.

See [channel adapters](channels.md), [the engine contract](engine-contract.md),
[profile management](profile-lifecycle.md) and [benchmarks](benchmarks.md).
Profile fitting tools are not included in this repository.
