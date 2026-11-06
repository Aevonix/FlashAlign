# Testing and release coverage

FlashAlign tests shared contracts on every release. Live qualification follows
the code that changed; it does not require rerunning every model, engine,
quantization and hardware combination. The [validation record](validation.md)
identifies the configurations exercised and the limits of that evidence.

## Choose coverage from the change

| Change | Required coverage |
| --- | --- |
| Shared authorization, identity, tool policy, lifecycle or setup code | Core and shared-engine regressions, affected adapter contract tests and installed-package checks. Use a representative live backend when the change crosses a boundary that fixtures cannot verify. |
| Profile operator or row selection | Small numerical fixtures for the affected operator family, stock arithmetic bypass, mixed rows, strength handling, malformed inputs and generation isolation. Run relevant device checks when execution code changes. |
| Engine adapter, kernel, graph path, distributed protocol or serving-runtime version | Affected adapter regressions and targeted live qualification of the changed execution paths. Include cache isolation and active speculative decoding where supported. |
| New model architecture, checkpoint representation or quantization support | Artifact and mapping checks, affected operator tests and a representative full-model run for the new path. |
| Packaging, installer or backend launch configuration | Installed-artifact checks plus a representative activation and shutdown test for the affected launch mechanism. A small CPU container can exercise Docker resource ownership and cleanup without loading a model. |
| Documentation or branding | Link, formatting and claim checks. No inference rerun unless the change makes a new compatibility or performance claim. |

Parameterize fixtures by capability: operator family, batch layout, lifecycle
state, cache generation, process ownership and deployment mechanism. Use small
synthetic profiles and temporary credentials. Avoid fixtures tied to a real
contact, deployment or large checkpoint when a shared contract is being tested.

An unresolved defect requires focused coverage of the affected behavior.
Unrelated configurations need not be rerun. Record known unqualified paths
separately and keep their support claims within the available evidence.

## Shared regression commands

Run from the repository root in an isolated environment with the dependencies
from [CONTRIBUTING.md](../CONTRIBUTING.md). The
[CI workflow](../.github/workflows/ci.yml) pins its Python, Node and tensor-test
environments.

```sh
python -m unittest discover -s hermes-owner-mode/tests
PYTHONPATH=engines/common python -m unittest discover -s engines/tests
python -m unittest discover -s engines/common -p test_sglang_lifecycle.py
```

The core suite includes identity and authorization, harness routing, profile
validation, setup and lifecycle fixtures. Shared-engine tests cover replay
protection and HTTP boundaries across the shipped gate implementations.

Relevant test entry points include:

- [External resource cleanup](../hermes-owner-mode/tests/test_external_cleanup.py):
  process reaping, bounded helpers, failure, retry, activation ownership and
  integrity checks. A stopped process alone must not imply a stopped container.
- [Local unit ownership](../hermes-owner-mode/tests/test_local_unit_ownership.py)
  and [managed shutdown](../hermes-owner-mode/tests/test_managed_stop.py):
  owned process handling and shutdown state.
- [Profile lifecycle](../hermes-owner-mode/tests/test_profile_lifecycle.py):
  generation transitions and administrative contracts.
- [Native artifact fixtures](../hermes-owner-mode/tests/test_native_profile.py):
  encoded payload validation, partitioning, download integrity and setup contracts.

## CPU operator and adapter checks

With the CPU tensor dependencies installed as specified in CI:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=engines/vllm python -m pytest -q engines/vllm/tests
CUDA_VISIBLE_DEVICES='' PYTHONPATH=engines/sglang python -m pytest -q engines/sglang/tests/test_runtime.py engines/sglang/tests/test_graph_contract.py
python -m pytest -q engines/vllm-deepseek-native/tests
```

The projection [runtime fixtures](../engines/vllm/tests/test_runtime.py) check
small tensors, selected rows, strength interpolation and stock bypass. The
[generation fixtures](../engines/vllm/tests/test_generations.py) exercise
different profile generations, retirement and failed-hook recovery. SGLang's
[runtime](../engines/sglang/tests/test_runtime.py) and
[graph contract](../engines/sglang/tests/test_graph_contract.py) fixtures check
its row layouts and generation contracts.

Native [component fixtures](../engines/vllm-deepseek-native/tests/test_components.py)
check encoded partitions, execution classification and selected-row reference
behavior. Its [cleanup tests](../engines/vllm-deepseek-native/tests/test_cleanup.py)
exercise Docker ownership, daemon failures, durable creation intent and retry.
These are CPU contracts; they do not execute the native GPU kernels.

For each newly supported operator family, add its small reference fixtures and
explicit unsupported-capability checks. A passing test for one operator does
not establish another operator's implementation.

## Installed package checks

Build and check the actual wheel, rather than relying on checkout imports:

```sh
python tools/package_assets.py --check
python -m unittest discover -s packaging -p 'test_*.py'
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python packaging/check_wheel.py dist/aevonix_flashalign-1.2.0-py3-none-any.whl
```

Use the version being released in the final command. The offline checker
installs into a temporary directory, runs outside the checkout, compares module
and asset bytes, discovers engine kits and profiles, and checks wizard entry
points. See [package checks](../packaging/README.md).

## Live qualification and retained results

Choose a representative backend that exercises the changed boundary. Check
effective off/on selection, mixed requests where applicable, separate cold/warm
cache namespaces, profile generations, preserved speculative decoding, and
verified shutdown. A launch or cleanup change may need only its corresponding
lifecycle checks. Numerical, kernel or runtime changes need relevant device
execution as well.

CPU references and mocks do not prove GPU kernels, graph capture, distributed
execution or full-model behavior. A live pass applies to its recorded scope;
it does not establish maximum context, answer quality or throughput without
those measurements. Carry forward earlier qualification only when the relevant
implementation, artifact and runtime dependencies are unchanged.

Keep failed and incomplete runs in the record. Later generic passes do not
rewrite an earlier model-specific failure as a pass. Release notes must
distinguish shared regression coverage, qualified compatibility and any known
unqualified deployment path.
