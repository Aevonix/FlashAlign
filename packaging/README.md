# Release package checks

The wheel includes engine integration kits, starter profiles and the OpenClaw
adapter under `owner_mode/_assets`. Their original directory layout and license
files are preserved. The large DeepSeek native profile payload retains its
declared download step.

`assets.json` is an explicit list of tracked resources. After adding or removing
setup assets, run `python tools/package_assets.py` and review the changed list.
`python tools/package_assets.py --check` verifies that tests, caches and
untracked files have not become package inputs.

Build the wheel, then run the offline installation check with an interpreter
that has the project's runtime dependencies installed:

```sh
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python packaging/check_wheel.py dist/aevonix_flashalign-1.2.0-py3-none-any.whl
python -m unittest discover -s packaging -p 'test_*.py'
```

The check installs into a temporary directory, runs from outside the checkout,
compares all resource bytes, discovers every engine kit and starter profile,
and checks the wizard and its resume command. It starts no engine, downloads no
model and uses no GPU. Source archives include the same resource inventory.
