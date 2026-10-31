# Developing Aevonix FlashAlign

Use Python 3.10+ and a supported Node runtime. The CI matrix uses Python 3.10 and 3.12
with Node 24.22.1 and 26.12.1. Install the pinned test dependencies inside an
isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r .github/requirements-ci.txt
.venv/bin/python -m unittest discover -s hermes-owner-mode/tests
node --check openclaw-owner-mode/index.mjs
bash -n install.sh
```

The tests use temporary credentials and a local engine fixture. They cover
identity binding, request integrity, persistent approvals, revocation, tool
scope, delegated work, configuration preservation and the Node/Python bridge.
Run separate checks for native harness installation and real-engine behavior.

Run the shared replay-store tests separately:

```sh
PYTHONPATH=engines/common .venv/bin/python -m unittest discover \
  -s engines/tests -p test_flashalign_replay.py
```

The CI dependency file includes NumPy. Omit the `-p test_flashalign_replay.py`
filter to also test the shipped gates and identical vendored implementations. The replay tests
cover independent processes, restart windows, private storage, capacity and
failure handling. Test model execution and speculative decoding with each
engine's runtime checks.

Keep request signing, contact verification and tool permissions distinct. Test
both approved and unapproved paths when changing authorization. Preserve tool
arguments, results and exceptions. Do not put real contact IDs, keys or request
content in test fixtures or issue reports.

Use the supplied [Aevonix artwork](assets/brand/README.md) for product graphics.
Write interface text about what users can configure, observe and verify.

FlashAlign is licensed under [AGPL-3.0-only](LICENSE), with a separate
[commercial licensing option](COMMERCIAL.md). Before a contribution can be
merged, each contributor must accept the [contribution terms](CLA.md) for
that contribution. These let Aevonix use and license contributions in
closed-source products and services, and pass those rights to others, without
further approval or royalties. Contributors retain their copyright.

Accept in the pull-request acknowledgment or an authenticated submission
message. No signed form or signed commit is required. Maintainers check each
rights holder's acceptance and third-party notices before merging. Send private
rights disclosures to **contact@aevonix.com**, not a public issue.
Record the accepting account or entity, acceptance link and date, covered
commit IDs, and the terms version and file hash. Retain the record when
squashing commits; keep private legal details outside the public repository.

The **Contribution acknowledgment** check reads the declaration in the pull
request body and creates a JSON artifact for the current head commit. It records
the submitting account, observed event actor and time, PR link, terms version
and hash, and covered head commit. It does not post comments, require a signed
form, or replace the maintainer's review of other rights holders and notices.

The check runs its parser and reads the terms from the reviewed base commit.
The artifact expires after 90 days. Before merging, retain the acceptance link
and artifact in the contribution record, including when commits are squashed.
Do not put private rights disclosures in workflow artifacts.
