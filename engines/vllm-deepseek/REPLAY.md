# Persistent FlashAlign nonce store

`flashalign_replay.py` is a standalone Python standard-library module. Vendor
the file unchanged into an engine package and import `ReplayStore` from the
chosen package path. There is no model, profile, tensor or derivation code.

Call `deployment_replay_store(deployment_key, path=None)` once per API
process. It uses an explicit path, then `FLASHALIGN_REPLAY_STORE`, then
`$XDG_STATE_HOME/flashalign/replay/nonces.sqlite3` (falling back to
`~/.local/state/flashalign/replay/nonces.sqlite3`). It creates missing
directories with an owned0700 leaf and never changes existing permissions.
`ReplayStore(path, deployment_key)` remains available for explicit construction.
Only after
the existing raw-body HMAC check succeeds, call:

```python
accepted = replay_store.consume(nonce, authenticated_timestamp + 61, now=now)
```

Enable owner mode only if `accepted` is true. A false result includes replay,
capacity exhaustion, corruption, lock timeout, missing/replaced storage or
another filesystem failure. Constructor errors must also fail closed. The
module does not itself validate a request's signature or timestamp window.

Keep the database in an owned directory with mode0700, outside the repository
and model directories. The database is an owned0600 regular file. Persist the
same path across API restarts whenever the same deployment key is retained.
Use a local filesystem with working SQLite locking and fsync; do not place it
on a read-only or ephemeral container layer. Do not clean or restore an old
copy while the same key can authenticate in-window requests. A directory or
database symlink, a hard-linked file, wrong permissions or a replaced inode
is rejected. Creating/changing deployment state belongs to the installer;
this module does not loosen permissions.

SQLite transactions atomically insert a key-derived nonce digest under a
key-derived namespace. Only those two digests and expiry timestamps are
stored. No raw nonce, request body, user ID, authentication header or key is
stored. Expired records are removed before capacity checks. The default
100,000-record cap applies to the entire file, including namespaces from
recent key rotation. Records expire at the same `timestamp+61` used by the
existing in-memory gate, including future-dated accepted requests.

Run `python3 -m unittest -v test_flashalign_replay` in this directory. The
fifteen tests cover actual process restart and future timestamps, independent
concurrent connections, capacity/expiry, key separation and fail-closed
filesystem/SQLite cases, and consistent state-location precedence. They use
synthetic keys only.
