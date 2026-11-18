# Changelog

## Unreleased

- Profile sets: a route can apply up to eight profiles to one request, each with its own access rule, failure rule and strength.
- An optional profile the engine does not have is dropped and logged; the request continues with the remaining profiles. Mark a profile `fail: required` to fail the request instead. This replaces the HTTP 400 that Verified contacts mode returned for an unknown profile.
- The direct client accepts a list of profile names and a strength per profile.
- Digest pins: a profile rule can pin the content digest the engine reports. Audit rows record the reported digests, and a mismatch is logged as `OWNER_MODE_DIGEST_DRIFT`. `status` and `doctor` compare a pinned `profile_digest` in the client configuration.
- DeepSeek native kit: low-rank profiles (`input_low_rank_delta_v1` at `attn_out` and `shared_expert_out`) load beside the native profile with `--low-rank-profile`, and one request can select a set of them. Checked on four SM120 GPUs; see `docs/validation.md`.

## 1.2.0

- Keep one active Hermes plugin registration per process across duplicate loads and reloads.
- Include fixed identity-verification reasons in each request's audit record.
- Profile staging, activation, rollback and retirement through local administration.
- Immutable generations and worker recovery contracts for serving adapters.
- Status, doctor and terminal dashboard commands.
- Pinned FR443 collection, durable semantic judging and evidence exports.
- Distinguish a busy evaluation lock from invalid evaluation data.
- Qwen FR443 results and a paired SGLang timing study with reproducible data.
- Artifact-bound profile evaluation and controlled calibration sweeps.
- Shared request authentication and extensible engine, model, channel and profile registries.
- Complete runtime installation for managed engines and harnesses.
- Include engine kits, starter profiles and harness assets in the Python package.
- Check container memory limits before starting native DeepSeek.
- Automatic native profile-control startup and SSH profile staging for remote TensorFold deployments.
- Preserve saved profile state across restarts, including an empty inventory or deactivated aliases.
- Keep managed SSH forwards attached to their supervisor when SSH multiplexing is enabled.
- Record managed launch attempts before starting workers and preserve ownership through interrupted starts.
- Preserve the active Hermes policy and refresh message observers across plugin reloads and host renderer replacements.
- Share one configurable shutdown deadline across managed engine workers.
- Verify external engine resources are stopped before confirming managed shutdown.
- Report reused prefix tokens in the DeepSeek TensorFold runtime.
- Handle atomic status-file replacement without interrupting a managed engine.

## 1.0.0

Initial Aevonix FlashAlign release:

- FlashAlign identity, original vector artwork and mobile-compatible repo banner.
- Hermes and OpenClaw adapters with signed model requests.
- Persistent approvals scoped to channel, account and contact.
- Extensible channel registry, opt-in Telegram identities and revocable local
  terminal sessions for native Hermes and OpenClaw.
- Always On activation for every routed request, including all tool and subagent calls.
- Managed Always On API gateway for clients that do not use a harness or the SDK.
- Tool defaults and exact-name overrides for model access and child inheritance.
- Setup wizard with harness detection, model discovery and compatibility checks.
- Shared FlashAlign terminal interface with a pixel logo, a brief startup
  animation, keyboard selection panels and plain-text fallbacks.
- Five included experimental starter profiles for GLM, DeepSeek, Qwen and Gemma,
  fitted and packaged by Aevonix from [credited checkpoints](profiles/SOURCES.md).
- TensorFold 1.1.9 support for all five starter models, with tested hardware and
  checkpoint formats, active speculative decoding and separate prefix caches.
- Direct-engine setup, Python client and reusable custom-harness bridge client.
- Native plugin activation for Hermes and OpenClaw, with bridge startup checks.
- Configuration backups and local contact/tool administration.
- Native bindings for voice, attachments, forwarded/quoted messages and queued groups.
- AGPL-3.0-only software licensing, a separate commercial option and
  submission-based contributor terms with commercial sublicensing rights.
