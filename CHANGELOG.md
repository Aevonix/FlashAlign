# Changelog

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
- Shared FlashAlign terminal design with a sharp pixel logo, a brief startup
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
