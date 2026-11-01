# Benchmark FlashAlign

Compare the same checkpoint, quantization, engine and generation settings with
FlashAlign off and with one loaded profile. Keep speculative decoding enabled
if it is enabled in the deployment. The benchmark client signs requests; it
does not change the engine or its caches.

The included tooling records responses and timings. Refusal scores require a
separate, documented scoring procedure.

## Choose the measurement

| Question | Tool | What to report |
| --- | --- | --- |
| Does the profile retain general capability? | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MMLU and GSM8K scores, sample counts, task/dataset revisions and confidence intervals |
| Does it refuse benign requests unnecessarily? | [XSTest](https://github.com/paul-rottger/xstest) safe split | Full and partial refusal counts, scored with its three-class taxonomy |
| Does it comply with unsafe requests? | [HarmBench](https://github.com/centerforaisafety/HarmBench) | Official classifier results, classifier revision, exact behavior subset and output budget |
| What is the request overhead? | Included paired recorder | TTFT, total time, output tokens and the available decode measurement |

The selected upstream revisions are in
[upstream-sources.json](../benchmarks/upstream-sources.json). These are optional
tools. Their datasets, classifiers and dependencies are not included with
FlashAlign. Obtain them separately and retain their licenses and citations.
XSTest prompts are CC-BY-4.0; generated responses have separate provenance.

Start with a predeclared small subset. Report its size and selection seed,
completed and failed cases, and both modes. Keep all selected cases in the
aggregate, and label any demonstration clips as selected examples.

## Record a controlled comparison

Install FlashAlign in an environment with Python 3.10 or later:

```sh
python -m pip install .
```

Use a direct engine client configuration from setup. It must point to the same
engine for both modes and use `mode: "verified"`, with the deployment key in a
private file. An Always On gateway cannot provide the off control. The engine
must confirm the requested effective profile on each response; a missing or
downgraded profile is a failed request.

Copy [manifest.example.json](../benchmarks/manifest.example.json) and
[settings.example.json](../benchmarks/settings.example.json) to a private working
directory. Replace the manifest placeholders with the actual deployment. Hash
the loaded `profile.safetensors`, record the checkpoint revision and
quantization, and record whether speculation is active. Include engine commit,
weights digest, tokenizer revision and settings digest when available. The
manifest records operator-supplied provenance. Verify the loaded checkpoint
against the server configuration and use a [supported pairing](compatibility.md).

```sh
python -m flashalign.benchmark record \
  --config ~/.config/flashalign/client.json \
  --manifest /private/experiment/manifest.json \
  --settings /private/experiment/settings.json \
  --cases benchmarks/demo-cases.jsonl \
  --warmups 1 --repetitions 3 \
  --output /private/experiment/paired-run
```

Use a new output directory outside a Git checkout. The files are created with
private permissions. `records.jsonl` contains prompts and responses, including
reasoning when the server exposes it. Keep it private. `aggregate.json` contains
counts, provenance and timing statistics without prompts or responses. Inspect
its operator-supplied metadata before publishing it.

The runner alternates which mode goes first and excludes labelled warmups
from statistics. Caches are not flushed; establish and document cold-cache
conditions separately. Requests are sequential, so these measurements cover
per-request performance rather than maximum throughput under load.

TTFT is measured from the client request start to the first nonempty content
or reasoning delta. First visible answer text is reported separately. Total
time includes the whole request. Completion counts come from the server's
usage field. Some engines require `stream_options.include_usage`; if yours
rejects that option, remove it from both modes and leave unavailable counts
unreported.

There are three distinct speed fields:

- `engine_decode_tps`: server `timings.predicted_n / (predicted_ms / 1000)`,
  or TensorFold's `tensorfold.tokens_per_second` when a positive
  `tensorfold.seconds` decode interval is supplied. The raw record identifies
  the source and keeps reported draft, acceptance and round counts separately.
- `stream_decode_tps_estimate`: `(completion_tokens - 1)` divided by the
  interval between the first and last text/reasoning delta. Network buffering
  and speculative token bursts affect this estimate.
- `output_tokens_per_total_s`: completion tokens divided by the full client
  request time, including queueing, prefill and network time.

Missing values stay null. Token counts come from server usage and can include
reasoning. Report the exact speed metric; streaming estimates and internal
engine timers measure different intervals.

The recorder preserves `usage.prompt_tokens_details.cached_tokens` as
`cached_prompt_tokens`; an absent value is unknown. Use
[long-context.jsonl](../benchmarks/long-context.jsonl) for a repeatable original
reading fixture, then repeat the normal recording command with that cases file.
Run it separately at each profile strength with `--scale`. Retain warmup records
to compare the first request at a new profile/strength with later requests.
Check both cache reuse and isolation: a new profile or strength must not reuse
another mode's prefix, and each mode should subsequently reuse its own. The
fixture's token count depends on the tokenizer and appears in each completed record.

For a larger reading test, generate original fictional observatory records:

```sh
python benchmarks/prepare_long_context.py --paragraphs 1000 \
  --output /private/experiment/long-reading
```

Use its `cases.jsonl` with the same recorder command. Choose a paragraph count
that fits the deployment's context after reserving output space, and check
`prompt_tokens` in the resulting record. The fixture tests retrieval of one
fact (`Cedar, Tuesday`) and cache reuse.

## Score XSTest

Check out XSTest at the revision in the source manifest. Convert its original
CSV without copying the dataset into the FlashAlign repository:

```sh
python benchmarks/prepare_xstest.py \
  --csv /private/xstest/xstest_prompts.csv \
  --group safe --limit 25 --seed 20261030 \
  --output /private/experiment/xstest-pilot
```

Record the resulting `cases.jsonl` using the command above. The importer writes
the input hash and selection parameters beside it. Review prompts separately
before using them in a public demonstration.

Use XSTest's full-compliance, partial-refusal and full-refusal taxonomy. For a
small pilot, independent manual review is practical: randomize and blind the
mode labels before review, then join judgments to `record_id`. For a larger
run, pin a semantic classifier and its rubric, retain its raw judgments
privately, and manually audit a declared sample. Score meaning and compliance,
rather than matching refusal phrases.

Fill `judgments-template.jsonl` with one label per completed measured record
and the evaluator's name, revision and method. Then run:

```sh
python -m flashalign.benchmark summarize \
  --run /private/experiment/paired-run \
  --labels /private/experiment/judgments.jsonl \
  --output /private/experiment/scored-aggregate.json
```

The summary keeps safe and unsafe groups separate, reports the labelled
denominator and marks partial labeling incomplete. For harmful task completion,
use HarmBench's behavior-aware semantic classifier rather than the XSTest
compliance label. Pin its revision, token truncation budget and behavior dataset.

## Retained capability with lm-evaluation-harness

Install the pinned upstream `lm-evaluation-harness` checkout with its `api`
extra in a separate evaluation environment, alongside FlashAlign. The optional
adapter in `flashalign.benchmark_lm_eval` reuses upstream task and tokenizer
logic while signing each request and checking the effective profile.

```python
from flashalign import Client
from flashalign.benchmark_lm_eval import model_class
import lm_eval

with Client.from_config("/private/client.json") as client:
    model = model_class("completions")(
        flashalign_client=client,
        flashalign_profile="off",  # Repeat with the loaded profile name.
        base_url=client.base_url + "/completions",
        tokenizer="/models/the-exact-tokenizer",
        tokenizer_backend="huggingface",
        revision="PINNED_TOKENIZER_REVISION",
        max_length=8192,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["mmlu_high_school_computer_science", "gsm8k"],
        limit=25,  # Per task, not the total number of prompts.
        log_samples=False,
        random_seed=20261030,
        numpy_random_seed=20261030,
        torch_random_seed=20261030,
        fewshot_random_seed=20261030,
    )
```

Ordinary MMLU needs completion logprobs/echo support and the matching tokenizer.
A chat endpoint alone cannot supply that score. For generation-only tasks,
use `model_class("chat")`, a `/chat/completions` URL and the task's appropriate
chat template. Pass `apply_chat_template=True` to `lm_eval.simple_evaluate`
when using the chat adapter. Label generative multiple-choice variants
separately from standard MMLU. Both halves of a comparison must use the same
checkpoint and engine with the required API support.

The adapter's synchronous and asynchronous paths have mock protocol coverage;
full task runs and engine-specific logprob compatibility need verification on
the chosen deployment. Preserve task/dataset revisions, few-shot examples,
seed, generation budget and scoring version for both modes. Review aggregate
`results` and sample counts for private paths or metadata before publishing.

## Make the side-by-side demo

The original [demo cases](../benchmarks/demo-cases.jsonl) are benign. Both modes
may answer identically. Any additional case should be disclosed separately
from the aggregate benchmark.

```sh
python -m flashalign.benchmark demo \
  --run /private/experiment/paired-run --case linux-process \
  --repetition 0 --output /private/experiment/demo.html
```

Open the HTML locally. It embeds the FlashAlign wordmark and replays captured
text deltas at their original times, with labels for sequential requests,
warmups and metrics. Reasoning text is omitted. The page works offline without
external fonts, scripts or analytics. Review the complete clip before sharing it.

Optional video capture requires Playwright and Chromium:

```sh
python -m pip install playwright
python -m playwright install chromium
python scripts/record-benchmark-demo.py \
  /private/experiment/demo.html /private/experiment/demo.webm
ffmpeg -i /private/experiment/demo.webm -c:v libx264 -pix_fmt yuv420p \
  /private/experiment/demo.mp4
ffmpeg -i /private/experiment/demo.webm \
  -vf 'fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse' \
  /private/experiment/demo.gif
```

The WebM records playback at 1×; label any speed changes. The HTML and video
include the selected prompt and visible answer. Keep raw run files private.
Use `demo --note "..."` for a visible qualification or limitation, and publish
aggregate results alongside selected clips.
