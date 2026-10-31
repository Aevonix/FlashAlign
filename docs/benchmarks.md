# Benchmark FlashAlign

Compare the same checkpoint, quantization, engine and generation settings with
FlashAlign off and with one loaded profile. Keep speculative decoding enabled
if it is enabled in the deployment. The benchmark client signs requests; it
does not change the engine or its caches.

The included tooling records responses and timings. It does not supply a
refusal score by guessing from the first few words of an answer. A benchmark
result needs both the responses and a documented scoring procedure.

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

Start with a predeclared small subset to validate the workflow. A 25-prompt
pilot is a pilot, not a full benchmark. Report every completed and failed case,
the selection seed, and both modes. Choose any demonstration case separately
from the aggregate score. Do not remove cases because their result is awkward.

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
manifest is operator-supplied provenance; the recorder cannot establish that
the server loaded the stated weights from a model name alone.

The same interface can select any compatible model/profile. The recorder does
not expand the model/engine compatibility matrix.

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

The runner alternates which mode goes first. Warmup requests are labelled and
excluded from measured statistics. It does not flush caches, and it does not
claim a cold or warm cache from request order. If a test needs a cold-cache
condition, establish and document that separately. Sequential runs control
interference better than concurrent side-by-side calls; they do not measure
maximum server throughput under load.

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

Missing values stay null. SSE chunks are never counted as tokens. Output token
counts can include reasoning. Report the exact metric beside any speed claim;
do not compare an estimate with another engine's internal decode timer.

The recorder also preserves the server's `usage.prompt_tokens_details.cached_tokens`
as `cached_prompt_tokens`. An absent value is unknown, rather than zero. Use
[long-context.jsonl](../benchmarks/long-context.jsonl) for a repeatable original
reading fixture, then repeat the normal recording command with that cases file.
Run it separately at each profile strength with `--scale`. Retain warmup records
to compare the first request at a new profile/strength with later requests.
Reported cache hits demonstrate reuse; they do not by themselves prove isolation.
Check that a new profile or strength cannot reuse a previous mode's prefix and
that each mode subsequently reuses its own prefix. The fixture's actual token
count depends on the tokenizer and appears in every completed record.

For a larger reading test, generate original fictional observatory records:

```sh
python benchmarks/prepare_long_context.py --paragraphs 1000 \
  --output /private/experiment/long-reading
```

Use its `cases.jsonl` with the same recorder command. Choose a paragraph count
that fits the deployment's context after reserving output space. The generator
does not claim a token count; verify `prompt_tokens` in the resulting record.
Its single retrieval answer is `Cedar, Tuesday`. This tests one retained fact
and cache reuse, not general long-context reasoning quality.

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
the input hash and selection parameters beside it. It does not mark imported
prompts as reviewed for public demonstration.

Use XSTest's full-compliance, partial-refusal and full-refusal taxonomy. For a
small pilot, independent manual review is practical: randomize and blind the
mode labels before review, then join judgments to `record_id`. For a larger
run, pin a semantic classifier and its rubric, retain its raw judgments
privately, and manually audit a declared sample. Do not use a refusal-phrase
matcher as the headline score.

Fill `judgments-template.jsonl` with one label per completed measured record
and the evaluator's name, revision and method. Then run:

```sh
python -m flashalign.benchmark summarize \
  --run /private/experiment/paired-run \
  --labels /private/experiment/judgments.jsonl \
  --output /private/experiment/scored-aggregate.json
```

The summary keeps safe and unsafe groups separate and reports the labelled
denominator. Partial labeling is explicitly incomplete. Full compliance on an
unsafe prompt is not interchangeable with harmful task completion. Use
HarmBench's behavior-aware semantic classifier for that latter measure; pin
the classifier revision, token truncation budget and behavior dataset.

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
when using the chat adapter. A generative multiple-choice variant needs its own name and
must not be labelled standard MMLU. Stop if the engine does not support the
required API; do not switch checkpoints or engines in one half of a pair.

The adapter's synchronous and asynchronous request paths have mock protocol
coverage. Full upstream task runs and engine-specific logprob compatibility
must be verified on the chosen serving deployment. Do not describe this as a
completed capability benchmark until those jobs run. Preserve task/dataset
revisions, few-shot examples, seed, generation budget and scoring version for
both halves. Export aggregate `results` and sample counts only after reviewing
the upstream result object for private paths or metadata.

## Make the side-by-side demo

The original [demo cases](../benchmarks/demo-cases.jsonl) are benign. If the
model answers both modes identically, show that result or try a separately
disclosed case. The exporter does not invent contrast, and a selected clip is
not a benchmark score.

```sh
python -m flashalign.benchmark demo \
  --run /private/experiment/paired-run --case linux-process \
  --repetition 0 --output /private/experiment/demo.html
```

Open the HTML locally. It embeds the FlashAlign wordmark and replays actual
captured text deltas at their original times. It labels the sequential requests,
warmup count and metric conventions. It omits reasoning text and displays no
invented running token counter. The page works offline and does not fetch fonts,
scripts or analytics. Review the complete clip before sharing it.

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

The WebM records playback at 1×. Any sped-up edit should state its speed.
The HTML and video contain the selected prompt and visible answer, so sharing
them is a deliberate export of that reviewed pair. Raw run files stay private.
Use `demo --note "..."` to display a visible qualification or limitation beside
the recording. A selected successful pair must not hide failures elsewhere in a run.
