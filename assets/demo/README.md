# FlashAlign promo video

[Watch the 40-second promo](flashalign-promo.mp4) (1080p, 60 fps, with sound).
The [README preview](flashalign-promo-preview.gif) is a silent excerpt of its
first 8 seconds.

The promo is an edited video. Its model text comes from three recorded
DeepSeek V4.1 Flash 1017 request pairs (EXL3 3.5-bpw, TensorFold 1.1.9,
DSpark active, temperature 0, seed 20261030, thinking off), with FlashAlign
off and on (`deepseek41-flash1017-v1`, strength 1). Each pair was recorded
twice with matching text. The prompts are OR-Bench Hard-1K rows 321, 244 and
194, shown exactly as tested.

Every response on screen is an exact, continuous excerpt of the recorded text;
an ellipsis marks omitted text. In all three examples the stock model also
pointed to other help after refusing; the video shows only the refusal. The
recipe's glow claims were not tested.

The text reveal is animated for the edit, so the video does not show response
speed. These are selected examples, not a benchmark score; see the
[measured results](../../docs/benchmark-results.md). The setup scene shows the
installer's real output, shortened. The music and sound effects were made for
this video.

Prompts: [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) by
Justin Cui, Wei-Lin Chiang, Ion Stoica and Cho-Jui Hsieh, revision `e36d8b80`,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Prompt wording is
unchanged.
