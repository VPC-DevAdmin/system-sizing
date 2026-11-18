# OpenCompass check against Artificial Analysis (plan)

Status (2026-11-05): set up and queued on the XE7740 behind the last H3.4 capacity runs.

- **Installed:** OpenCompass at commit `3ab1d83` in `/data/capsim-planner/opencompass`.
- **Data:** HLE gives 2,158 text-only questions, AA-LCR v1.1 has 100, and Omniscience has the 300 public questions.
- **SciCode:** not set up yet. See "SciCode: decision needed" below.
- **Queue jobs:**
  - `OC0-gptoss120b-1gpu-high`: a gate run of 20 questions per test.
  - `OC0-judge-qwen35-4gpu`: grades the gate run.
  - `OC-gptoss120b-1gpu-high` and `OC-gptoss120b-1gpu-low`: the full runs. They are skipped automatically if the gate's judge fails.
  - `OC-judge-qwen35-4gpu`: grades the full runs.
- **Code (uncommitted):**
  - `scripts/opencompass_job.py`: the job script.
  - `scripts/oc/`: the HLE prompts, the final-answer cleaner and the judge parser.
  - `planner_campaign.py`: now has the `SCORE_JOBS` table.

## SciCode: decision needed

OpenCompass's SciCode does not match AA's setup:

- It sends the subproblems as a multi-turn chat. AA builds one prompt per step with the background, as the official harness does.
- It times out each step at 120 seconds. AA uses 300 seconds.

The faithful route is the official SciCode repo's Inspect AI task. It needs a second install, which is not part of the approved plan and was blocked, so SciCode waits for your choice:

1. Install the official harness.
2. Use OpenCompass's version, with the gaps noted.
3. Drop SciCode from the check.

## Purpose

Find out whether our own runs of Artificial Analysis's (AA) tests reproduce AA's published scores. If they do, the same setup can score the modes AA has not published (Qwen3.6, Qwen3.5 397B and Gemma 4 with reasoning on) on AA's own tests. The planner keeps AA's published index either way. Our runs are a supporting check, never a replacement.

## What can be checked on gpt-oss-120b

AA publishes these scores for gpt-oss-120b (read from AA's model pages on 2026-11-05):

| Test | Index weight (v4.3) | AA, high effort | AA, low effort | Check on gpt-oss? |
|---|---|---|---|---|
| HLE, 2,158 text-only questions | 10% | 19.6% | 5.9% | yes |
| SciCode, 288 test subproblems × 3 | 10% | 34.0% | not published | yes, high only |
| AA-LCR v1.1, 100 questions × 3 | 5% | 52.0% | 46.0% | yes |
| AA-Omniscience v1.1 | 15% | not published | not published | no (see below) |
| CritPt | 10% | 1.1% | 0% | no (see below) |

The three testable items carry 25% of the index weight. The earlier 50% figure counted all five.

- **Omniscience:** AA has no gpt-oss score to compare with. Only 300 of AA's 3,000 questions are public, so our score would be on a different sample. We can run it and report it, but not use it as a check.
- **CritPt:** its answers are hidden. OpenCompass can only grade it by sending model outputs to AA's grading server, and AA's gpt-oss scores are about 1%, so a match would prove nothing. Left out unless you approve sending outputs to AA.

## Pass bands (set before running)

A result passes if it lands within about two standard errors of AA's number:

| Test | High effort | Low effort |
|---|---|---|
| HLE | 19.6 ± 1.7 | 5.9 ± 1.0 |
| SciCode | 34.0 ± 5.5 | not checked |
| AA-LCR | 52.0 ± 5.7 | 46.0 ± 5.7 |

HLE is the tight check. SciCode and AA-LCR are small, so a 4-point gap there is not evidence of a broken setup.

## Settings to match AA

OpenCompass's stock configs differ from AA's settings in ways that would cause a false mismatch:

- **HLE:** OpenCompass's configs use a `\boxed{}` prompt and a generic grader. AA uses HLE's official system prompt (Explanation / Exact Answer / Confidence) and HLE's official equality-checking judge prompt. We write our own config with the official prompts, and check that the text-only filter gives 2,158 questions.
- **SciCode:** use the background-included config (`scicode_wbg`), only the 288 test subproblems, 3 repeats, and a 300-second step timeout. The stock `max_out_len` of 4,096 would cut off reasoning, so it must be raised.
- **AA-LCR v1.1:** OpenCompass's config already uses AA's v1.1 judge prompts and 3 repeats. Prompts are about 100K tokens against gpt-oss's 131,072-token context, so long reasoning can run out of room. Those answers are counted as cut off, separately from errors.
- **Sampling:** temperature 1.0 and top_p 1.0 (OpenAI's recommendation, as in our LiveBench runs), `reasoning_effort` high or low, and the `openai_gptoss` reasoning parser. We confirm from the engine log that the effort setting reaches vLLM.
- **Output limit:** about 100K tokens for HLE and SciCode (AA uses the lab's maximum), and whatever room is left for AA-LCR.
- **Timeouts:** a long client timeout and modest concurrency (16–32), so no answer fails on a harness timeout. LiveBench lost about 70 answers that way.
- **Engine:** the same build as the gpt-oss capacity jobs.

## Judge

AA grades HLE, AA-LCR and Omniscience with GPT-5.6 Luna at medium effort.

- **Step 1, local judge (no cost):** Qwen3.5 397B with thinking off at temperature 0, served on four GPUs, using the same judge prompts.
- **Step 2, only if a test misses its band:** re-grade a sample with AA's judge model, to tell whether the miss comes from the judge or from the setup. This needs the exact OpenAI model id and price confirmed first, a cost estimate and your approval. Only the final answers are sent, never the documents or reasoning.

## How it runs

- **Install:** OpenCompass in its own uv environment on `/data`, which has 250 GB free (`/data2` and `/data3` are 98–99% full). Pin the commit, since the AA datasets were added after release 0.5.4.
- **Data:**
  - HLE, which needs the HLE terms accepted on Hugging Face first (see below).
  - AA-LCR v1.1 at a pinned revision.
  - AA-Omniscience public v1.1.
  - SciCode test data (`scicode_test_data.zip` from OpenCompass's storage release).
- **Queue jobs (`kind: opencompass`):** run after the remaining H3.4 capacity runs, in the same way as the LiveBench jobs.
  1. `OC-gptoss120b-1gpu-high`: answers only, 1 GPU.
  2. `OC-gptoss120b-1gpu-low`: answers only, 1 GPU (HLE and AA-LCR).
  3. `OC-judge-qwen35-4gpu`: grades both runs.
  4. SciCode's generated code runs in a `--network none` container, because OpenCompass's evaluator otherwise runs it directly on the host.
- **GPU time:**
  - High effort: about 4–5 GPU-hours, mostly HLE.
  - Low effort: under 1 GPU-hour.
  - Judging: about 1 hour on four GPUs.
  - The answer runs can be split across GPUs, which brings the total to roughly half a day of wall time.
- **Cloud cost:** none, unless Step 2 of the judge is needed.

## Before it can start

1. **You:** accept the HLE dataset terms on Hugging Face (`cais/hle`), signed in as the account whose token is on the XE7740. The lab token currently gets a 403 on the data file.
2. **The queue:** the four H3.4 capacity runs finish.

## After the check

- **Pass:** run the same tests on the modes AA has not published, and report them per test beside AA's published numbers. They are not combined into an index score, because they cover only part of the index.
- **Miss:** find the cause (prompt, answer extraction, judge or cut-offs) before running anything else.
