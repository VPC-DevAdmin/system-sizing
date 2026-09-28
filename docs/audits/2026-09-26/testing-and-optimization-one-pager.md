# How we find the right AI configuration

**A broader search. Realistic demand. A clearer customer decision.**

Our testing explores how hardware, AI models and serving software work together. We look for configurations that deliver more useful capacity while meeting the response times customers need.

## 1,728 setting combinations in one example

That is one engine and one model family, before hardware compatibility checks. Adding engines, models and workloads expands the possibilities further.

### Search broadly, then focus the testing

We vary model formats, GPU arrangements, memory use and how requests are grouped. Guided search samples the options, measures promising candidates and refines around the leaders. We focus the test budget where it can reveal the most value.

**Customer value:** Find opportunities that a single default setup can miss.

### Compare multiple engines

We evaluate serving engines including vLLM, SGLang and NVIDIA TensorRT-LLM where compatible. Each engine has its own strengths and tuning controls. We measure how those choices affect the actual model and workload.

**Customer value:** Choose software that makes better use of the hardware.

### Adapt to the server

We vary how models are split across GPUs, how many independent copies run, and how they use available memory. We account for GPU connections and proximity to processors, with CPU/GPU hybrid options where appropriate.

**Customer value:** Identify when a different arrangement can deliver more capacity or reduce GPU needs.

### Simulate different users and real arrival patterns

Personas represent quick lookups, conversations, writing, document questions and coding assistance. They vary request length, response length, follow-up questions and reading time. New sessions keep arriving even when the server slows down, exposing queues that a fixed user count can hide.

**Customer value:** Size for the customer's mix of demand and expected response times.

### Measure usable capacity as well as peak output

We increase demand to find where queues start growing and where response-time targets stop being met. We examine slower responses as well as averages, using statistical checks to assess stability and service performance.

**Customer value:** Explain how much demand the server can sustain while keeping users productive.

### Confirm results and explain the limits

We allow workloads to settle, remeasure leading configurations, check that the traffic generator keeps up, and capture failures and per-device activity. Power, memory, processor and GPU measurements help identify what limits performance.

**Customer value:** Give Dell evidence for the configuration recommendation and the next optimization.

**The customer outcome: a clear recommendation on which configuration to use, how much demand it can support and where tuning may improve value. Together with task-quality results, these measurements give Dell a stronger basis for sizing, value and performance claims.**
