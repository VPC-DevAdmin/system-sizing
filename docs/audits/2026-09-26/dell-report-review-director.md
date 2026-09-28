# Stronger benchmarks. More persuasive Dell stories.

## Director’s review of five recent reports

**The opportunity:** give customers a clear reason to choose a Dell configuration—and evidence they can trust.

These reports contain useful results, but several leave important questions unanswered: Was the server configured to perform at its best? Will users get answers quickly? Does the model produce good answers? Are the savings calculated correctly?

Better testing could uncover additional performance, identify a more economical configuration and support stronger sales claims. It also reduces the risk of publishing a result that a customer cannot reproduce.

**Our approach connects three things: the right configuration, a good user experience and a clear business benefit.**

The tables below put the most important issues first. Read the first two columns for the business issue and our response; the third adds one concrete example without requiring specialist knowledge. “Our approach” includes the testing methods we use and the reporting standards we need to meet. Detailed evidence and internal release checks are retained in the [technical review](docs/audits/2026-09-26/dell-report-review.md) and [internal audit](docs/audits/2026-09-26/internal-project-audit.md).

<!-- testing-overview:start -->
## How we find the right AI configuration

**A broader search. Realistic demand. A clearer customer decision.**

Our testing explores how hardware, AI models and serving software work together. We look for configurations that deliver more useful capacity while meeting the response times customers need.

### 1,728 setting combinations in one example

That is one engine and one model family, before hardware compatibility checks. Adding engines, models and workloads expands the possibilities further.

#### Search broadly, then focus the testing

We vary model formats, GPU arrangements, memory use and how requests are grouped. Guided search samples the options, measures promising candidates and refines around the leaders. We focus the test budget where it can reveal the most value.

**Customer value:** Find opportunities that a single default setup can miss.

#### Compare multiple engines

We evaluate serving engines including vLLM, SGLang and NVIDIA TensorRT-LLM where compatible. Each engine has its own strengths and tuning controls. We measure how those choices affect the actual model and workload.

**Customer value:** Choose software that makes better use of the hardware.

#### Adapt to the server

We vary how models are split across GPUs, how many independent copies run, and how they use available memory. We account for GPU connections and proximity to processors, with CPU/GPU hybrid options where appropriate.

**Customer value:** Identify when a different arrangement can deliver more capacity or reduce GPU needs.

#### Simulate different users and real arrival patterns

Personas represent quick lookups, conversations, writing, document questions and coding assistance. They vary request length, response length, follow-up questions and reading time. New sessions keep arriving even when the server slows down, exposing queues that a fixed user count can hide.

**Customer value:** Size for the customer's mix of demand and expected response times.

#### Measure usable capacity as well as peak output

We increase demand to find where queues start growing and where response-time targets stop being met. We examine slower responses as well as averages, using statistical checks to assess stability and service performance.

**Customer value:** Explain how much demand the server can sustain while keeping users productive.

#### Confirm results and explain the limits

We allow workloads to settle, remeasure leading configurations, check that the traffic generator keeps up, and capture failures and per-device activity. Power, memory, processor and GPU measurements help identify what limits performance.

**Customer value:** Give Dell evidence for the configuration recommendation and the next optimization.

**The customer outcome: a clear recommendation on which configuration to use, how much demand it can support and where tuning may improve value. Together with task-quality results, these measurements give Dell a stronger basis for sizing, value and performance claims.**

<!-- testing-overview:end -->

## 1. Beyond the Baseline: Llama 3.3 70B on PowerEdge

[Read the report](https://infohub.delltechnologies.com/en-us/t/beyond-the-baseline-llama-3-3-70b-inference-on-dell-poweredge-servers-with-nvidia-rtx-pro-gpus/)

**The missed opportunity:** show customers how much more useful work each server can deliver when it is properly configured.

| Issue and why it matters to Dell | Our approach: what we do differently | A concrete example, in plain language |
|---|---|---|
| **The comparison changes more than the hardware.** Software settings, model compression and the runs selected also differ. Customers cannot tell how much of the improvement comes from the server itself. | Test comparable configurations, then optimize each server separately. Give Dell two clear stories: the hardware advantage and the additional benefit of tuning. | The main results mix FP8 and NVFP4, two different levels of model compression. How each model was split across GPUs was not recorded as a controlled setting. |
| **The software choice is not clear enough.** Naming the NVIDIA serving package does not identify the engine and settings doing the work. A less suitable setup could hide the server’s potential. | Compare compatible engines and settings on the actual hardware. Select the combination that delivers the best result for the customer’s workload. | NVIDIA NIM is the serving package. Its selected engine and performance profile determine how the model runs; those choices are not established in the accessible sections. |
| **The older server’s setup needs clarification.** Its stated model format raises questions about how it actually ran. An unclear baseline makes the upgrade story easier to challenge. | Give the older system a fair, well-tuned baseline. A credible starting point makes Dell’s improvement claim more persuasive. | The older L40S is labeled as running NVFP4, a format designed for newer Blackwell GPUs. The report needs to explain the execution path actually used. |
| **The GPUs may not be arranged for the best result.** Splitting a model across more GPUs is not always faster than running several independent copies. The wrong arrangement can leave capacity unused. | Test different ways to distribute the work. Identify whether the customer benefits most from more simultaneous users, faster answers or fewer GPUs. | On two 96 GB GPUs, compare two independent model copies with one copy shared across both GPUs, where memory allows. More copies can serve separate requests simultaneously. |
| **High output can hide a poor user experience.** One comparison reports average waits of roughly 18 seconds to three minutes before an answer starts. That limits the value of a throughput headline. | Report how much demand the server can handle while keeping response times acceptable. Connect performance to the experience customers actually receive. | With 64 requests active, the long-input comparison reports average waits of 181, 20.2 and 18.1 seconds before the first generated text appears. |
| **The business recommendations go beyond the test.** Generating text of a certain length does not demonstrate accurate document analysis or useful code review. | Combine performance testing with task-quality results. Recommend a model and server because they do the customer’s job well. | The longest input tested is 4,096 tokens—pieces of text processed by the model. This measures text-generation performance, not retrieval accuracy or whether suggested code works. |
| **The buying decision is incomplete.** The servers use different numbers and types of GPUs. More output alone does not establish better value. | Show useful capacity alongside purchase cost, electricity use and space requirements. Help customers choose the right configuration for their budget. | The systems contain two L40S GPUs, two RTX PRO 6000 GPUs, or eight RTX PRO 4500 GPUs. The performance comparison therefore changes both GPU type and quantity. |
| **Supporting details and labels need more care.** Unclear units, inconsistent descriptions and limited accessible repeat information make the results harder to trust. | Keep the result, configuration, chart and explanation connected. Make the evidence easy to check and the message easy to read. | Table 18 omits time units, and Figure 2’s accessible description misnames hardware. Four linked methodology chapters were inaccessible during review, so their contents remain unassessed. |

**The stronger Dell story:** “Here is the configuration that serves your workload well—and the extra capacity that careful tuning can unlock.”

## 2. Signal65: Trusted Answers at Enterprise Scale

[Read the report](https://signal65.com/wp-content/uploads/2026/09/Signal65-Insights_Trusted-Answers-at-Enterprise-Scale.pdf)

**The missed opportunity:** turn an AI accuracy ranking into a practical buying guide.

This report is more careful about its scope than several of the others. It asks which models answer questions accurately from supplied documents. The next step is showing what it takes to deliver those answers quickly and economically on Dell hardware.

| Issue and why it matters to Dell | Our approach: what we do differently | A concrete example, in plain language |
|---|---|---|
| **The most accurate model may not be the best purchase.** A small improvement in accuracy could require much more time, memory or computing power. The ranking does not show that tradeoff. | Start with the quality the task requires. Then compare speed, capacity and cost among models that meet it. Help customers pay for the capability they need. | The study measures answer accuracy at document lengths up to 200,000 tokens. It does not measure the response time or cost of delivering those answers. |
| **The report does not clearly connect every result to a deployment.** Readers cannot easily tell which server, software and settings produced each score. That limits its usefulness for sales recommendations. | Connect each recommended model to an exact, tested configuration. Give sales teams a result they can turn into a specific customer proposal. | The detailed diagram lists vLLM 0.10.2, while the study names several GPU platforms. It does not provide a complete model-by-model map to the deployed software and hardware. |
| **Small score differences may be less meaningful than they look.** The report does not provide enough detail about repeated tests and variation to judge every close result. | Show which differences are consistent. Avoid making a purchasing recommendation depend on a narrow lead that may not repeat. | The report says results were averaged across repeats, but does not give enough repeat counts and score variation to judge whether small ranking differences are dependable. |
| **The longer-document comparisons are harder to interpret than they appear.** The set of tested models changes, and equal text limits can represent different amounts of information for different models. | Compare the same models on the same documents, with clear handling of length limits. Make the effect of larger documents easy to understand. | The comparison changes from 91 configurations at 32K tokens to 72 at 128K and 54 at 200K. A clean comparison also needs the same configurations at every length. |
| **Answering one huge document does not show how many customers a server can support.** Large documents consume resources that could otherwise serve additional users. | Test simultaneous demand, first-time document processing and repeated questions. Show both answer quality and the number of users the configuration can support well. | The server stores working memory for each active document. Processing a new document and answering another question about an already-cached document can have very different costs. |
| **Controlled documents do not cover every enterprise challenge.** Real documents can be messy, contradictory or incomplete. A model can also avoid mistakes simply by declining to answer. | Measure useful correct answers, wrong answers and unnecessary refusals separately. Include representative business material alongside controlled tests. | The test uses generated records from three business domains. Its missing-information questions test refusal behavior; they do not establish broad accuracy on messy real-world documents. |
| **The charts can make differences look larger than they are.** Changing scales and crowded labels make comparison harder. | Use consistent charts and a readable summary. Let the strength of the result carry the message. | The three context-length charts start their scales at 90%, 82% and 0%. The same visual distance therefore represents different-sized accuracy gaps. |

**The stronger Dell story:** “These models meet your accuracy needs. This Dell configuration delivers those answers at the best measured service level and cost.”

## 3. Prowess: Xeon processor comparisons

[Read the report](https://infohub.delltechnologies.com/en-us/p/benefits-and-comparison-of-intel-r-xeon-r-processors-with-e-cores-p-cores-and-ap-cores-for-select-workloads/)

**The immediate priority:** correct the technical foundation before using the conclusions in customer guidance.

This paper has problems with how it describes the processors, documents the tests and presents efficiency. It also misses opportunities to show what properly tuned software can deliver.

| Issue and why it matters to Dell | Our approach: what we do differently | A concrete example, in plain language |
|---|---|---|
| **The processor explanation is incorrect.** The report presents “AP” as a third type of core, although the tested processor uses P-cores. That weakens the reasoning behind its recommendations. | Explain the actual differences between the systems and tie recommendations to measured results. Keep the customer story simple and technically sound. | The processor called “AP” is the Intel Xeon 6978P, a P-core processor. It is a different platform class, not a third core architecture. |
| **The published instructions do not reproduce the claimed scientific test.** One command does not assign processor cores as intended, and the instructions do not fully show how the simultaneous jobs were launched. | Require the exact executed setup and verify that work reaches the intended resources. A repeatable result is easier for Dell to defend. | With a value of 8, the printed CPU-binding command selects core number 8 rather than eight cores. Actual execution logs are needed to establish the placement used. |
| **The efficiency table contradicts itself.** Labels disagree about which system is better, and one stated maximum is below its minimum. The document also conflicts on whether results were measured or simulated. | Build claims and charts from one checked set of results. Make both the advantage and the source of the evidence unambiguous. | Table 2 gives a 52% maximum and a 56% minimum for one result. The legal notice also says “simulated,” although the narrative describes measured tests. |
| **The software engine is unclear.** The report names one inference engine while the instructions describe another. That makes it difficult to know what produced the result. | Identify the engine actually tested and compare suitable alternatives. Demonstrate what the platform can deliver with a well-chosen software stack. | The report describes OpenVINO. The methodology installs Intel TensorFlow and uses its reference scripts. Readers need to know which software produced the published numbers. |
| **Installing optimized software does not prove it was used effectively.** Processor settings, memory placement and numerical formats can materially affect speed. The report does not establish the best combination. | Tune the settings that determine real performance and verify their effect. For this study, that means testing the processor’s acceleration features and checking that answer accuracy is preserved. | The image-classification test uses two numerical formats, FP32 and INT8. It does not compare another option, BF16, or verify use of the processor’s specialized AI acceleration. |
| **The web test can hide spare capacity.** Two servers can look equal when both are asked to do less work than they can handle. The test client or web-server settings can also become the limit. | Increase demand until a clear response-time or error limit is reached. Verify that the testing tools keep up. Show how much useful work each server can actually sustain. | The wrk2 tool sends a preset request rate. Two servers both serving 10,000 requests per second may simply be meeting the same imposed limit. |
| **More processor cores are presented as a broader performance advantage.** The scientific workload’s output rises almost in proportion to the extra cores assigned. That supports a capacity story more directly than a faster-core story. | Separate “finishes one job faster” from “runs more jobs at once.” Both can be valuable, but they answer different customer needs. | Thirty eight-core jobs versus sixteen allocate 87.5% more cores. Reported output rises roughly 84–87%—almost the same increase as the resources assigned. |
| **Memory descriptions and instructions need correction.** The stated memory advantage conflicts with the configuration table, and formatting damages executable commands. | Check the final report against the tested configuration and provide usable instructions. Remove avoidable reasons for a customer to question the work. | The system described as having more memory actually lists 1,536 GB versus 2,048 GB. Some published commands also contain curly quotation marks that can break execution. |

**The stronger Dell story:** “Choose this configuration for lower energy at your required workload, this one for faster individual jobs, and this one for more jobs at once.”

## 4. PowerEdge R770 with RTX PRO 4500: enterprise AI inference

[Read the report](https://infohub.delltechnologies.com/en-us/t/dell-poweredge-r770-with-nvidia-rtx-pro-4500-accelerating-enterprise-ai-inference/)

**The immediate priority:** reconcile the configuration and electricity calculations. Then show the real value of an affordable AI server.

| Issue and why it matters to Dell | Our approach: what we do differently | A concrete example, in plain language |
|---|---|---|
| **The larger models do not fit in memory as described.** The report says each GPU runs a complete model, but the stated models need more memory than each GPU provides. Something important is missing or incorrect. | Verify that the model fits and document how it is distributed or compressed. Give customers a configuration they can actually deploy. | Each GPU has 32 GB. In the stated format, the 27-billion-parameter model needs roughly 54 GB and the 35-billion model roughly 70 GB for weights alone. |
| **The electricity-efficiency calculation mixes one GPU with the whole server’s output.** As written, one example overstates GPU efficiency by about four times. Other summary values also fail to reconcile. | Measure work and electricity over the same equipment and time period. Base savings claims on a calculation customers can follow and reproduce. | The worked example divides all four GPUs’ output by one GPU’s 151 W. The denominator should be about 604 W for those GPUs, before other server power. |
| **The test may miss the best use of these GPUs.** A more compact model format can free memory and allow more users or more independent model copies. That opportunity is not established by the reported setup. | Compare model formats and GPU arrangements while checking answer quality. Look for the combination that delivers the most useful capacity within the customer’s budget. | A smaller model format can free space for requests or additional copies. But even an approximately 35 GB model still cannot fit entirely on a 32 GB GPU. |
| **One software setup per model does not establish the best performance.** Different engines and settings can produce different results on the same hardware. | Test compatible engines and tune how they schedule work and use memory. Select the winner from measurements rather than a default choice. | Qwen3-8B uses vLLM 0.17.1; the larger models use 0.19.0. No comparison establishes which compatible engine and settings deliver the best result on these GPUs. |
| **The “optimal” result can still make users wait.** One selected operating point takes about 20 seconds to start an answer. Maximum output is not necessarily the most useful setting. | Optimize for a stated customer experience. Show the workload the server can sustain while responding within the required time. | The larger 35B model’s selected point reports 20,091 milliseconds—about 20 seconds—to start responding. The report needs a separate result for an acceptable customer wait. |
| **Larger model names are used to imply better answers.** Parameter count and output speed do not demonstrate business accuracy or reasoning quality. | Compare actual task results. Recommend the smallest or most economical model that meets the customer’s quality needs. | The 35B model activates only part of its network for each step. That reduces some computation, but does not prove 35B-level accuracy or eliminate its full memory requirement. |
| **The models are not always compared on the same work.** Input lengths differ, and long generated answers feature prominently. Rankings may change for the shorter answers many applications need. | Use common comparison tasks and a realistic application mix. Make clear which model is best for which kind of work. | The longer-input comparison gives the smaller model 1,024 input tokens and the larger models 2,048. Featured peak results also generate lengthy 2,048-token answers. |
| **The cost story and presentation need stronger support.** Cloud comparisons lack a complete cost calculation, while some software labels conflict. | Build the economics from verified capacity and energy results, using equivalent service requirements. Align every diagram and claim with the tested setup. | The methodology table names vLLM, while the diagram description names TensorRT. The cloud-cost discussion does not supply a complete, reproducible calculation. |

**The stronger Dell story:** “This affordable server delivers the answer quality your application needs, for this many users, at this measured operating cost.”

## 5. Principled Technologies: XE7740 agentic AI

[Read the report](https://www.principledtechnologies.com/Dell/PowerEdge-XE7740-agentic-AI-performance-0726/)

**The missed opportunity:** replace an ambiguous count of “agents” with a clear measure of business productivity.

The report provides useful workload and system details. Its recommendations would be stronger if it connected completed business tasks, customer demand and hardware configuration more directly.

| Issue and why it matters to Dell | Our approach: what we do differently | A concrete example, in plain language |
|---|---|---|
| **The cost chart reverses the utilization labels.** It makes the lower-use scenario appear cheaper per unit of work. The cost and energy calculations also need clearer links to the performance results. | Use one checked calculation for the report and charts. Explain savings through successful business work, with clear assumptions about how much the server is used. | Figure 5 assigns about $0.05 per million tokens to 50% utilization and $0.09 to 100%. The supporting cost table gives the opposite, economically consistent relationship. |
| **An “agent” can mean very different amounts of work.** The suite combines simple AI calls, fixed business processes and agents that choose their own tools. A single count does not tell customers what their applications can support. | Define the work in business terms: investigations completed, documents reviewed or inspections processed. Explain what each task includes. | Filing questions use a single AI step; fraud investigation includes repeated tool choices. Counting each as one “agent” combines very different demands on the server. |
| **The test waits for work to finish before replacing it.** Real demand can keep arriving while a server slows down. The reported count does not fully show what happens when a queue builds. | Test independently arriving demand as well as maximum output. Show how much work can arrive while response times remain acceptable. | The harness admits a new task when another finishes and runs task types in ordered waves. It does not test a continuously arriving, mixed stream of business demand. |
| **The promised service level needs a clearer definition.** Customers cannot readily reconstruct how the reported supported-user counts meet the timing targets across different workflows. | Report completion times and successful tasks at each demand level. Make the service promise specific enough to use in sizing a deployment. | The targets are 30 seconds for finance and 90 seconds for manufacturing. It is not clear enough whether the supported counts protect typical users or slower responses too. |
| **GPU arrangement may be holding back some configurations.** The six-GPU setup includes communication across processor groups. Separate allocations for text and vision may also leave resources waiting. | Test different ways to place and share the work. Find the configuration that keeps useful tasks moving with the fewest unnecessary delays. | The six-GPU setup has three GPUs near each processor. Pairing all six into two-GPU groups makes one pair communicate across processors, adding a possible delay. |
| **The model and engine choices deserve a broader comparison.** A large language model may be unnecessary for tasks that mostly explain calculations already completed by other software. | Compare suitable models and engines at the same task-quality requirement. A smaller model could support more work or reduce hardware needs; test that opportunity rather than assume it. | Text runs on Llama 3.1 70B FP8 through vLLM 0.20.0; vision uses Qwen3.5-2B. Several workflows do calculations first and ask the large model to explain them. |
| **Fast completion is not enough if the task is wrong.** Timing alone does not prove that an investigation, financial explanation or inspection reached the right conclusion. | Connect capacity to task-quality results. Make correct completed work the primary productivity measure. | A completed financial narrative can still contain incorrect conclusions. A returned inspection result can still miss a defect. Those outcomes need scoring alongside timing. |
| **Additional GPUs do not help every workload equally.** Doubling GPU count raises financial-workflow output by about 30%, versus about 123% for manufacturing. Other parts of the application may be limiting the financial workload. | Find the limiting step before recommending more GPUs. The better investment may be processor capacity, memory, database tuning or a different configuration. | From four to eight GPUs, financial output rises from 905 to 1,179 text tokens per second; manufacturing rises from 144 to 321. Scaling depends strongly on the work. |
| **Test preparation and presentation need clearer boundaries.** It is unclear how synthetic-data preparation is timed, and dates and hardware terminology conflict. | Separate test setup from genuine customer work, document the actual environment and proofread the final claims. Keep the productivity result focused on work the customer needs done. | Testing is dated complete on June 11, but operating-system updates are dated June 20. That leaves the software environment used for the test unclear. |

**The stronger Dell story:** “This configuration completes this many correct business tasks within your required response time, at this cost per task.”

## The message we should take to Dell

Our testing is designed to turn server performance into a clear customer decision. We compare practical configurations, tune the software and measure capacity against the response times the application needs. We connect those results to answer quality and cost so Dell can explain which configuration to buy, what it can support and why it offers value.

The opportunity is larger than correcting these reports. Better testing can reveal additional capacity, show when fewer GPUs are enough and give sales teams more persuasive evidence for the right configuration.

**The commercial value: stronger claims, clearer recommendations and greater customer confidence.**

*Potential improvements must be measured before a percentage is claimed. The technical review retains the evidence, source limitations and detailed corrective actions behind this simplified version.*
