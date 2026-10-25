"""Strict, versioned input contracts; pilot measurements never imply publishable claims."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    project_id: Literal["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
    mode: Literal["pilot", "confirmation"] = "pilot"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    paired_blocks: int = Field(default=2, ge=1, le=20)
    duration_s: int = Field(default=60, ge=10, le=10800)
    warmup_s: int = Field(default=15, ge=0, le=600)
    concurrency: int = Field(default=4, ge=1, le=16384)
    request_timeout_s: int = Field(default=120, ge=1, le=600)
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)
    corpus_size: int = Field(default=256, ge=16, le=4096)
    document_records: int = Field(default=16, ge=4, le=4096)
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=8)
    context_limit: int = Field(default=8192, ge=2048, le=131072)
    min_requests: int = Field(default=30, ge=1, le=1000000)
    min_quality: float = Field(default=0.95, gt=0, le=1)
    max_error_fraction: float = Field(default=0.01, ge=0, le=0.2)
    min_slo_fraction: float = Field(default=0.95, gt=0, le=1)
    response_slo_s: float = Field(default=60, gt=0, le=600)
    structured_output: bool = True
    speculative_concurrencies: list[int] = Field(default_factory=list, max_length=8)
    speculative_method: Literal["ngram", "mtp"] = "ngram"
    speculative_tokens: int = Field(default=5, ge=1, le=16)
    cache_samples: int = Field(default=32, ge=16, le=256)
    cache_record_counts: list[int] = Field(
        default_factory=lambda: [16, 64, 256], min_length=1, max_length=8
    )
    cache_concurrencies: list[int] = Field(
        default_factory=lambda: [1, 4, 16], min_length=1, max_length=8
    )
    prefill_workers: int = Field(default=1, ge=1, le=7)
    p01_block_size: Literal[16, 32, 64, 128] | None = None
    p01_transport: Literal["local", "auto"] = "local"
    p01_cuda_ipc_get: Literal["default", "enabled"] = "default"
    p01_memtype_cache: Literal["default", "disabled"] = "default"
    p01_cross_layer_blocks: bool = False
    p01_max_batched_tokens: int | None = Field(default=None, ge=1024, le=131072)
    p01_prefill_grid: list[int] = Field(default_factory=list, max_length=7)
    p01_request_rates: list[float] = Field(default_factory=list, max_length=8)
    p01_placement: Literal["pooled", "socket_local", "cross_socket"] = "pooled"
    p01_placement_grid: list[Literal["pooled", "socket_local", "cross_socket"]] = Field(
        default_factory=lambda: ["pooled"], min_length=1, max_length=2
    )
    p01_execution: Literal["default", "sync", "eager"] = "default"
    p01_cuda_launch_blocking: bool = False
    p01_client_max_concurrency: int = Field(default=512, ge=512, le=8192)
    p01_engine_max_seqs: int | None = Field(default=None, ge=32, le=4096)
    p01_runtime_image: str | None = Field(
        default=None, pattern=r"^(sha256:[a-f0-9]{64}|[^\s]+@sha256:[a-f0-9]{64})$"
    )
    p01_frontend_format: Literal["auto", "mistral"] = "auto"
    p01_chat_processor: Literal["dynamo", "vllm"] = "dynamo"
    p01_attention_backend: Literal["auto", "TRITON_ATTN"] = "auto"
    p01_kv_lease_s: int = Field(default=30, ge=30, le=600)
    p01_token_search: bool = False
    p01_token_shapes: list[tuple[int, int]] = Field(
        default_factory=lambda: [(128, 256), (4096, 256), (16384, 256), (4096, 1024)],
        min_length=1,
        max_length=16,
    )
    p01_streams: list[int] = Field(
        default_factory=lambda: [128, 512, 2048, 8192], min_length=2, max_length=12
    )
    p01_goal_seek: bool = False
    p01_rate_resolution: float = Field(default=1, ge=0.1, le=10)
    p01_document_grid: list[int] = Field(
        default_factory=lambda: [64, 128, 256], min_length=1, max_length=6
    )
    p01_answer_grid: list[int] = Field(default_factory=lambda: [4, 8], min_length=1, max_length=4)
    p01_queue_growth_s: float = Field(default=2, ge=0, le=60)
    p01_ttft_slo_s: float = Field(default=5, gt=0, le=600)
    tenant_workflow: Literal["independent", "post_call"] = "independent"
    call_request_rates: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5], min_length=1, max_length=8
    )
    call_max_wer: float = Field(default=0.15, ge=0, le=1)
    session_mode: Literal["resume", "capacity"] = "resume"
    session_workload: Literal["document", "knowledge_worker"] = "document"
    session_prime_documents: bool = False
    session_document_padding: bool = False
    session_document_pad_repeats: int = Field(default=0, ge=0, le=100000)
    session_min_returns: int = Field(default=4, ge=4, le=64)
    session_gpu_capacity_documents: int | None = Field(default=None, ge=1, le=4096)
    session_capacity_evidence: str | None = None
    session_capacity_multiples: list[int] = Field(default_factory=list, max_length=2)
    session_kv_cache_dtype: Literal["source", "auto", "bfloat16", "fp8"] = "source"
    session_active_records: int | None = Field(default=None, ge=2, le=4096)
    session_context_tokens: int = Field(default=96000, ge=1024, le=120000)
    session_new_input_tokens: int = Field(default=512, ge=32, le=8192)
    # Stage 2 (docs/p02-stage2-compaction.md): growing sessions, probes, compaction.
    session_growth_tokens: int = Field(default=0, ge=0, le=16384)
    session_probe_mix: list[
        Literal["reference", "session_recall", "knowledge_update", "abstention"]
    ] = Field(default_factory=lambda: ["reference"], min_length=1, max_length=8)
    session_limit_policy: Literal["fail", "new_session"] = "fail"
    # Context budget for the new-session policy; None means the engine limit.
    session_limit_tokens: int | None = Field(default=None, ge=4096, le=131072)
    # Start each worker at a uniformly random point of its steady-state cycle by priming a
    # synthetic prior history (fabricated correct answers), so a window shows steady state.
    session_prime_phase: bool = False
    session_compaction_arms: list[Literal["none", "xeon"]] = Field(
        default_factory=lambda: ["none"], min_length=1, max_length=2
    )
    session_compaction_trigger_tokens: int = Field(default=96000, ge=4096, le=131072)
    session_compaction_target_tokens: int = Field(default=32000, ge=1024, le=131072)
    session_compaction_budget_tokens: int = Field(default=4096, ge=256, le=16384)
    session_compaction_url: str | None = None
    session_compaction_model: str | None = None
    session_compaction_timeout_s: int = Field(default=900, ge=10, le=3600)
    session_compaction_ledger: bool = True
    session_compaction_chunk_turns: int = Field(default=3, ge=1, le=64)
    # blocking: the next turn waits for the summary (the pause is the measured cost).
    # async: the session keeps growing while the Xeon works (future optimization).
    session_compaction_mode: Literal["blocking", "async"] = "blocking"
    session_cache_arms: list[Literal["baseline", "treatment"]] = Field(
        default_factory=lambda: ["baseline", "treatment"], min_length=1, max_length=2
    )
    session_think_s: float = Field(default=60, gt=0, le=3600)
    session_population_grid: list[int] = Field(
        default_factory=lambda: [32, 64, 128, 256], min_length=1, max_length=6
    )
    session_rate_grid: list[float] = Field(
        default_factory=lambda: [1 / 60, 1 / 30], min_length=1, max_length=10
    )
    session_engine_max_seqs: int = Field(default=256, ge=32, le=2048)
    session_arrival_rates: list[float] = Field(default_factory=list, max_length=12)
    session_max_arrival_rate: float = Field(default=128, gt=0, le=1000)
    session_refinement_rounds: int = Field(default=2, ge=0, le=5)
    session_slo_grid_s: list[float] = Field(
        default_factory=lambda: [5, 10, 30, 60, 120], min_length=1, max_length=12
    )
    session_turns: int = Field(default=8, ge=2, le=64)
    session_answer_records: int = Field(default=2, ge=1, le=8)
    session_max_pending: int = Field(default=2048, ge=16, le=16384)
    session_queue_growth_fraction: float = Field(default=0.05, ge=0, le=0.2)
    session_client_lag_s: float = Field(default=0.5, gt=0, le=5)
    session_tier: Literal["cpu", "cpu_disk"] = "cpu"
    session_cpu_backend: Literal["kvbm", "simple_cpu"] = "kvbm"
    session_ttft_slo_s: float = Field(default=5, gt=0, le=600)
    session_count: int = Field(default=8, ge=4, le=256)
    session_churn: int = Field(default=16, ge=4, le=512)
    session_gpu_gb: int | None = Field(default=4, ge=1, le=64)
    session_cpu_gb: int = Field(default=32, ge=4, le=256)
    session_disk_gb: int = Field(default=64, ge=8, le=256)
    attention_baseline: Literal["FLASHINFER", "TRITON_ATTN"] = "FLASHINFER"
    attention_treatment: Literal["FLASHINFER", "TRITON_ATTN"] = "TRITON_ATTN"
    attention_kv_dtype: Literal["fp8", "bfloat16"] = "fp8"
    attention_logprob_tolerance: float = Field(default=0.1, gt=0, le=0.5)
    tenant_requests_per_second: float = Field(default=0.5, gt=0, le=10)
    tenant_noise_multiplier: float = Field(default=4, ge=2, le=10)
    tenant_kv_gib: int = Field(default=8, ge=1, le=16)
    diffusion_samples: int = Field(default=8, ge=4, le=64)
    diffusion_teacher_steps: list[int] = Field(
        default_factory=lambda: [20, 50], min_length=1, max_length=4
    )
    max_hours: float = Field(default=12, gt=0, le=168)

    @model_validator(mode="after")
    def validate_contract(self):
        if self.concurrency > 512 and not (
            self.project_id == "P02" and self.session_mode == "capacity"
        ):
            raise ValueError("Concurrency above 512 is only supported by P02 capacity")
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(
            g < 0 or g > 255 for g in self.gpu_ids
        ):
            raise ValueError("GPU IDs must be distinct nonnegative integers below 256")
        if self.project_id == "P01" and len(self.gpu_ids) <= self.prefill_workers:
            raise ValueError("P01 requires at least one prefill and one decode GPU")
        if self.p01_frontend_format == "mistral" and self.p01_chat_processor != "vllm":
            raise ValueError("Mistral frontend format requires the vLLM chat processor")
        if self.p01_token_search:
            if (
                self.project_id != "P01"
                or self.p01_goal_seek
                or self.p01_request_rates
                or not self.p01_prefill_grid
            ):
                raise ValueError(
                    "Token search requires P01 splits and excludes document/arrival-rate search"
                )
            if self.p01_prefill_grid != sorted(set(self.p01_prefill_grid)) or any(
                p < 1 or p >= len(self.gpu_ids) for p in self.p01_prefill_grid
            ):
                raise ValueError("Invalid token search GPU splits")
            if self.p01_streams != sorted(set(self.p01_streams)) or any(
                c < 1 or c > 8192 for c in self.p01_streams
            ):
                raise ValueError("Token search streams must increase within 1..8192")
            if len(set(self.p01_token_shapes)) != len(self.p01_token_shapes) or any(
                i < 128 or o < 128 or i + o + 256 > self.context_limit
                for i, o in self.p01_token_shapes
            ):
                raise ValueError(
                    "Token shapes must be distinct, at least 128 tokens, and fit context with template reserve"
                )
            if any(x != "pooled" for x in self.p01_placement_grid) and any(
                p < 2 or p > len(self.gpu_ids) - 2 for p in self.p01_prefill_grid
            ):
                raise ValueError("Local/cross pools require both roles on each socket")
        if not self.p01_token_search and (self.p01_prefill_grid or self.p01_request_rates):
            if (
                self.project_id != "P01"
                or self.mode != "pilot"
                or not self.p01_prefill_grid
                or not self.p01_request_rates
            ):
                raise ValueError("P01 search requires pilot mode, a split grid and arrival rates")
            if self.p01_prefill_grid != sorted(set(self.p01_prefill_grid)) or any(
                p < 1 or p >= len(self.gpu_ids) for p in self.p01_prefill_grid
            ):
                raise ValueError("Every split must leave at least one GPU for each role")
            if self.p01_request_rates != sorted(set(self.p01_request_rates)) or any(
                not 0 < r <= 50 for r in self.p01_request_rates
            ):
                raise ValueError("Arrival rates must be increasing, positive and at most 50/s")
            if self.duration_s * min(self.p01_request_rates) < self.min_requests:
                raise ValueError("Every search rate must meet the minimum sample")
        if self.p01_goal_seek:
            if (
                self.project_id != "P01"
                or not self.p01_prefill_grid
                or len(self.p01_request_rates) < 3
            ):
                raise ValueError(
                    "Goal search requires P01, GPU splits and at least three arrival rates"
                )
            for values, low, high in (
                (self.p01_document_grid, 4, 1024),
                (self.p01_answer_grid, 1, 32),
            ):
                if values != sorted(set(values)) or any(v < low or v > high for v in values):
                    raise ValueError("Shape grids must be increasing and within supported bounds")
            if max(self.p01_answer_grid) > min(self.p01_document_grid):
                raise ValueError("Requested answers must fit every document")
            if self.corpus_size * sum(self.p01_document_grid) * len(self.p01_answer_grid) > 500000:
                raise ValueError("Goal search exceeds the frozen corpus record budget")
            if 128 * max(self.p01_document_grid) + self.max_output_tokens > self.context_limit:
                raise ValueError(
                    "Reserve 128 tokens per document record plus the answer cap in context"
                )
            if 128 * max(self.p01_answer_grid) > self.max_output_tokens:
                raise ValueError("Reserve 128 output tokens per requested record")
        if len(set(self.p01_placement_grid)) != len(self.p01_placement_grid) or (
            "pooled" in self.p01_placement_grid and len(self.p01_placement_grid) > 1
        ):
            raise ValueError("Use pooled routing alone, or socket-local and cross-socket pools")
        if self.p01_goal_seek and "cross_socket" in self.p01_placement_grid:
            if (
                len(self.gpu_ids) < 4
                or len(self.gpu_ids) % 2
                or any(p < 2 or p > len(self.gpu_ids) - 2 for p in self.p01_prefill_grid)
            ):
                raise ValueError(
                    "Cross-socket pools need equal GPU counts and a prefill and decode GPU per pool"
                )
        if self.call_request_rates != sorted(set(self.call_request_rates)) or any(
            not 0 < r <= 10 for r in self.call_request_rates
        ):
            raise ValueError("Call rates must be increasing and within (0, 10]")
        if self.tenant_workflow == "post_call":
            if (
                self.project_id != "P06"
                or self.duration_s * min(self.call_request_rates) < self.min_requests
            ):
                raise ValueError(
                    "Post-call workflow requires P06 and enough arrivals in every window"
                )
        if self.mode == "confirmation" and (
            self.paired_blocks < 5
            or self.duration_s < 600
            or (
                self.warmup_s < 120
                and not (self.project_id == "P02" and self.session_mode == "capacity")
            )
        ):
            raise ValueError(
                "Confirmation requires five pairs, 600 s measurement, and warmup (P02 capacity uses explicit priming)"
            )
        for values, low, high in (
            (self.cache_record_counts, 4, 1024),
            (self.cache_concurrencies, 1, 512),
        ):
            if values != sorted(set(values)) or any(v < low or v > high for v in values):
                raise ValueError(
                    "Cache grid lists must be strictly increasing and within supported limits"
                )
        if self.speculative_concurrencies:
            if (
                self.project_id != "P04"
                or self.speculative_concurrencies != sorted(set(self.speculative_concurrencies))
                or any(c < 1 or c > 512 for c in self.speculative_concurrencies)
            ):
                raise ValueError(
                    "Speculative concurrency grid requires P04 and strictly increasing values from 1 to 512"
                )
        generated_records = self.corpus_size * (
            sum(self.cache_record_counts) if self.project_id == "P03" else self.document_records
        )
        if generated_records > 500_000:
            raise ValueError(
                "Frozen corpus exceeds the record budget; reduce corpus_size or document sizes"
            )
        if self.project_id == "P10" and (
            len(self.gpu_ids) != 1 or self.attention_baseline == self.attention_treatment
        ):
            raise ValueError("P10 requires one GPU and two different explicit attention backends")
        if self.project_id == "P06":
            if len(self.gpu_ids) != 2 or self.mode != "pilot":
                raise ValueError(
                    "P06 requires two baseline GPUs and pilot mode; the last GPU is partitioned and restored"
                )
            if (
                self.duration_s * self.tenant_requests_per_second < self.min_requests
                or self.duration_s * self.tenant_requests_per_second * self.tenant_noise_multiplier
                > 100000
            ):
                raise ValueError(
                    "Tenant arrivals must meet the minimum sample and remain within 100000 requests per window"
                )
        if self.project_id == "P05":
            if len(self.gpu_ids) != 1 or self.mode != "pilot":
                raise ValueError("P05 currently requires one GPU and pilot mode")
            if self.diffusion_teacher_steps != sorted(set(self.diffusion_teacher_steps)) or any(
                s < 10 or s > 100 for s in self.diffusion_teacher_steps
            ):
                raise ValueError("Teacher steps must be increasing, from 10 to 100")
        if self.session_cpu_backend == "simple_cpu" and (
            self.project_id != "P02"
            or self.session_mode != "capacity"
            or self.session_tier != "cpu"
        ):
            raise ValueError("Native CPU offload currently requires P02 CPU capacity mode")
        if self.session_active_records is not None and (
            self.session_workload != "knowledge_worker"
            or not self.session_answer_records
            <= self.session_active_records
            <= self.document_records
        ):
            raise ValueError(
                "Active references require knowledge workers and a valid record subset"
            )
        if self.session_workload == "knowledge_worker":
            if self.project_id != "P02" or self.session_mode != "capacity":
                raise ValueError("Knowledge workers require P02 capacity mode")
            if "xeon" in self.session_compaction_arms:
                if not self.session_compaction_url or not self.session_compaction_model:
                    raise ValueError("Xeon compaction requires a summarizer endpoint and model")
                if not (
                    self.session_compaction_target_tokens
                    < self.session_compaction_trigger_tokens
                    <= self.context_limit - self.max_output_tokens - self.session_growth_tokens
                ):
                    raise ValueError(
                        "Compaction target must be below the trigger, and the trigger must leave output and growth headroom"
                    )
                if self.session_growth_tokens == 0:
                    raise ValueError("Compaction arms require growing sessions")
            if self.session_prime_phase and self.session_growth_tokens == 0:
                raise ValueError("Phase priming requires growing sessions")
            if self.session_limit_policy == "new_session" and self.session_growth_tokens == 0:
                raise ValueError("A new-session limit policy only applies to growing sessions")
            if self.session_limit_tokens is not None and not (
                self.session_context_tokens
                + self.session_growth_tokens
                + self.session_new_input_tokens
                < self.session_limit_tokens
                <= self.context_limit - self.max_output_tokens
            ):
                raise ValueError(
                    "Session context budget must exceed the initial context plus one turn and fit the engine limit"
                )
            if (
                any(p != "reference" for p in self.session_probe_mix)
                and self.session_growth_tokens == 0
            ):
                raise ValueError("Session-history probes require growing sessions")
            if (
                self.session_context_tokens + self.session_new_input_tokens + self.max_output_tokens
                >= self.context_limit
            ):
                raise ValueError("Knowledge-worker context lacks output headroom")
            if self.concurrency < max(self.session_population_grid):
                raise ValueError("Knowledge-worker population exceeds client streams")
        if self.project_id == "P02" and self.session_mode == "resume":
            if self.session_gpu_gb is None:
                raise ValueError("Resume pilot requires explicit GPU cache budget")
            if len(self.gpu_ids) != 1 or self.mode != "pilot":
                raise ValueError("P02 currently requires one GPU and pilot mode")
            if self.session_count + self.session_churn > self.corpus_size:
                raise ValueError("Session and churn counts must fit the frozen corpus")
            if self.session_cpu_gb <= self.session_gpu_gb:
                raise ValueError("CPU session cache must exceed GPU cache")
            if self.session_tier == "cpu_disk" and self.session_disk_gb < self.session_cpu_gb:
                raise ValueError("Disk session cache must be at least the CPU cache size")
            if self.min_requests > self.session_count:
                raise ValueError("Minimum requests must not exceed the finite resume cohort")
        if self.project_id == "P02" and self.session_mode == "capacity":
            if self.session_document_padding and (
                self.session_workload != "document"
                or self.session_context_tokens + self.max_output_tokens >= self.context_limit
            ):
                raise ValueError("Padded documents need the document workload and output headroom")
            if self.warmup_s != 0:
                raise ValueError("Capacity mode includes cold work; set warmup_s to zero")
            if self.session_tier != "cpu":
                raise ValueError("Document capacity currently supports CPU offload only")
            if (
                self.mode == "confirmation"
                and self.session_workload == "document"
                and not self.session_prime_documents
            ):
                raise ValueError("Document confirmation requires separate priming")
            if self.session_gpu_gb is not None:
                raise ValueError("Capacity mode requires automatic production GPU cache sizing")
            if self.session_answer_records > self.document_records:
                raise ValueError("Answer records exceed document size")
            if self.session_max_pending < self.concurrency:
                raise ValueError("Pending request limit must cover client concurrency")
            for grid, upper in [(self.session_population_grid, 4096), (self.session_rate_grid, 10)]:
                if grid != sorted(set(grid)) or any(x <= 0 or x > upper for x in grid):
                    raise ValueError("Session grids must be increasing, unique and bounded")
            if len(self.session_population_grid) < 2 and not self.session_arrival_rates:
                raise ValueError("Capacity needs at least two session populations")
            if self.session_arrival_rates:
                if (
                    self.session_arrival_rates != sorted(set(self.session_arrival_rates))
                    or min(self.session_arrival_rates) <= 0
                ):
                    raise ValueError("Arrival rates must be positive, unique and increasing")
                if (
                    max(self.session_arrival_rates) > self.session_max_arrival_rate
                    or self.duration_s * self.session_max_arrival_rate > 100000
                ):
                    raise ValueError("Saturation ceiling exceeds rate or trace budget")
                if self.duration_s * min(self.session_arrival_rates) / 2 < self.min_requests:
                    raise ValueError("Arrival rate needs enough final-window samples")
            if (
                self.session_slo_grid_s != sorted(set(self.session_slo_grid_s))
                or min(self.session_slo_grid_s) <= 0
                or max(self.session_slo_grid_s) > self.request_timeout_s
            ):
                raise ValueError("SLO thresholds must be increasing and within measurement timeout")
            if self.session_workload == "document":
                expected = self.duration_s * (
                    min(self.session_arrival_rates) / max(self.session_population_grid)
                    if self.session_arrival_rates
                    else min(self.session_rate_grid)
                )
                if (
                    not self.session_prime_documents
                    and self.session_turns <= self.session_min_returns
                ):
                    raise ValueError(
                        "Document lifetime must allow minimum returns after first touch"
                    )
                needed = self.session_min_returns + (0 if self.session_prime_documents else 1)
                if expected < needed:
                    raise ValueError(
                        "Capacity needs at least four expected returns per session; increase duration or rate"
                    )
                if self.session_capacity_multiples:
                    if (
                        self.session_capacity_multiples
                        != sorted(set(self.session_capacity_multiples))
                        or any(m not in (2, 4) for m in self.session_capacity_multiples)
                        or not self.session_gpu_capacity_documents
                        or not self.session_capacity_evidence
                        or not self.session_prime_documents
                        or not self.session_arrival_rates
                    ):
                        raise ValueError(
                            "Capacity multiples require measured GPU capacity evidence, priming and open-loop arrivals"
                        )
                    if self.session_population_grid != [
                        self.session_gpu_capacity_documents * m
                        for m in self.session_capacity_multiples
                    ]:
                        raise ValueError("Populations must match measured GPU capacity multiples")
            elif self.session_prime_documents or self.session_capacity_multiples:
                raise ValueError(
                    "Document priming and capacity multiples apply to document workloads only"
                )
            if (
                self.duration_s < 60
                or self.duration_s * max(self.session_rate_grid) * max(self.session_population_grid)
                > 100000
            ):
                raise ValueError(
                    "Capacity windows need at least 60 seconds and at most 100000 expected arrivals"
                )
            if (
                not self.session_arrival_rates
                and self.session_workload == "document"
                and self.duration_s
                * min(self.session_rate_grid)
                * min(self.session_population_grid)
                / 2
                < self.min_requests
            ):
                raise ValueError("Lowest arrival rate cannot meet final-window minimum samples")
        if self.project_id == "P03" and (
            self.cache_samples > self.corpus_size
            or self.min_requests > self.cache_samples
            or max(self.cache_concurrencies) > self.cache_samples
        ):
            raise ValueError(
                "Cache samples must fit the corpus and meet the minimum sample count and largest concurrency"
            )
        if self.project_id == "P03" and self.mode != "pilot":
            raise ValueError("P03 currently supports exploratory cache grids only")
        if (
            self.context_limit < 128 * max(self.cache_record_counts) + self.max_output_tokens
            and self.project_id == "P03"
        ):
            raise ValueError(
                "Increase context_limit to reserve input space for the cache document grid"
            )
        if self.context_limit <= self.max_output_tokens:
            raise ValueError("Context must leave room for the input document")
        return self
