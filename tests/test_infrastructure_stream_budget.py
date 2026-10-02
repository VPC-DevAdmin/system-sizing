"""Regression coverage for delayed headers in fixed-token saturation runs."""
import asyncio
from dataclasses import asdict

from simulator import streaming
from simulator.headline_sweep import Rung, _Acc


class EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        pass


def test_delayed_headers_respect_explicit_budget_and_preserve_default(monkeypatch):
    monkeypatch.setattr(streaming, 'CONNECT_TIMEOUT_S', 0.01)

    async def opening():
        await asyncio.sleep(0.05)
        return EmptyStream()

    async def check():
        budgets = dict(pre_ttft_timeout_s=1, inter_token_timeout_s=1, hard_timeout_s=1)
        default = await streaming.consume_with_tiers(opening, **budgets)
        explicit = await streaming.consume_with_tiers(opening, stream_open_timeout_s=0.5, **budgets)
        assert default.error == 'connection_timeout'
        assert explicit.error == 'no_tokens'  # Header wait succeeded; empty stream is a separate failure.

    asyncio.run(check())


def test_opening_override_cannot_exceed_hard_request_budget():
    cancelled = []

    async def opening():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    async def check():
        result = await streaming.consume_with_tiers(
            opening, pre_ttft_timeout_s=10, inter_token_timeout_s=10,
            hard_timeout_s=0.01, stream_open_timeout_s=10,
        )
        assert result.error == 'connection_timeout'
        assert cancelled == [True]

    asyncio.run(check())


def test_rung_error_evidence_keeps_causes_and_reasoning_only_separate():
    acc = _Acc()
    acc.add([{'error': 'connection_timeout'}, {'error': 'connection_timeout'},
             {'error': 'decode_stalled'}, {'error': 'no_content_tokens'},
             {'ttft_ms': 42, 'tpot_ms': 3}])
    assert acc.errors == 3 and acc.no_content == 1 and acc.ttft == [42]
    assert acc.error_counts == {'connection_timeout': 2, 'decode_stalled': 1}
    assert _Acc().error_counts == {}
    rung = Rung(concurrency=1, in_flight=1, queue_depth=0, out_tok_s=1,
                prompt_tok_s=1, total_tok_s=2, error_counts=dict(acc.error_counts))
    assert asdict(rung)['error_counts'] == acc.error_counts
