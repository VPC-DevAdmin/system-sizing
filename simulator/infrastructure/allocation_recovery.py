"""Bounded allocation retries without hiding failed attempts or changing settings."""


class AllocationFailure(RuntimeError):
    """A serving or measurement failure after allocation preparation."""


async def run_allocation(directory, operation, failed, *, enabled):
    """operation must finish owned cleanup before raising AllocationFailure.

    Other exceptions (including cleanup failure and cancellation) stop the campaign.
    Completed measurement rows are owned by the caller and survive a retry.
    """
    for attempt in range(2 if enabled else 1):
        evidence = directory if attempt == 0 else directory / f"retry-{attempt:02d}"
        try:
            await operation(evidence, attempt)
            return True
        except AllocationFailure as exc:
            if not enabled:
                raise
            failed(evidence, attempt, exc)
    return False
