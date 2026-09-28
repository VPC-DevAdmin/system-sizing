"""Auditable, explicit human/model review; technical generation never implies acceptance."""

import shutil
import time

from ..benchmarks.process import Lease
from ..benchmarks.storage import read_json, sha256, write_json
from .diffusion_report import RUBRIC, inventory


def submit(path, reviewer, kind, items):
    if not reviewer.strip() or kind not in ("human", "model"):
        raise ValueError("A named human or model reviewer is required")
    with Lease(path / ".execution.lock"), Lease(path / ".worker.lock"):
        state = read_json(path / "state.json")
        if state["phase"] != "complete":
            raise ValueError("Finish generation before reviewing images")
        known = {r["blind_id"] for r in inventory(path)}
        if not items or len({r["blind_id"] for r in items}) != len(items):
            raise ValueError("Review items must be nonempty and unique")
        for row in items:
            if (
                row["blind_id"] not in known
                or set(row["scores"]) != set(RUBRIC)
                or any(type(v) is not bool for v in row["scores"].values())
            ):
                raise ValueError(
                    "Review requires known image IDs and a boolean for every rubric item"
                )
        review = (
            read_json(path / "image-review.json")
            if (path / "image-review.json").exists()
            else {"items": {}}
        )
        revision = str(time.time_ns())
        previous = None
        if (path / "evidence.zip").is_file():
            archive = path / "evidence-revisions" / f"before-review-{revision}.zip"
            archive.parent.mkdir(exist_ok=True)
            shutil.copy2(path / "evidence.zip", archive)
            previous = {"path": str(archive.relative_to(path)), "sha256": sha256(archive)}
        audit = {
            "time": time.time(),
            "reviewer": reviewer.strip(),
            "reviewer_kind": kind,
            "items": items,
            "previous_evidence": previous,
        }
        write_json(path / f"image-review-audit-{time.time_ns()}.json", audit)
        for row in items:
            review["items"][row["blind_id"]] = dict(
                row, reviewer=reviewer.strip(), reviewer_kind=kind
            )
        write_json(path / "image-review.json", review)
        human_complete = bool(known) and all(
            review["items"].get(identity, {}).get("reviewer_kind") == "human" for identity in known
        )
        state.update(
            outcome="reviewed_pilot" if human_complete else "pending_review", updated_at=time.time()
        )
        write_json(path / "state.json", state)
        from .report import export

        export(path, read_json(path / "comparison.json"))
        return {"reviewed": len(review["items"]), "total": len(known)}
