"""Synthetic public document extraction tasks, exact-match JSON grading, v1.

Tasks test retrieval/transcription, not domain expertise or broad model quality.
The generated corpus is identical between paired arms; no sensitive user data.
"""

import json
import random


def build_corpus(size, records, seed, answer_records=4):
    rng = random.Random(seed)
    corpus = []
    for i in range(size):
        rows = []
        for j in range(records):
            rows.append(
                {
                    "record_id": f"R{i:04d}-{j:04d}",
                    "location": rng.choice(["Chicago", "Austin", "Boston", "Seattle"]),
                    "part": f"PART-{rng.randrange(100000, 999999)}",
                    "quantity": rng.randrange(1, 10000),
                    "status": rng.choice(["ready", "pending", "held"]),
                    "instructions": rng.choice(
                        [
                            "Keep the sealed container upright and inspect the label before dispatch.",
                            "Verify the batch number and retain the inspection record for the receiving team.",
                            "Place the shipment in the designated storage area until the supervisor approves release.",
                        ]
                    ),
                }
            )
        selected = sorted(rng.sample(range(records), min(answer_records, records)))
        wanted = [rows[j] for j in selected]
        prompt = (
            "Extract the requested records from this synthetic logistics document. "
            "Return ONLY a JSON array, in the requested order, containing every field exactly "
            "as recorded, with no additional fields or prose.\nRequested record IDs: "
            + json.dumps([r["record_id"] for r in wanted])
            + "\nDocument:\n"
            + "\n".join(json.dumps(r) for r in rows)
        )
        corpus.append({"id": f"document-{i:05d}", "prompt": prompt, "expected": wanted})
    return corpus


def grade(text, expected):
    try:

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON field")
                result[key] = value
            return result

        actual = json.loads(text, object_pairs_hook=unique_object)
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False
        )
    except (ValueError, TypeError):
        return False
