"""Frozen synthetic support calls and explicit two-stage quality gates."""

import re

ASR_MODEL = "Systran/faster-whisper-large-v3"
ASR_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
SPEECH_IMAGE = "capsim-speech:whisper-v1"


def call_corpus():
    cases = [
        ("damaged parcel", "send a replacement", "support team", "tomorrow"),
        ("missing invoice", "email the invoice", "billing team", "today"),
        ("late delivery", "track the shipment", "shipping team", "tomorrow"),
        ("wrong item", "arrange a return", "support team", "today"),
        ("duplicate charge", "issue a refund", "billing team", "friday"),
        ("broken screen", "book a repair", "repair team", "monday"),
        ("missing cable", "send a replacement", "shipping team", "friday"),
        ("incorrect address", "update the address", "support team", "monday"),
        ("lost receipt", "email the receipt", "billing team", "tomorrow"),
    ]
    rows = []
    for i, (issue, action, owner, deadline) in enumerate(cases):
        text = (
            f"Hello, thank you for calling customer support. How can I help you today? "
            f"Hello. I am calling about a {issue}. Please help me resolve this. "
            f"I understand. Let me check the order details. Thank you for waiting. "
            f"The problem recorded on your account is {issue}. We will {action}. "
            f"The {owner} will take responsibility for this action. "
            f"The agreed deadline is {deadline}. Can you confirm that this is what you need? "
            f"Yes. Please {action}. I understand that the {owner} will do this by {deadline}. "
            "Thank you for confirming. We have recorded the action and the deadline. Goodbye."
        )
        rows.append(
            dict(
                id=f"call-{i:02d}",
                reference=text,
                expected=dict(issue=issue, action=action, owner=owner, deadline=deadline),
                voice="en-us" if i % 2 == 0 else "en-gb",
                speed=145 + (i % 3) * 10,
                warmup=i == 8,
            )
        )
    return rows


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def word_error_rate(reference, hypothesis):
    a, b = words(reference), words(hypothesis)
    if not a:
        raise ValueError("Reference transcript is empty")
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / len(a)


def summary_prompt(transcript):
    return (
        "Summarize this support call as one JSON object with exactly four string fields: "
        "issue (short noun phrase), action (verb phrase without 'we will' or 'please'), "
        "owner (team name), deadline (day word). Use the explicit final agreement in the call. "
        "Use lower case. Do not invent missing information. Transcript follows:\n" + transcript
    )


def grade_summary(output, expected):
    return (
        isinstance(output, dict)
        and set(output) == set(expected)
        and all(
            isinstance(output[k], str) and words(output[k]) == words(v) for k, v in expected.items()
        )
    )
