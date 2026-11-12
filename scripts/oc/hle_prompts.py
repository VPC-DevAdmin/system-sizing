"""HLE's official prompts and the capsim final-answer marker, with no
OpenCompass dependency, so the job script and tests can use them."""
import re

FINAL_MARK = '\n<<<capsim-final-answer>>>\n'

# HLE's original prompts (centerforaisafety/hle, hle_eval/run_model_predictions.py
# before the June 2025 "unify system prompt" change): one for exact-answer
# questions, one for multiple choice. AA describes using exactly these two.
SYSTEM_EXACT_ANSWER = ('Your response should be in the following format:\n'
                       'Explanation: {your explanation for your final answer}\n'
                       'Exact Answer: {your succinct, final answer}\n'
                       'Confidence: {your confidence score between 0% and 100% for your answer}')
SYSTEM_MC = ('Your response should be in the following format:\n'
             'Explanation: {your explanation for your answer choice}\n'
             'Answer: {your chosen answer}\n'
             'Confidence: {your confidence score between 0% and 100% for your answer}')

# HLE's official judge prompt (hle_eval/run_judge_results.py), unchanged
# except that its placeholders are OpenCompass's column names.
HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {prediction}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


def hle_system_prompt(answer_type: str) -> str:
    return SYSTEM_EXACT_ANSWER if answer_type == 'exactMatch' else SYSTEM_MC


def final_answer(text: str, mark: str = FINAL_MARK) -> str:
    """The answer after the reasoning. A reply without the mark ran out of
    tokens while still reasoning: it has no answer."""
    if mark in text:
        return text.rsplit(mark, 1)[1].strip()
    return ''


_CORRECT = re.compile(r'correct\W{0,6}(yes|no)\b', re.IGNORECASE)


def hle_judgement(judge_reply: str) -> str:
    """'A' for correct, 'B' for incorrect, 'unknown' if the reply has no
    verdict. The last ``correct:`` line wins, since the prompt asks for it
    after the reasoning."""
    found = _CORRECT.findall(judge_reply or '')
    if not found:
        return 'unknown'
    return 'A' if found[-1].lower() == 'yes' else 'B'
