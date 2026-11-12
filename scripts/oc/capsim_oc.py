"""OpenCompass add-ins for the capsim check against Artificial Analysis.

Imported into every OpenCompass process by ``sitecustomize.py`` in this
folder (the job puts the folder on PYTHONPATH and sets CAPSIM_OC=1), so the
names below are registered wherever OpenCompass builds a dataset, a
postprocessor or an evaluator.

- ``HLEOfficialDataset``: HLE's 2,158 text-only questions, each with the
  official system prompt for its answer type (the two prompts AA describes).
- ``capsim-final-answer``: hands graders only the final answer. The model
  client joins reasoning and answer with ``FINAL_MARK``; a reply without the
  mark ran out of tokens while still reasoning, so it gets an empty answer
  and grades as wrong, as it would in AA's runs.
- ``hle_official_judge_postprocess``: reads the ``correct: yes/no`` line of
  HLE's official judge prompt.
"""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.generic import get_final_results
from opencompass.registry import (DICT_POSTPROCESSORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from hle_prompts import (FINAL_MARK, HLE_JUDGE_PROMPT,  # noqa: F401
                         SYSTEM_EXACT_ANSWER, SYSTEM_MC)
from hle_prompts import final_answer as _final_answer
from hle_prompts import hle_judgement as _hle_judgement
from hle_prompts import hle_system_prompt


@LOAD_DATASET.register_module()
class HLEOfficialDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'cais/hle'):
        ds = load_dataset(path, split='test').filter(lambda x: x['image'] == '')
        ds = ds.map(lambda x: {'system': hle_system_prompt(x['answer_type'])})
        ds = ds.select_columns(['id', 'question', 'answer', 'answer_type', 'category', 'system'])
        return DatasetDict(train=ds, test=ds)


@TEXT_POSTPROCESSORS.register_module('capsim-final-answer')
def final_answer(text: str, mark: str = FINAL_MARK) -> str:
    return _final_answer(text, mark)


@DICT_POSTPROCESSORS.register_module()
def hle_official_judge_postprocess(output: dict, output_path: str) -> dict:
    judged, refs, replies = [], [], []
    for _, v in output.items():
        judged.append(_hle_judgement(v.get('prediction', '')))
        refs.append(v.get('gold', ''))
        replies.append(v.get('prediction', ''))
    result = get_final_results(judged, refs, replies)
    result['details'] = output
    return result
