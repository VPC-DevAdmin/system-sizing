"""GuideLLM's public data extension: cycle the frozen corpus until time expires.

The scheduler, request handling, measurements and export remain upstream code.
This file is copied into each evidence bundle with its exact invocation.
"""

import json
from pathlib import Path
from typing import Literal

from datasets import IterableDataset
from guidellm.data.deserializers.deserializer import DatasetDeserializer, DatasetDeserializerFactory
from guidellm.data.schemas import DataArgs


def repeat_rows(path: str):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Benchmark corpus is empty")
    while True:
        yield from rows


@DataArgs.register("capsim_corpus_cycle")
class CorpusArgs(DataArgs):
    kind: Literal["capsim_corpus_cycle"] = "capsim_corpus_cycle"
    path: str


@DatasetDeserializerFactory.register("capsim_corpus_cycle")
class CorpusDeserializer(DatasetDeserializer):
    def __call__(self, config, processor_factory, random_seed):
        return IterableDataset.from_generator(repeat_rows, gen_kwargs={"path": config.path})


if __name__ == "__main__":
    from guidellm.cli import cli

    cli()
