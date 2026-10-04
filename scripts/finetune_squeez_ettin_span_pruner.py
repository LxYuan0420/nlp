#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.1",
#   "huggingface-hub==1.30.0",
#   "numpy==2.3.3",
#   "tensorboard==2.21.0",
#   "torch==2.14.0",
#   "transformers==5.16.1",
# ]
# ///
"""Fine-tune Ettin-68M (default) or Ettin-32M as a tool-output span pruner.

This tutorial script trains a native Hugging Face token classifier. It does not
generate XML, rewrite lines, or publish to the Hub. The visible pipeline is:

1. Load the official GLiNER train/dev/test files and the canonical 618-row
   Squeez test, both at pinned revisions.
2. Convert GLiNER rows with the upstream ``prepare_squeez_spans`` rule: drop
   the ``Query: ...\\n\\nTool output:\\n`` prefix and locate each ``RELEVANT``
   mention in the remaining chunk.
3. Pack ``(question, chunk)`` into 4096-token sliding windows. Query, special,
   and padding tokens are loss-masked (``-100``). Evidence tokens receive
   binary labels.
4. Fully fine-tune every parameter with positive class weight clipped at
   ``MAX_POSITIVE_CLASS_WEIGHT`` and float32 token cross-entropy. Trainer
   checkpoint selection uses validation token-F1. Final candidate and decode
   threshold selection uses validation Word-F1. Token-F1 is never treated as
   Word-F1.
5. Write TensorBoard logs, both GLiNER test Word-F1 views, and canonical line
   metrics. The reproduced 150M baseline uses all 9,595 rows. The
   evidence-bearing slice is a companion diagnostic, not the 0.588 protocol.

Default command, full data, 68M:

    uv run --script scripts/finetune_squeez_ettin_span_pruner.py

32M candidate:

    uv run --script scripts/finetune_squeez_ettin_span_pruner.py \
        --model-id jhu-clsp/ettin-encoder-32m \
        --output-dir squeez-ettin-32m-span-pruner

Resume:

    uv run --script scripts/finetune_squeez_ettin_span_pruner.py \
        --output-dir /content/drive/MyDrive/squeez-ettin-68m-span-pruner \
        --resume-from-checkpoint /content/drive/MyDrive/squeez-ettin-68m-span-pruner/checkpoints/checkpoint-100

Smoke: CPU conversion/unit checks, then four GPU steps:

    uv run --script scripts/finetune_squeez_ettin_span_pruner.py --smoke-run

Publishing stays outside this repository. The model consumes a question plus
one tool observation and emits verbatim character spans from that observation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import TensorBoardCallback

# Conversion and Word-F1 follow KRLabsOrg/acl-verbatim
# (scripts/experiments/prepare_squeez_spans.py and
# acl_verbatim/eval/span_metrics.py). Line projection follows
# KRLabsOrg/squeez scripts/evaluate_baselines.py::baseline_verbatim_v2.
# Canonical 618 P/R/F1 and empty-set behavior follow
# squeez.training.evaluate.compute_span_metrics (per-example set overlap,
# then macro mean). Compression follows that module's exact line-count formula.

GLINER_DATASET_ID = "KRLabsOrg/tool-output-extraction-swebench-gliner"
GLINER_REVISION = "915d80062eacdfe225118a67dd4353f38320db3a"
CANONICAL_DATASET_ID = "KRLabsOrg/tool-output-extraction-swebench"
CANONICAL_REVISION = "f7a861808a1bd40acb8c9e9ed858f5d3986568fe"
TOOL_OUTPUT_MARKER = "\n\nTool output:\n"
GLINER_FILES = {
    "train": "gliner_train.jsonl",
    "validation": "gliner_dev.jsonl",
    "test": "gliner_test.jsonl",
}
EXPECTED_GLINER_COUNTS = {"train": 51_917, "validation": 2_579, "test": 9_595}
EXPECTED_CANONICAL_TEST = 618
EXPECTED_CANONICAL_SWE_HELDOUT = 430
EXPECTED_PARENT_COUNTS = {"train": 4_108, "validation": 126, "test": 308}
SWE_SOURCE = "swe"

MODEL_REVISIONS = {
    "jhu-clsp/ettin-encoder-32m": "1b8ba06455dd44f80fc9c1ca9e22806157a57379",
    "jhu-clsp/ettin-encoder-68m": "ac19ae4bc51093b31c475665ac872a936d056cc2",
}
DEFAULT_MODEL_ID = "jhu-clsp/ettin-encoder-68m"
DEFAULT_OUTPUT_DIR = Path("squeez-ettin-68m-span-pruner")
SUPPORTED_32M_OUTPUT_DIR = Path("squeez-ettin-32m-span-pruner")

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 6.0
MAX_LENGTH = 4096
DOC_STRIDE = 256
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
INFER_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2.0e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
MIN_SPAN_CHARS = 30
MERGE_GAP_CHARS = 20
MAX_POSITIVE_CLASS_WEIGHT = 10.0
THRESHOLD_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
SMOKE_THRESHOLD_GRID = (0.20, 0.50)
SMOKE_TRAIN_ROWS = 64
SMOKE_EVAL_ROWS = 16
SMOKE_MAX_STEPS = 4
ID2LABEL = {0: "O", 1: "EVIDENCE"}
LABEL2ID = {"O": 0, "EVIDENCE": 1}
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class CharSpan:
    """One verbatim evidence span inside a chunk.

    Example::

        CharSpan(start=12, end=31, text="columns = []")
    """

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class SpanExample:
    """One GLiNER-native extractive row after prefix stripping.

    Example input fields after conversion::

        question="Find the failing test"
        chunk="test_auth.py::test_login FAILED\\n..."
        spans=(CharSpan(start=0, end=32, text="test_auth.py::test_login FAILED"),)
    """

    example_id: str
    question: str
    chunk: str
    paper_id: str
    chunk_index: int
    instance_id: str
    tool_type: str
    source: str
    spans: tuple[CharSpan, ...]


@dataclass(frozen=True, slots=True)
class CanonicalExample:
    """One official Squeez canonical observation.

    Example::

        query="Find the CSRF referer check"
        gold_line_ids=frozenset({41, 42, 43})
        tool_output="class CsrfViewMiddleware:\\n    ..."
    """

    instance_id: str
    query: str
    tool_output: str
    gold_line_ids: frozenset[int]
    is_irrelevant: bool
    tool_type: str
    source: str
    n_lines: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable, serializable experiment inputs."""

    model_id: str
    output_dir: str
    epochs: float
    seed: int
    resume_from_checkpoint: str | None
    smoke_run: bool

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def model_revision(self) -> str:
        return MODEL_REVISIONS[self.model_id]

    @property
    def resume_checkpoint_path(self) -> Path | None:
        if self.resume_from_checkpoint is None:
            return None
        return Path(self.resume_from_checkpoint)

    @property
    def threshold_grid(self) -> tuple[float, ...]:
        return SMOKE_THRESHOLD_GRID if self.smoke_run else THRESHOLD_GRID


class ContractChecks:
    """CPU-only checks for the upstream conversion and scorers."""

    def run(self) -> None:
        """Run synthetic conversion, Word-F1, and line-projection checks.

        Input:
            None. Uses fixed strings, not Hub rows.

        Output:
            ``None``. Raises ``ValueError`` when a contract fails.
        """

        self._check_conversion()
        self._check_word_f1()
        self._check_line_projection()
        self._check_canonical_line_metrics()
        self._check_pos_weight_cap()
        self._check_gliner_test_views()
        self._check_lost_supervision_guard()

    @staticmethod
    def _check_conversion() -> None:
        chunk = (
            "232: data = table.data\n233: columns = []\n234: for col in data.columns:"
        )
        entity = "233: columns = []\n"
        start = chunk.find(entity)
        row = {
            "input": f"Query: Find the columns list\n\nTool output:\n{chunk}",
            "output": {"entities": {"RELEVANT": [entity]}},
            "meta": {
                "instance_id": "demo__demo-1",
                "source": "swe",
                "tool_type": "read_file",
                "query": "Find the columns list",
                "chunk_index": 0,
            },
        }
        example = GlinerSpanSource.convert_row(row, row_index=0)
        if example is None:
            raise ValueError("Expected the synthetic GLiNER row to convert.")
        expected = (CharSpan(start=start, end=start + len(entity), text=entity),)
        if example.spans != expected:
            raise ValueError(f"Unexpected converted spans: {example.spans!r}")

    @staticmethod
    def _check_word_f1() -> None:
        chunk = "alpha beta gamma"
        gold = [(0, 5)]
        perfect = WordSpanMetrics.micro_word_prf([chunk], [gold], [gold])
        if perfect != (1.0, 1.0, 1.0):
            raise ValueError(f"Perfect Word-F1 failed: {perfect}")
        empty = WordSpanMetrics.micro_word_prf([chunk], [[]], [gold])
        if empty[1] != 0.0:
            raise ValueError(f"Empty prediction should have zero recall: {empty}")

    @staticmethod
    def _check_line_projection() -> None:
        text = "keep me\nskip me\nkeep me too"
        pred_lines = CanonicalLineMetrics.lines_from_spans(text, [(0, 7), (16, 27)])
        if pred_lines != frozenset({1, 3}):
            raise ValueError(f"Unexpected projected lines: {pred_lines}")

    @staticmethod
    def _check_canonical_line_metrics() -> None:
        empty_gold = CanonicalExample(
            instance_id="empty",
            query="q",
            tool_output="aa bb\ncc dd",
            gold_line_ids=frozenset(),
            is_irrelevant=True,
            tool_type="t",
            source="s",
            n_lines=2,
        )
        nonempty = CanonicalExample(
            instance_id="keep",
            query="q",
            tool_output="keep this line\nskip this line",
            gold_line_ids=frozenset({1}),
            is_irrelevant=False,
            tool_type="t",
            source="s",
            n_lines=2,
        )
        scorer = CanonicalLineMetrics()
        both_empty = scorer.score([empty_gold], [frozenset()])["strict_line"]
        if both_empty != {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": 1.0,
        }:
            raise ValueError(f"Both-empty sets must score 1.0: {both_empty}")
        pred_only = scorer.score([empty_gold], [frozenset({1})])["strict_line"]
        if pred_only != {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }:
            raise ValueError(
                f"Empty gold and nonempty pred must score 0.0: {pred_only}"
            )
        gold_only = scorer.score([nonempty], [frozenset()])["strict_line"]
        if gold_only != {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }:
            raise ValueError(
                f"Nonempty gold and empty pred must score 0.0: {gold_only}"
            )
        mixed = scorer.score(
            [empty_gold, nonempty],
            [frozenset(), frozenset({1})],
        )["strict_line"]
        if mixed["f1"] != 1.0 or mixed["exact_match"] != 1.0:
            raise ValueError(f"Macro mean of two perfect rows must be 1.0: {mixed}")
        mismatch = scorer.score([nonempty], [frozenset({2})])
        if mismatch["strict_line"]["exact_match"] != 0.0:
            raise ValueError(f"Mismatched sets must have exact_match 0.0: {mismatch}")
        if mismatch["records"][0]["strict_line"]["exact_match"] != 0.0:
            raise ValueError("Per-example exact_match must be stored on the record.")
        lines = CanonicalExample(
            instance_id="comp",
            query="q",
            tool_output="aaaa bbbb cccc\ndddd eeee",
            gold_line_ids=frozenset({1}),
            is_irrelevant=False,
            tool_type="t",
            source="s",
            n_lines=2,
        )
        compression = scorer.score([lines], [frozenset({1})])["compression"]
        if compression != 0.5:
            raise ValueError(
                "Compression must match upstream line counting "
                f"(1 - 1 filtered line / 2 original lines = 0.5): {compression}"
            )

    @staticmethod
    def _check_pos_weight_cap() -> None:
        class _LabelRows:
            def __getitem__(self, key: str) -> list[list[int]]:
                if key != "labels":
                    raise KeyError(key)
                return [[0] * 20 + [1]]

        weights = TokenSpanPreparer.class_weights(_LabelRows())
        if weights != (1.0, MAX_POSITIVE_CLASS_WEIGHT):
            raise ValueError(
                f"Positive weight must clip at {MAX_POSITIVE_CLASS_WEIGHT}: {weights}"
            )

    @staticmethod
    def _check_gliner_test_views() -> None:
        empty = SpanExample(
            example_id="empty:0:0",
            question="q",
            chunk="noise here",
            paper_id="squeez/t/empty",
            chunk_index=0,
            instance_id="empty",
            tool_type="t",
            source="s",
            spans=(),
        )
        positive = SpanExample(
            example_id="keep:0:1",
            question="q",
            chunk="keep this evidence",
            paper_id="squeez/t/keep",
            chunk_index=0,
            instance_id="keep",
            tool_type="t",
            source="s",
            spans=(CharSpan(start=0, end=18, text="keep this evidence"),),
        )
        views, _records = WordSpanMetrics().score_test_views(
            [empty, positive],
            [[(0, 10)], [(0, 18)]],
        )
        comparison = views["all_rows_150m_comparison"]
        diagnostic = views["evidence_bearing_diagnostic"]
        if comparison["n_examples"] != 2:
            raise ValueError(f"150M comparison must keep every row: {comparison}")
        if diagnostic["n_examples"] != 1 or diagnostic["n_relevant"] != 1:
            raise ValueError(
                f"Evidence diagnostic must keep positive rows only: {diagnostic}"
            )
        if diagnostic["word_level"] != {
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1": 1.0,
        }:
            raise ValueError(f"Unexpected evidence-bearing Word-F1: {diagnostic}")
        if comparison["protocol"] != "reproduced_0.588_all_rows":
            raise ValueError(
                f"Unexpected comparison protocol: {comparison['protocol']}"
            )

    @staticmethod
    def _check_lost_supervision_guard() -> None:
        GlinerSpanSource._require_no_lost_supervision("train", 0, 0)
        try:
            GlinerSpanSource._require_no_lost_supervision("validation", 1, 0)
        except ValueError as error:
            if "dropped_rows=1" not in str(error):
                raise ValueError(f"Unexpected supervision error: {error}") from error
        else:
            raise ValueError("Dropped rows must fail the full-data supervision guard.")
        try:
            GlinerSpanSource._require_no_lost_supervision("test", 0, 2)
        except ValueError as error:
            if "missed_entities=2" not in str(error):
                raise ValueError(f"Unexpected supervision error: {error}") from error
        else:
            raise ValueError(
                "Missed entities must fail the full-data supervision guard."
            )


class GlinerSpanSource:
    """Download, convert, and leakage-check official GLiNER splits."""

    def load_and_validate(self) -> tuple[dict[str, list[SpanExample]], dict[str, Any]]:
        """Load pinned GLiNER jsonl files and convert every official row.

        Input:
            Hub dataset ``KRLabsOrg/tool-output-extraction-swebench-gliner``
            at ``GLINER_REVISION``. One raw row looks like::

                {
                  "input": "Query: Find X\\n\\nTool output:\\n233: columns = []\\n",
                  "output": {"entities": {"RELEVANT": ["233: columns = []\\n"]}},
                  "meta": {
                    "instance_id": "astropy__astropy-12544",
                    "query": "Find X",
                    "chunk_index": 7,
                    "tool_type": "read_file",
                    "source": "swe"
                  }
                }

        Output:
            ``(splits, manifest)`` where ``splits`` maps
            ``train`` / ``validation`` / ``test`` to converted ``SpanExample``
            lists and ``manifest`` records counts, dropped rows, and disjoint
            ``instance_id`` sets.
        """

        raw = load_dataset(
            "json",
            data_files={
                split_name: hf_hub_download(
                    repo_id=GLINER_DATASET_ID,
                    filename=filename,
                    revision=GLINER_REVISION,
                    repo_type="dataset",
                )
                for split_name, filename in GLINER_FILES.items()
            },
        )
        splits: dict[str, list[SpanExample]] = {}
        dropped_rows: dict[str, int] = {}
        missed_entities: dict[str, int] = {}
        for split_name in ("train", "validation", "test"):
            if split_name not in raw:
                raise ValueError(f"GLiNER dataset is missing split {split_name!r}.")
            expected = EXPECTED_GLINER_COUNTS[split_name]
            if len(raw[split_name]) != expected:
                raise ValueError(
                    f"GLiNER {split_name} has {len(raw[split_name])} rows; "
                    f"expected {expected}."
                )
            converted: list[SpanExample] = []
            dropped = 0
            missed = 0
            for row_index, row in enumerate(raw[split_name]):
                example = self.convert_row(dict(row), row_index=row_index)
                if example is None:
                    dropped += 1
                    continue
                missed += self._missed_entity_count(dict(row), example)
                converted.append(example)
            self._require_no_lost_supervision(split_name, dropped, missed)
            dropped_rows[split_name] = dropped
            missed_entities[split_name] = missed
            splits[split_name] = converted

        instance_ids = {
            split_name: {example.instance_id for example in examples}
            for split_name, examples in splits.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = instance_ids[left] & instance_ids[right]
            if overlap:
                raise ValueError(
                    f"GLiNER {left}/{right} share instance_id values: "
                    f"{sorted(overlap)[:5]}"
                )
        for split_name, expected in EXPECTED_PARENT_COUNTS.items():
            if len(instance_ids[split_name]) != expected:
                raise ValueError(
                    f"GLiNER {split_name} covers {len(instance_ids[split_name])} "
                    f"parents; expected {expected}."
                )

        manifest = {
            "dataset_id": GLINER_DATASET_ID,
            "revision": GLINER_REVISION,
            "rows": {name: len(rows) for name, rows in splits.items()},
            "parents": {name: len(ids) for name, ids in instance_ids.items()},
            "dropped_rows": dropped_rows,
            "missed_entities": missed_entities,
            "positive_rows": {
                name: sum(1 for row in rows if row.spans)
                for name, rows in splits.items()
            },
        }
        return splits, manifest

    @staticmethod
    def _require_no_lost_supervision(
        split_name: str, dropped_rows: int, missed_entities: int
    ) -> None:
        if dropped_rows or missed_entities:
            raise ValueError(
                f"GLiNER {split_name} lost supervision: "
                f"dropped_rows={dropped_rows}, missed_entities={missed_entities}. "
                "Full-data runs cannot silently drop rows or miss entities."
            )

    @classmethod
    def convert_row(cls, row: dict[str, Any], row_index: int) -> SpanExample | None:
        """Apply the upstream GLiNER-to-span conversion.

        Attribution: KRLabsOrg/acl-verbatim
        ``scripts/experiments/prepare_squeez_spans.py``.
        """

        chunk = cls._extract_chunk(row.get("input") or "")
        meta = row.get("meta") or {}
        question = meta.get("query") or ""
        if chunk is None or not question:
            return None

        entities = ((row.get("output") or {}).get("entities") or {}).get(
            "RELEVANT"
        ) or []
        spans: list[CharSpan] = []
        cursor = 0
        for entity_text in entities:
            if not entity_text:
                continue
            start = chunk.find(entity_text, cursor)
            if start == -1:
                start = chunk.find(entity_text)
            if start == -1:
                continue
            end = start + len(entity_text)
            spans.append(CharSpan(start=start, end=end, text=entity_text))
            cursor = end

        instance_id = str(meta.get("instance_id") or "")
        tool_type = str(meta.get("tool_type") or "")
        chunk_index = int(meta.get("chunk_index", 0) or 0)
        return SpanExample(
            example_id=f"{instance_id}:{chunk_index}:{row_index}",
            question=question,
            chunk=chunk,
            paper_id=f"squeez/{tool_type}/{instance_id}",
            chunk_index=chunk_index,
            instance_id=instance_id,
            tool_type=tool_type,
            source=str(meta.get("source") or ""),
            spans=tuple(spans),
        )

    @staticmethod
    def _extract_chunk(input_text: str) -> str | None:
        index = input_text.find(TOOL_OUTPUT_MARKER)
        if index == -1:
            return None
        return input_text[index + len(TOOL_OUTPUT_MARKER) :]

    @classmethod
    def _missed_entity_count(cls, row: dict[str, Any], example: SpanExample) -> int:
        entities = ((row.get("output") or {}).get("entities") or {}).get(
            "RELEVANT"
        ) or []
        requested = sum(1 for text in entities if text)
        return max(0, requested - len(example.spans))


class CanonicalLineSource:
    """Load the official 618-row canonical Squeez test."""

    def load_and_validate(self) -> tuple[list[CanonicalExample], dict[str, Any]]:
        """Load pinned ``canonical_test.jsonl``.

        Input:
            One Hub row::

                {
                  "instance_id": "django__django-11270",
                  "query": "Find the CSRF referer check",
                  "tool_output": "class CsrfViewMiddleware:\\n...",
                  "gold_spans": [{"start_line": 41, "end_line": 52}],
                  "is_irrelevant": false
                }

        Output:
            ``(examples, manifest)``. Line ids are 1-indexed inclusive over
            ``tool_output.split("\\n")``.
        """

        raw = load_dataset(
            "json",
            data_files={
                "test": hf_hub_download(
                    repo_id=CANONICAL_DATASET_ID,
                    filename="canonical_test.jsonl",
                    revision=CANONICAL_REVISION,
                    repo_type="dataset",
                )
            },
            split="test",
        )
        if len(raw) != EXPECTED_CANONICAL_TEST:
            raise ValueError(
                f"Canonical test has {len(raw)} rows; expected {EXPECTED_CANONICAL_TEST}."
            )
        examples = [self._convert_row(dict(row)) for row in raw]
        swe_rows = sum(1 for example in examples if example.source == SWE_SOURCE)
        if swe_rows != EXPECTED_CANONICAL_SWE_HELDOUT:
            raise ValueError(
                f"Canonical SWE held-out slice has {swe_rows} rows; "
                f"expected {EXPECTED_CANONICAL_SWE_HELDOUT}."
            )
        manifest = {
            "dataset_id": CANONICAL_DATASET_ID,
            "revision": CANONICAL_REVISION,
            "rows": len(examples),
            "swe_repo_held_out": swe_rows,
            "irrelevant": sum(1 for example in examples if example.is_irrelevant),
        }
        return examples, manifest

    @staticmethod
    def _convert_row(row: dict[str, Any]) -> CanonicalExample:
        tool_output = row.get("tool_output") or ""
        n_lines = len(tool_output.split("\n"))
        gold_ids: set[int] = set()
        for span in row.get("gold_spans") or []:
            start_line = int(span["start_line"])
            end_line = int(span["end_line"])
            if start_line < 1 or end_line > n_lines or start_line > end_line:
                raise ValueError(
                    f"{row.get('instance_id')} has OOB gold_spans "
                    f"{start_line}-{end_line} for {n_lines} lines."
                )
            gold_ids.update(range(start_line, end_line + 1))
        is_irrelevant = bool(row.get("is_irrelevant"))
        if is_irrelevant != (not gold_ids):
            raise ValueError(
                f"{row.get('instance_id')} is_irrelevant={is_irrelevant} "
                f"but gold lines={sorted(gold_ids)[:8]}."
            )
        return CanonicalExample(
            instance_id=str(row.get("instance_id") or ""),
            query=str(row.get("query") or ""),
            tool_output=tool_output,
            gold_line_ids=frozenset(gold_ids),
            is_irrelevant=is_irrelevant,
            tool_type=str(row.get("tool_type") or ""),
            source=str(row.get("source") or ""),
            n_lines=n_lines,
        )


class TokenSpanPreparer:
    """Turn span examples into loss-masked token-classification windows."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def windows(self, examples: Sequence[SpanExample]) -> Dataset:
        """Encode each example into one or more 4096-token windows.

        Input:
            Converted GLiNER rows. Long chunks overflow with ``DOC_STRIDE``.

        Output:
            A ``Dataset`` of ``input_ids`` / ``attention_mask`` / ``labels``.
            Query, special, and empty offsets are ``-100``::

                {"input_ids": [50281, 12, 45, ...], "labels": [-100, -100, 1, 0]}
        """

        rows: list[dict[str, list[int]]] = []
        for example in examples:
            rows.extend(
                {
                    "input_ids": window["input_ids"],
                    "attention_mask": window["attention_mask"],
                    "labels": window["labels"],
                }
                for window in self._encode_example(example, label=True)
            )
        if not rows:
            raise ValueError("Window encoding produced no labeled rows.")
        return Dataset.from_list(rows)

    def _encode_example(
        self,
        example: SpanExample,
        label: bool,
    ) -> list[dict[str, Any]]:
        question_tokens = self.tokenizer(example.question, add_special_tokens=False)
        if len(question_tokens["input_ids"]) >= MAX_LENGTH - 8:
            raise ValueError(
                f"Question for {example.example_id} does not leave room for the chunk."
            )
        encoded = self.tokenizer(
            example.question,
            example.chunk,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        gold = [(span.start, span.end) for span in example.spans]
        windows: list[dict[str, Any]] = []
        for window_index in range(len(encoded["input_ids"])):
            offsets = encoded["offset_mapping"][window_index]
            sequence_ids = encoded.sequence_ids(window_index)
            labels: list[int] = []
            for sequence_id, (start, end) in zip(sequence_ids, offsets, strict=True):
                if sequence_id != 1 or start == end:
                    labels.append(-100)
                    continue
                if not label:
                    labels.append(0)
                    continue
                labels.append(
                    1
                    if any(
                        start < gold_end and end > gold_start
                        for gold_start, gold_end in gold
                    )
                    else 0
                )
            if all(value == -100 for value in labels):
                continue
            windows.append(
                {
                    "input_ids": list(encoded["input_ids"][window_index]),
                    "attention_mask": list(encoded["attention_mask"][window_index]),
                    "labels": labels,
                    "offset_mapping": [list(span) for span in offsets],
                    "sequence_ids": [
                        -1 if sequence_id is None else sequence_id
                        for sequence_id in sequence_ids
                    ],
                }
            )
        if not windows and example.spans:
            raise ValueError(f"No supervised window for {example.example_id}.")
        return windows

    @staticmethod
    def class_weights(dataset: Dataset) -> tuple[float, float]:
        """Inverse-frequency weights over non-masked token labels.

        Input:
            Window dataset with ``labels``.

        Output:
            ``(weight_O, weight_EVIDENCE)``. The positive weight is
            ``min(n_neg / n_pos, MAX_POSITIVE_CLASS_WEIGHT)``, matching official
            ``PooledLineClassifier`` which clamps at ``10.0``.
        """

        negative = 0
        positive = 0
        for labels in dataset["labels"]:
            array = np.asarray(labels)
            negative += int((array == 0).sum())
            positive += int((array == 1).sum())
        if positive == 0:
            raise ValueError("Training windows contain no positive evidence tokens.")
        return (1.0, min(negative / positive, MAX_POSITIVE_CLASS_WEIGHT))


class WordSpanMetrics:
    """Upstream whitespace Word-P/R/F1 and card companion metrics.

    Attribution: KRLabsOrg/acl-verbatim ``acl_verbatim/eval/span_metrics.py``.
    """

    def score(
        self,
        examples: Sequence[SpanExample],
        predicted: Sequence[Sequence[tuple[int, int]]],
    ) -> dict[str, Any]:
        """Score predicted char spans against GLiNER gold.

        Input:
            ``examples`` and aligned ``predicted`` span lists, for example
            ``[[(20, 36)], []]``.

        Output:
            Card-style summary::

                {
                  "n_examples": 9595,
                  "word_level": {"micro_precision": 0.51, "micro_recall": 0.70, "micro_f1": 0.59},
                  "span_level_iou@0.5": {"micro_f1": 0.51},
                  "recall_any_overlap": 0.81,
                  "over_prediction_ratio": 1.57
                }
        """

        word_tp = word_fp = word_fn = 0
        iou_tp = iou_fp = iou_fn = 0
        any_hit = 0
        any_total = 0
        pred_total = 0
        gold_total = 0
        records: list[dict[str, Any]] = []
        for example, pred_spans in zip(examples, predicted, strict=True):
            gold_spans = [(span.start, span.end) for span in example.spans]
            words = tokenize(example.chunk)
            pred_words = spans_to_word_indices(words, pred_spans)
            gold_words = spans_to_word_indices(words, gold_spans)
            tp = len(pred_words & gold_words)
            fp = len(pred_words - gold_words)
            fn = len(gold_words - pred_words)
            word_tp += tp
            word_fp += fp
            word_fn += fn
            _, _, _, span_tp, span_fp, span_fn = span_iou_prf(
                pred_spans, gold_spans, 0.5
            )
            iou_tp += span_tp
            iou_fp += span_fp
            iou_fn += span_fn
            hit, total = recall_any_overlap(pred_spans, gold_spans)
            any_hit += hit
            any_total += total
            pred_total += len(pred_spans)
            gold_total += len(gold_spans)
            precision, recall, f1 = prf(tp, fp, fn)
            records.append(
                {
                    "example_id": example.example_id,
                    "paper_id": example.paper_id,
                    "chunk_index": example.chunk_index,
                    "gold_spans": [asdict(span) for span in example.spans],
                    "predicted_spans": [
                        {"start": start, "end": end} for start, end in pred_spans
                    ],
                    "word_metrics": {
                        "p": precision,
                        "r": recall,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    },
                }
            )
        word_p, word_r, word_f1 = prf(word_tp, word_fp, word_fn)
        iou_p, iou_r, iou_f1 = prf(iou_tp, iou_fp, iou_fn)
        return {
            "n_examples": len(examples),
            "n_relevant": sum(1 for example in examples if example.spans),
            "n_irrelevant": sum(1 for example in examples if not example.spans),
            "word_level": {
                "micro_precision": word_p,
                "micro_recall": word_r,
                "micro_f1": word_f1,
            },
            "span_level_iou@0.5": {
                "micro_precision": iou_p,
                "micro_recall": iou_r,
                "micro_f1": iou_f1,
            },
            "recall_any_overlap": (any_hit / any_total) if any_total else 0.0,
            "over_prediction_ratio": (pred_total / gold_total) if gold_total else 0.0,
            "records": records,
        }

    @staticmethod
    def micro_word_prf(
        chunks: Sequence[str],
        predicted: Sequence[Sequence[tuple[int, int]]],
        gold: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[float, float, float]:
        word_tp = word_fp = word_fn = 0
        for chunk, pred_spans, gold_spans in zip(chunks, predicted, gold, strict=True):
            words = tokenize(chunk)
            pred_words = spans_to_word_indices(words, pred_spans)
            gold_words = spans_to_word_indices(words, gold_spans)
            word_tp += len(pred_words & gold_words)
            word_fp += len(pred_words - gold_words)
            word_fn += len(gold_words - pred_words)
        return prf(word_tp, word_fp, word_fn)

    def score_test_views(
        self,
        examples: Sequence[SpanExample],
        predicted: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Score both official GLiNER test views from one aligned prediction list.

        ``all_rows_150m_comparison`` keeps every row, including empty-gold
        negatives, and matches the reproduced 0.588 baseline protocol.
        ``evidence_bearing_diagnostic`` keeps only rows with gold spans.
        """

        all_rows = self.score(examples, predicted)
        records = all_rows.pop("records")
        evidence_examples: list[SpanExample] = []
        evidence_preds: list[Sequence[tuple[int, int]]] = []
        for example, pred_spans in zip(examples, predicted, strict=True):
            if example.spans:
                evidence_examples.append(example)
                evidence_preds.append(pred_spans)
        evidence = self.score(evidence_examples, evidence_preds)
        evidence.pop("records", None)
        evidence["protocol"] = "evidence_bearing_only"
        evidence["role"] = "companion diagnostic"
        all_rows["protocol"] = "reproduced_0.588_all_rows"
        all_rows["role"] = "150M comparison"
        return {
            "all_rows_150m_comparison": all_rows,
            "evidence_bearing_diagnostic": evidence,
        }, records


class CanonicalLineMetrics:
    """Project char spans onto 1-indexed lines and score official gold sets.

    Attribution: KRLabsOrg/squeez ``baseline_verbatim_v2`` keeps any line
    whose character range intersects a predicted span. Per-example set
    P/R/F1, empty-set behavior, and the macro mean follow
    ``squeez.training.evaluate.compute_span_metrics``. Compression uses
    that module's exact newline-count definition.
    """

    def score(
        self,
        examples: Sequence[CanonicalExample],
        predicted_line_ids: Sequence[frozenset[int]],
    ) -> dict[str, Any]:
        """Score predicted line sets against canonical ``gold_spans``.

        Input:
            Official canonical rows and aligned 1-indexed pred line sets.

        Output::

            {
              "n_examples": 618,
              "strict_line": {
                "precision": 0.80,
                "recall": 0.86,
                "f1": 0.80,
                "exact_match": 0.55
              },
              "empty_accuracy": 0.9,
              "compression": 0.92
            }

        ``strict_line`` is the macro mean of per-example set scores, including
        official ``exact_match``. Both-empty sets score 1.0; exactly one empty
        set scores 0.0.
        """

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        exact_matches: list[float] = []
        empty_flags: list[float] = []
        compression_values: list[float] = []
        records: list[dict[str, Any]] = []
        for example, pred_ids in zip(examples, predicted_line_ids, strict=True):
            pred_ids = frozenset(pred_ids)
            lines = example.tool_output.split("\n")
            gold_lines = [
                lines[index - 1].strip()
                for index in sorted(example.gold_line_ids)
                if lines[index - 1].strip()
            ]
            pred_lines = [
                lines[index - 1].strip()
                for index in sorted(pred_ids)
                if 1 <= index <= len(lines) and lines[index - 1].strip()
            ]
            precision, recall, f1, exact_match = official_line_set_prf(
                pred_lines, gold_lines
            )
            empty = 1.0 if (not gold_lines) == (not pred_lines) else 0.0
            compression = official_line_compression(
                example.tool_output, "\n".join(pred_lines)
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            exact_matches.append(exact_match)
            empty_flags.append(empty)
            compression_values.append(compression)
            records.append(
                {
                    "instance_id": example.instance_id,
                    "gold_line_ids": sorted(example.gold_line_ids),
                    "predicted_line_ids": sorted(pred_ids),
                    "strict_line": {
                        "p": precision,
                        "r": recall,
                        "f1": f1,
                        "exact_match": exact_match,
                    },
                    "empty_accuracy": empty,
                    "compression": compression,
                }
            )
        return {
            "n_examples": len(examples),
            "aggregation": "macro_mean",
            "strict_line": {
                "precision": (
                    round(statistics.mean(precisions), 4) if precisions else 0.0
                ),
                "recall": round(statistics.mean(recalls), 4) if recalls else 0.0,
                "f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
                "exact_match": (
                    round(statistics.mean(exact_matches), 4) if exact_matches else 0.0
                ),
            },
            "empty_accuracy": (
                round(statistics.mean(empty_flags), 4) if empty_flags else 0.0
            ),
            "compression": (
                round(statistics.mean(compression_values), 4)
                if compression_values
                else 0.0
            ),
            "records": records,
        }

    @staticmethod
    def lines_from_spans(
        text: str,
        spans: Sequence[tuple[int, int]],
    ) -> frozenset[int]:
        """Return 1-indexed line ids overlapped by ``spans``."""

        lines = text.split("\n")
        line_offsets: list[tuple[int, int]] = []
        cursor = 0
        for line in lines:
            line_offsets.append((cursor, cursor + len(line)))
            cursor += len(line) + 1
        kept: set[int] = set()
        for start, end in spans:
            for index, (line_start, line_end) in enumerate(line_offsets, start=1):
                if end <= line_start or start >= line_end:
                    continue
                if lines[index - 1].strip():
                    kept.add(index)
        return frozenset(kept)


class SpanPredictor:
    """Windowed token-probability inference and span decode."""

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        preparer: TokenSpanPreparer,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.preparer = preparer

    def char_scores(self, question: str, context: str) -> np.ndarray:
        """Max-pool evidence probabilities onto context characters."""

        return self.char_score_rows([(question, context)])[0]

    def char_score_rows(self, pairs: Sequence[tuple[str, str]]) -> list[np.ndarray]:
        """Batch windows across examples, then max-pool onto each context.

        Input:
            ``(question, context)`` pairs. Empty contexts stay all zeros.

        Output:
            One ``float32`` score array per pair. Overlapping windows still
            use per-character ``max``, identical to one-window-at-a-time.
        """

        scores = [np.zeros(len(context), dtype=np.float32) for _, context in pairs]
        jobs: list[tuple[int, dict[str, Any]]] = []
        for index, (question, context) in enumerate(pairs):
            if not context:
                continue
            example = SpanExample(
                example_id=f"infer:{index}",
                question=question,
                chunk=context,
                paper_id="",
                chunk_index=0,
                instance_id="",
                tool_type="",
                source="",
                spans=(),
            )
            for window in self.preparer._encode_example(example, label=False):
                jobs.append((index, window))
        if not jobs:
            return scores
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id for batched inference."
            )
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(jobs), INFER_BATCH_SIZE):
                batch_jobs = jobs[start : start + INFER_BATCH_SIZE]
                padded = self.tokenizer.pad(
                    {
                        "input_ids": [window["input_ids"] for _, window in batch_jobs],
                        "attention_mask": [
                            window["attention_mask"] for _, window in batch_jobs
                        ],
                    },
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(
                    input_ids=padded["input_ids"].to(device),
                    attention_mask=padded["attention_mask"].to(device),
                ).logits
                probabilities = (
                    F.softmax(logits.float(), dim=-1)[..., 1].detach().cpu().numpy()
                )
                for (example_index, window), row in zip(
                    batch_jobs, probabilities, strict=True
                ):
                    self._accumulate_window(
                        scores[example_index],
                        row[: len(window["input_ids"])],
                        window,
                    )
        return scores

    @staticmethod
    def _accumulate_window(
        scores: np.ndarray,
        probabilities: np.ndarray,
        window: dict[str, Any],
    ) -> None:
        for probability, sequence_id, offset in zip(
            probabilities,
            window["sequence_ids"],
            window["offset_mapping"],
            strict=True,
        ):
            start, end = int(offset[0]), int(offset[1])
            if int(sequence_id) != 1 or start == end:
                continue
            end = min(end, len(scores))
            if start >= end:
                continue
            scores[start:end] = np.maximum(scores[start:end], float(probability))

    def predict_spans(
        self,
        examples: Sequence[SpanExample],
        threshold: float,
    ) -> list[list[tuple[int, int]]]:
        score_rows = self.char_score_rows(
            [(example.question, example.chunk) for example in examples]
        )
        return [
            decode_char_spans(
                scores,
                threshold=threshold,
                min_span_chars=MIN_SPAN_CHARS,
                merge_gap_chars=MERGE_GAP_CHARS,
            )
            for scores in score_rows
        ]

    def predict_canonical_lines(
        self,
        examples: Sequence[CanonicalExample],
        threshold: float,
    ) -> list[frozenset[int]]:
        score_rows = self.char_score_rows(
            [(example.query, example.tool_output) for example in examples]
        )
        predicted: list[frozenset[int]] = []
        for example, scores in zip(examples, score_rows, strict=True):
            spans = decode_char_spans(
                scores,
                threshold=threshold,
                min_span_chars=MIN_SPAN_CHARS,
                merge_gap_chars=MERGE_GAP_CHARS,
            )
            predicted.append(
                CanonicalLineMetrics.lines_from_spans(example.tool_output, spans)
            )
        return predicted


class WeightedTokenTrainer(Trainer):
    """Trainer that applies capped inverse-frequency weights to float32 CE."""

    def __init__(
        self, *args: Any, class_weights: Sequence[float], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = tuple(class_weights)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.as_tensor(
            self.class_weights, device=logits.device, dtype=torch.float32
        )
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weights,
            ignore_index=-100,
        )
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


class SpanPrunerExperiment:
    """Train one candidate, select its checkpoint and threshold, then evaluate it."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.word_metrics = WordSpanMetrics()
        self.line_metrics = CanonicalLineMetrics()

    def run(self) -> None:
        set_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        ContractChecks().run()
        gliner_splits, gliner_manifest = GlinerSpanSource().load_and_validate()
        canonical_examples, canonical_manifest = (
            CanonicalLineSource().load_and_validate()
        )
        self._print_source_summary(gliner_manifest, gliner_splits["train"][0])

        if self.config.smoke_run:
            gliner_splits = self._smoke_gliner(gliner_splits)
            canonical_examples = canonical_examples[:SMOKE_EVAL_ROWS]

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            use_fast=True,
        )
        preparer = TokenSpanPreparer(tokenizer)
        train_windows = preparer.windows(gliner_splits["train"])
        eval_windows = preparer.windows(gliner_splits["validation"])
        class_weights = TokenSpanPreparer.class_weights(train_windows)
        manifest = {
            "gliner": gliner_manifest,
            "canonical": canonical_manifest,
            "smoke_run": self.config.smoke_run,
            "train_windows": len(train_windows),
            "validation_windows": len(eval_windows),
            "class_weights": {
                "O": class_weights[0],
                "EVIDENCE": class_weights[1],
            },
            "used_rows": {name: len(rows) for name, rows in gliner_splits.items()},
            "canonical_eval_rows": len(canonical_examples),
        }
        self._write_json(output_dir / "dataset_manifest.json", manifest)

        self._require_training_device()
        model = self._load_model().to(torch.device("cuda"))
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if total_parameters != trainable_parameters:
            raise ValueError(
                f"Expected full fine-tuning, but only {trainable_parameters:,}/"
                f"{total_parameters:,} parameters are trainable."
            )
        print(
            f"Training {trainable_parameters:,} parameters on "
            f"{len(train_windows):,} windows "
            f"(pos_weight={class_weights[1]:.3f})."
        )

        trainer = WeightedTokenTrainer(
            class_weights=class_weights,
            model=model,
            args=self._training_arguments(),
            train_dataset=train_windows,
            eval_dataset=eval_windows,
            data_collator=DataCollatorForTokenClassification(
                tokenizer, padding=True, label_pad_token_id=-100
            ),
            processing_class=tokenizer,
            compute_metrics=compute_token_metrics,
            callbacks=[
                TensorBoardCallback(
                    tb_writer=SummaryWriter(
                        log_dir=str(self.config.output_path / "tensorboard")
                    )
                )
            ],
        )
        started_at = time.perf_counter()
        resume_checkpoint = self.config.resume_checkpoint_path
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )
        runtime_seconds = time.perf_counter() - started_at
        trainer.save_state()
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

        predictor = SpanPredictor(trainer.model, tokenizer, preparer)
        selected_threshold, sweep = self._select_threshold(
            predictor, gliner_splits["validation"]
        )
        self._stamp_decode_config(trainer.model, selected_threshold)
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        final_word, word_records = self._gliner_bundle(
            predictor, gliner_splits["test"], selected_threshold
        )
        final_line, line_records = self._canonical_bundle(
            predictor, canonical_examples, selected_threshold
        )
        comparison = {
            "fine_tuned_model": {
                "selected_threshold": selected_threshold,
                "threshold_sweep_dev": sweep,
                "gliner_test_word": final_word,
                "canonical_line": final_line,
            },
            "selection": {
                "checkpoint": "validation token-F1",
                "candidate_threshold": "validation Word-F1",
            },
            "decode": {
                "min_span_chars": MIN_SPAN_CHARS,
                "merge_gap_chars": MERGE_GAP_CHARS,
                "max_length": MAX_LENGTH,
                "doc_stride": DOC_STRIDE,
            },
        }
        self._write_json(
            output_dir / "training_metrics.json",
            {
                "train_result": train_result.metrics,
                "runtime_seconds": runtime_seconds,
                "epochs_requested": self.config.epochs,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "log_history": trainer.state.log_history,
            },
        )
        self._write_json(output_dir / "final_metrics.json", comparison)
        self._write_jsonl(output_dir / "gliner_test_predictions.jsonl", word_records)
        self._write_jsonl(output_dir / "canonical_test_predictions.jsonl", line_records)
        self._write_json(
            output_dir / "run_config.json",
            self._run_config(total_parameters, class_weights, selected_threshold),
        )
        self._write_json(output_dir / "package_versions.json", self._package_versions())
        print(f"Training and evaluation artifacts saved to {output_dir.resolve()}")

    def _select_threshold(
        self,
        predictor: SpanPredictor,
        dev_examples: Sequence[SpanExample],
    ) -> tuple[float, list[dict[str, Any]]]:
        """Pick the final candidate decode threshold on validation Word-F1.

        Trainer checkpoint selection uses validation token-F1. This sweep
        does not choose checkpoints.

        Input:
            Fine-tuned predictor and the official (or smoke) validation rows.

        Output:
            ``(0.2, [{"threshold": 0.2, "micro_f1": 0.61}, ...])``.
        """

        char_rows = predictor.char_score_rows(
            [(example.question, example.chunk) for example in dev_examples]
        )
        sweep: list[dict[str, Any]] = []
        best_threshold = self.config.threshold_grid[0]
        best_f1 = -1.0
        for threshold in self.config.threshold_grid:
            predicted = [
                decode_char_spans(
                    scores,
                    threshold=threshold,
                    min_span_chars=MIN_SPAN_CHARS,
                    merge_gap_chars=MERGE_GAP_CHARS,
                )
                for scores in char_rows
            ]
            metrics = self.word_metrics.score(dev_examples, predicted)
            word = metrics["word_level"]
            sweep.append(
                {
                    "threshold": threshold,
                    "micro_precision": word["micro_precision"],
                    "micro_recall": word["micro_recall"],
                    "micro_f1": word["micro_f1"],
                }
            )
            if word["micro_f1"] > best_f1:
                best_f1 = word["micro_f1"]
                best_threshold = threshold
        return best_threshold, sweep

    def _gliner_bundle(
        self,
        predictor: SpanPredictor,
        examples: Sequence[SpanExample],
        threshold: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        predicted = predictor.predict_spans(examples, threshold)
        return self.word_metrics.score_test_views(examples, predicted)

    def _canonical_bundle(
        self,
        predictor: SpanPredictor,
        examples: Sequence[CanonicalExample],
        threshold: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        predicted = predictor.predict_canonical_lines(examples, threshold)
        all_rows = self.line_metrics.score(examples, predicted)
        records = all_rows.pop("records")
        swe_examples = [example for example in examples if example.source == SWE_SOURCE]
        swe_predicted = [
            pred
            for example, pred in zip(examples, predicted, strict=True)
            if example.source == SWE_SOURCE
        ]
        if not self.config.smoke_run:
            if len(examples) != EXPECTED_CANONICAL_TEST:
                raise ValueError(
                    f"Canonical bundle has {len(examples)} rows; "
                    f"expected {EXPECTED_CANONICAL_TEST}."
                )
            if len(swe_examples) != EXPECTED_CANONICAL_SWE_HELDOUT:
                raise ValueError(
                    f"Canonical SWE held-out slice has {len(swe_examples)} rows; "
                    f"expected {EXPECTED_CANONICAL_SWE_HELDOUT}."
                )
        swe_rows = self.line_metrics.score(swe_examples, swe_predicted)
        swe_rows.pop("records", None)
        return {
            "all_rows": all_rows,
            "swe_repo_held_out": swe_rows,
        }, records

    def _stamp_decode_config(self, model: Any, selected_threshold: float) -> None:
        model.config.update(
            {
                "selected_threshold": float(selected_threshold),
                "span_decode": {
                    "min_span_chars": MIN_SPAN_CHARS,
                    "merge_gap_chars": MERGE_GAP_CHARS,
                    "max_length": MAX_LENGTH,
                    "doc_stride": DOC_STRIDE,
                },
                "base_model_id": self.config.model_id,
                "base_model_revision": self.config.model_revision,
            }
        )

    def _load_model(self) -> Any:
        return AutoModelForTokenClassification.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dtype=torch.float32,
            attn_implementation="sdpa",
            ignore_mismatched_sizes=True,
        )

    def _training_arguments(self) -> TrainingArguments:
        smoke = self.config.smoke_run
        return TrainingArguments(
            output_dir=str(self.config.output_path / "checkpoints"),
            num_train_epochs=1 if smoke else self.config.epochs,
            max_steps=SMOKE_MAX_STEPS if smoke else -1,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=1 if smoke else GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            eval_strategy="no" if smoke else "epoch",
            save_strategy="no" if smoke else "epoch",
            logging_strategy="steps",
            logging_steps=1 if smoke else LOGGING_STEPS,
            report_to=[],
            load_best_model_at_end=not smoke,
            metric_for_best_model="eval_token_f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config.seed,
            data_seed=self.config.seed,
            remove_unused_columns=True,
        )

    def _require_training_device(self) -> None:
        if torch.cuda.is_available():
            return
        raise RuntimeError(
            "Training and the smoke GPU step require CUDA. CPU-only unit checks "
            "already finished before this guard."
        )

    def _smoke_gliner(
        self,
        splits: dict[str, list[SpanExample]],
    ) -> dict[str, list[SpanExample]]:
        rng = random.Random(self.config.seed)
        sliced: dict[str, list[SpanExample]] = {}
        for split_name, examples in splits.items():
            order = list(range(len(examples)))
            rng.shuffle(order)
            limit = SMOKE_TRAIN_ROWS if split_name == "train" else SMOKE_EVAL_ROWS
            sliced[split_name] = [examples[index] for index in order[:limit]]
        return sliced

    def _run_config(
        self,
        total_parameters: int,
        class_weights: tuple[float, float],
        selected_threshold: float,
    ) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "model_revision": self.config.model_revision,
            "gliner_dataset": GLINER_DATASET_ID,
            "gliner_revision": GLINER_REVISION,
            "canonical_dataset": CANONICAL_DATASET_ID,
            "canonical_revision": CANONICAL_REVISION,
            "total_parameters": total_parameters,
            "trainable_parameters": total_parameters,
            "full_fine_tuning": True,
            "loss": "float32 weighted token cross-entropy",
            "max_positive_class_weight": MAX_POSITIVE_CLASS_WEIGHT,
            "class_weights": {"O": class_weights[0], "EVIDENCE": class_weights[1]},
            "checkpoint_selection": "validation token-F1",
            "candidate_threshold_selection": "validation Word-F1",
            "max_length": MAX_LENGTH,
            "doc_stride": DOC_STRIDE,
            "selected_threshold": selected_threshold,
            "min_span_chars": MIN_SPAN_CHARS,
            "merge_gap_chars": MERGE_GAP_CHARS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "precision": "float32",
            "tracker": "tensorboard",
            "created_at": datetime.now(UTC).isoformat(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }

    @staticmethod
    def _package_versions() -> dict[str, str]:
        packages = (
            "accelerate",
            "datasets",
            "huggingface-hub",
            "numpy",
            "tensorboard",
            "torch",
            "transformers",
        )
        return {package: importlib.metadata.version(package) for package in packages}

    @staticmethod
    def _print_source_summary(
        manifest: dict[str, Any],
        example: SpanExample,
    ) -> None:
        print(
            f"Validated GLiNER rows {manifest['rows']} from "
            f"{manifest['parents']} parents."
        )
        print(
            json.dumps(
                {
                    "example_question": example.question,
                    "example_chunk_prefix": example.chunk[:160],
                    "example_spans": [asdict(span) for span in example.spans],
                    "paper_id": example.paper_id,
                    "chunk_index": example.chunk_index,
                },
                indent=2,
            )
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def tokenize(chunk: str) -> list[tuple[int, int]]:
    """Return whitespace-delimited word offsets. Upstream span_metrics.py."""

    return [(match.start(), match.end()) for match in WORD_RE.finditer(chunk)]


def spans_to_word_indices(
    words: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> set[int]:
    """Word indices overlapped by any span. Upstream span_metrics.py."""

    covered: set[int] = set()
    for span_start, span_end in spans:
        for index, (word_start, word_end) in enumerate(words):
            if word_end <= span_start or word_start >= span_end:
                continue
            covered.add(index)
    return covered


def official_line_set_prf(
    predicted: Sequence[str],
    reference: Sequence[str],
) -> tuple[float, float, float, float]:
    """Per-example line-set P/R/F1/EM. Upstream ``compute_span_metrics``.

    Both-empty sets score ``(1.0, 1.0, 1.0, 1.0)``. Exactly one empty set
    scores ``(0.0, 0.0, 0.0, 0.0)``.
    """

    pred_set = set(predicted)
    ref_set = set(reference)
    if not ref_set and not pred_set:
        return 1.0, 1.0, 1.0, 1.0
    if not ref_set or not pred_set:
        return 0.0, 0.0, 0.0, 0.0
    true_positive = len(pred_set & ref_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(ref_set)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    exact_match = 1.0 if pred_set == ref_set else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4), exact_match


def official_line_compression(original: str, filtered: str) -> float:
    """Return the upstream line compression ratio.

    Attribution: ``squeez.training.evaluate.compute_compression_ratio``.
    Python's ``"".split("\\n")`` contains one empty line, so an empty
    prediction still counts as one filtered line, exactly as upstream.
    """

    original_lines = len(original.split("\n"))
    filtered_lines = len(filtered.split("\n"))
    if original_lines == 0:
        return 0.0
    return round(1.0 - filtered_lines / original_lines, 4)


def prf(
    true_positive: int, false_positive: int, false_negative: int
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if intersection == 0:
        return 0.0
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / union if union else 0.0


def span_iou_prf(
    pred_spans: Sequence[tuple[int, int]],
    gold_spans: Sequence[tuple[int, int]],
    threshold: float,
) -> tuple[float, float, float, int, int, int]:
    """Greedy span IoU P/R/F1. Upstream span_metrics.py."""

    n_pred = len(pred_spans)
    n_gold = len(gold_spans)
    if n_pred == 0 and n_gold == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if n_pred == 0 or n_gold == 0:
        return 0.0, 0.0, 0.0, 0, n_pred, n_gold
    pairs = [
        (iou(pred, gold), pred_index, gold_index)
        for pred_index, pred in enumerate(pred_spans)
        for gold_index, gold in enumerate(gold_spans)
    ]
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    true_positive = 0
    for score, pred_index, gold_index in pairs:
        if score < threshold:
            break
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        true_positive += 1
    false_positive = n_pred - true_positive
    false_negative = n_gold - true_positive
    precision, recall, f1 = prf(true_positive, false_positive, false_negative)
    return precision, recall, f1, true_positive, false_positive, false_negative


def recall_any_overlap(
    pred_spans: Sequence[tuple[int, int]],
    gold_spans: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    hit = 0
    for gold in gold_spans:
        if any(iou(pred, gold) > 0 for pred in pred_spans):
            hit += 1
    return hit, len(gold_spans)


def decode_char_spans(
    scores: np.ndarray,
    threshold: float,
    min_span_chars: int,
    merge_gap_chars: int,
) -> list[tuple[int, int]]:
    """Merge characters above ``threshold`` into decoded spans.

    Input:
        Per-character evidence probabilities, for example
        ``np.array([0.1, 0.8, 0.9, 0.2])``.

    Output:
        Inclusive-start exclusive-end spans, for example ``[(1, 3)]``.
    """

    selected = scores >= threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, keep in enumerate(selected):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(selected)))

    merged: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        if merged and run_start - merged[-1][1] <= merge_gap_chars:
            merged[-1] = (merged[-1][0], run_end)
            continue
        merged.append((run_start, run_end))
    return [
        (run_start, run_end)
        for run_start, run_end in merged
        if run_end - run_start >= min_span_chars
    ]


def compute_token_metrics(eval_pred: Any) -> dict[str, float]:
    """Window-level token F1 used only for checkpoint selection."""

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    true = labels[mask]
    pred = predictions[mask]
    true_positive = int(((true == 1) & (pred == 1)).sum())
    false_positive = int(((true == 0) & (pred == 1)).sum())
    false_negative = int(((true == 1) & (pred == 0)).sum())
    precision, recall, f1 = prf(true_positive, false_positive, false_negative)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        choices=sorted(MODEL_REVISIONS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume Trainer state from an existing checkpoint.",
    )
    parser.add_argument("--smoke-run", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    output_dir = args.output_dir
    if (
        args.model_id == "jhu-clsp/ettin-encoder-32m"
        and output_dir == DEFAULT_OUTPUT_DIR
    ):
        output_dir = SUPPORTED_32M_OUTPUT_DIR
    return ExperimentConfig(
        model_id=args.model_id,
        output_dir=str(output_dir),
        epochs=args.epochs,
        seed=args.seed,
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
        smoke_run=args.smoke_run,
    )


def main() -> None:
    SpanPrunerExperiment(parse_args()).run()


if __name__ == "__main__":
    main()
