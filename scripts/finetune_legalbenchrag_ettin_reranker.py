#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.1",
#   "huggingface-hub==1.30.0",
#   "numpy==2.3.3",
#   "rank-bm25==0.2.2",
#   "sentence-transformers==6.0.1",
#   "tensorboard==2.21.0",
#   "torch==2.14.0",
#   "transformers==5.16.1",
# ]
# ///
"""Fine-tune a compact legal passage reranker on full LegalBench-RAG.

This tutorial script builds a useful second-stage reranker rather than a legal
answer generator. The complete experiment has one visible pipeline:

1. Download and validate all 6,889 LegalBench-RAG queries, all 714 source
   documents, and all four domains: ContractNLI, CUAD, MAUD, and PrivacyQA.
2. Split by source document so that near-duplicate questions from one contract
   cannot leak between training and evaluation.
3. Chunk documents into overlapping, token-bounded passages. Each annotated
   evidence span supplies a positive passage; BM25 supplies difficult passages
   from the same document as negatives and as organic evaluation candidates.
4. Measure BM25 and the untouched Ettin reranker, then fully fine-tune all 150M
   parameters with binary cross-entropy and select the best validation NDCG@10.
5. Save TensorBoard logs, configuration, prediction ranks, and per-domain
   metrics; generate the model card; publish; reload from the Hub; and verify
   the documented ``CrossEncoder.rank`` usage.

The default is sized for a free Colab T4. Runtime is initially estimated at
2-5 hours and is replaced by observed timing in the published model card.
Colab availability and session duration are controlled by Google, so run the
GPU smoke mode before committing the full session.

Run the complete full-data experiment and publish it:

    uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py

Validate the downloaded corpus and exact character spans without a GPU:

    uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py \
        --validate-only

Prepare every query, passage, hard negative, and organic candidate without
loading the model weights:

    uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py \
        --prepare-only

Run a small local-only GPU smoke test:

    uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py \
        --smoke-run --no-push-to-hub

Recover publication from a completed output directory without retraining:

    uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py \
        --publish-only

The model expects ``(query, candidate passage)`` pairs and emits one relevance
logit per pair. It does not search a corpus or provide legal advice. In a real
workflow, BM25 or an embedding model first retrieves perhaps 30-100 candidates;
this model then puts the most useful evidence passages first.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi, ModelCard
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.integrations import TensorBoardCallback

BASE_MODEL_ID = "cross-encoder/ettin-reranker-150m-v1"
BASE_MODEL_REVISION = "025501c4e0f9bbeb4c5b198318e0089ff061cc14"
MODEL_REPO_NAME = "LegalBenchRAG-Ettin-150M-Reranker"
UPSTREAM_REPOSITORY = "https://github.com/ZeroEntropy-AI/legalbenchrag"
DATA_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/"
    "AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1"
)
DATA_ARCHIVE_SHA256 = "27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d"
DATASET_RELEASE_DATE = "2025-11-01"
DATA_CACHE_DIR = Path.home() / ".cache" / "legalbenchrag" / DATA_ARCHIVE_SHA256[:12]
DEFAULT_OUTPUT_DIR = Path("legalbenchrag-ettin-150m-reranker")

DOMAINS = ("contractnli", "cuad", "maud", "privacy_qa")
EXPECTED_QUERY_COUNTS = {
    "contractnli": 977,
    "cuad": 4_042,
    "maud": 1_676,
    "privacy_qa": 194,
}
EXPECTED_DOCUMENT_COUNTS = {
    "contractnli": 95,
    "cuad": 462,
    "maud": 150,
    "privacy_qa": 7,
}
EXPECTED_EVIDENCE_SPANS = 10_928

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5.0
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
MAX_PAIR_TOKENS = 512
PASSAGE_TOKENS = 384
PASSAGE_OVERLAP_TOKENS = 96
HARD_NEGATIVES_PER_QUERY = 4
EVALUATION_CANDIDATES = 32
RANKING_AT_K = 10
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2.0e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
EARLY_STOPPING_PATIENCE = 2
MIN_RECOMMENDED_NDCG_GAIN = 0.02

SMOKE_DOCUMENTS_PER_DOMAIN = 3
SMOKE_QUERIES_PER_SPLIT = 16
SMOKE_EPOCHS = 0.03

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")
REQUIRED_LOCAL_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "README.md",
    "run_config.json",
    "dataset_manifest.json",
    "baseline_metrics.json",
    "final_metrics.json",
    "training_metrics.json",
    "trainer_state.json",
)
REQUIRED_REMOTE_ARTIFACTS = REQUIRED_LOCAL_ARTIFACTS + ("verification_results.json",)


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One verified gold excerpt within a source document."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class LegalQuery:
    """One retrieval question and its exact evidence coordinates."""

    query_id: str
    domain: str
    query: str
    document_path: str
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True, slots=True)
class Passage:
    """A token-bounded document passage with reversible character offsets."""

    passage_id: str
    document_path: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One query and its organic BM25 candidates, without injected positives."""

    query: LegalQuery
    candidates: tuple[Passage, ...]
    relevant_passage_ids: frozenset[str]


@dataclass(slots=True)
class PreparedExperiment:
    """In-memory training pairs and realistic validation/test candidate lists."""

    train_dataset: Dataset
    validation_dataset: Dataset
    validation_cases: list[EvaluationCase]
    test_cases: list[EvaluationCase]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable, serializable experiment inputs."""

    output_dir: str
    data_dir: str
    epochs: float
    seed: int
    publish: bool
    validate_only: bool
    prepare_only: bool
    smoke_run: bool
    publish_only: bool

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


class LegalBenchRAGSource:
    """Own downloading, extraction, parsing, and boundary validation."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def ensure_available(self) -> None:
        """Download and safely extract the pinned upstream dataset release."""

        if self._has_expected_layout():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.data_dir / "LegalBench-RAG.zip"
        if (
            not archive_path.is_file()
            or self._sha256(archive_path) != DATA_ARCHIVE_SHA256
        ):
            print(f"Downloading the full LegalBench-RAG archive to {archive_path}")
            self._download(archive_path)

        actual_sha256 = self._sha256(archive_path)
        if actual_sha256 != DATA_ARCHIVE_SHA256:
            raise ValueError(
                "LegalBench-RAG archive checksum changed. Expected "
                f"{DATA_ARCHIVE_SHA256}, received {actual_sha256}. Review the "
                "new upstream release before changing the pinned checksum."
            )
        self._safe_extract(archive_path)
        if not self._has_expected_layout():
            raise FileNotFoundError(
                f"The archive did not create corpus/ and benchmarks/ under {self.data_dir}."
            )

    def load_and_validate(
        self,
    ) -> tuple[list[LegalQuery], dict[str, str], dict[str, Any]]:
        """Load every source row and verify every referenced character span.

        Input:
            ``data_dir`` must contain the upstream ``corpus/`` text files and
            ``benchmarks/*.json`` files. Each benchmark row has this shape::

                {
                    "query": "Consider the NDA; may either party terminate?",
                    "snippets": [{
                        "file_path": "contractnli/example.txt",
                        "span": [120, 248],
                        "answer": "Either party may terminate ..."
                    }]
                }

        Returns:
            ``(queries, documents, manifest)``. ``queries`` contains typed
            ``LegalQuery`` objects, ``documents`` maps each relative file path
            to its complete text, and ``manifest`` records exact domain counts
            and validation results. The answer text is not retained because the
            verified character coordinates are the source of truth.
        """

        queries: list[LegalQuery] = []
        documents: dict[str, str] = {}
        query_counts: Counter[str] = Counter()
        evidence_count = 0

        for domain in DOMAINS:
            benchmark_path = self.data_dir / "benchmarks" / f"{domain}.json"
            payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            tests = payload.get("tests")
            if not isinstance(tests, list):
                raise TypeError(f"{benchmark_path} must contain a list named 'tests'.")

            for row_index, row in enumerate(tests):
                snippets = row.get("snippets")
                if not isinstance(snippets, list) or not snippets:
                    raise ValueError(f"{benchmark_path}:{row_index} has no snippets.")
                file_paths = {snippet["file_path"] for snippet in snippets}
                if len(file_paths) != 1:
                    raise ValueError(
                        f"{benchmark_path}:{row_index} references {len(file_paths)} documents."
                    )
                document_path = next(iter(file_paths))
                if not document_path.startswith(f"{domain}/"):
                    raise ValueError(
                        f"{benchmark_path}:{row_index} references unexpected path {document_path}."
                    )
                document = documents.get(document_path)
                if document is None:
                    source_path = self.data_dir / "corpus" / document_path
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"Missing source document: {source_path}"
                        )
                    document = source_path.read_text(encoding="utf-8")
                    documents[document_path] = document

                evidence: list[EvidenceSpan] = []
                for snippet in snippets:
                    start, end = snippet["span"]
                    answer = snippet["answer"]
                    if not (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start < end
                    ):
                        raise ValueError(
                            f"Invalid span {snippet['span']} in {benchmark_path}:{row_index}."
                        )
                    if document[start:end] != answer:
                        raise ValueError(
                            f"Evidence text does not match {document_path}:{start}-{end}."
                        )
                    evidence.append(EvidenceSpan(start=start, end=end))

                query_id = f"{domain}:{row_index:04d}"
                queries.append(
                    LegalQuery(
                        query_id=query_id,
                        domain=domain,
                        query=row["query"].strip(),
                        document_path=document_path,
                        evidence=tuple(evidence),
                    )
                )
                query_counts[domain] += 1
                evidence_count += len(evidence)

        document_counts = Counter(path.split("/", 1)[0] for path in documents)
        if dict(query_counts) != EXPECTED_QUERY_COUNTS:
            raise ValueError(
                f"Query counts changed: expected {EXPECTED_QUERY_COUNTS}, got {dict(query_counts)}."
            )
        if dict(document_counts) != EXPECTED_DOCUMENT_COUNTS:
            raise ValueError(
                "Document counts changed: expected "
                f"{EXPECTED_DOCUMENT_COUNTS}, got {dict(document_counts)}."
            )
        if evidence_count != EXPECTED_EVIDENCE_SPANS:
            raise ValueError(
                f"Expected {EXPECTED_EVIDENCE_SPANS} evidence spans, got {evidence_count}."
            )

        manifest = {
            "source_repository": UPSTREAM_REPOSITORY,
            "archive_sha256": DATA_ARCHIVE_SHA256,
            "release_date": DATASET_RELEASE_DATE,
            "domains": list(DOMAINS),
            "query_counts": dict(query_counts),
            "document_counts": dict(document_counts),
            "total_queries": len(queries),
            "total_documents": len(documents),
            "evidence_spans": evidence_count,
            "missing_documents": 0,
            "span_mismatches": 0,
        }
        return queries, documents, manifest

    def _has_expected_layout(self) -> bool:
        return all(
            (self.data_dir / "benchmarks" / f"{domain}.json").is_file()
            and (self.data_dir / "corpus" / domain).is_dir()
            for domain in DOMAINS
        )

    @staticmethod
    def _download(destination: Path) -> None:
        temporary = destination.with_suffix(".download")
        request = urllib.request.Request(
            DATA_ARCHIVE_URL,
            headers={"User-Agent": "legalbenchrag-training-script/1.0"},
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)

    def _safe_extract(self, archive_path: Path) -> None:
        root = self.data_dir.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if not relative.parts or relative.is_absolute():
                    continue
                if relative.parts[0] not in {"corpus", "benchmarks"}:
                    continue
                destination = (self.data_dir / Path(*relative.parts)).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class DocumentGroupSplitter:
    """Create deterministic, domain-stratified, document-disjoint splits."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def split(self, queries: Sequence[LegalQuery]) -> dict[str, list[LegalQuery]]:
        """Assign every query according to the source document.

        Input:
            A sequence of all 6,889 typed queries. Multiple questions commonly
            point to the same contract.

        Returns:
            ``{"train": [...], "validation": [...], "test": [...]}``. Every
            query appears once, every domain contributes documents to every
            split, and no ``document_path`` occurs in more than one split.
        """

        by_domain_document: dict[str, dict[str, list[LegalQuery]]] = {
            domain: defaultdict(list) for domain in DOMAINS
        }
        for query in queries:
            by_domain_document[query.domain][query.document_path].append(query)

        result = {"train": [], "validation": [], "test": []}
        for domain in DOMAINS:
            documents = list(by_domain_document[domain])
            documents.sort(key=self._stable_document_key)
            validation_count = max(1, round(len(documents) * DEFAULT_VALIDATION_RATIO))
            test_count = max(
                1,
                round(
                    len(documents)
                    * (1.0 - DEFAULT_TRAIN_RATIO - DEFAULT_VALIDATION_RATIO)
                ),
            )
            train_count = len(documents) - validation_count - test_count
            if train_count < 1:
                raise ValueError(
                    f"Not enough {domain} documents for three disjoint splits."
                )

            assignments = {
                "train": documents[:train_count],
                "validation": documents[train_count : train_count + validation_count],
                "test": documents[train_count + validation_count :],
            }
            for split_name, split_documents in assignments.items():
                for document_path in split_documents:
                    result[split_name].extend(by_domain_document[domain][document_path])

        self._validate(result, expected_queries=len(queries))
        for split_queries in result.values():
            split_queries.sort(key=lambda query: query.query_id)
        return result

    def _stable_document_key(self, document_path: str) -> str:
        return hashlib.sha256(f"{self.seed}:{document_path}".encode()).hexdigest()

    @staticmethod
    def _validate(splits: dict[str, list[LegalQuery]], expected_queries: int) -> None:
        if sum(map(len, splits.values())) != expected_queries:
            raise ValueError("Document splitting dropped or duplicated queries.")
        document_sets = {
            name: {query.document_path for query in split_queries}
            for name, split_queries in splits.items()
        }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = document_sets[left].intersection(document_sets[right])
            if overlap:
                raise ValueError(
                    f"Document leakage between {left} and {right}: {sorted(overlap)[:3]}"
                )


class TokenPassageChunker:
    """Build overlapping legal passages while retaining character coordinates."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if not tokenizer.is_fast:
            raise ValueError(
                "A fast tokenizer is required for character offset mapping."
            )
        self.tokenizer = tokenizer

    def chunk(self, document_path: str, text: str) -> list[Passage]:
        """Convert one document into section-aware token windows.

        Args:
            document_path: Relative corpus path such as ``cuad/example.txt``.
            text: Complete source text whose character coordinates match the
                LegalBench-RAG evidence annotations.

        Returns:
            Ordered passages shaped like::

                Passage(
                    passage_id="cuad/example.txt:0000120-0001840",
                    document_path="cuad/example.txt",
                    start=120,
                    end=1840,
                    text="... exact source substring ...",
                )

            Each passage has at most ``PASSAGE_TOKENS`` model tokens before the
            query is added. Adjacent passages overlap, and a nearby blank-line
            boundary is preferred when it does not make the passage too short.
        """

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = [
            tuple(offset)
            for offset in encoded["offset_mapping"]
            if offset[1] > offset[0]
        ]
        if not offsets:
            return []
        token_starts = [start for start, _ in offsets]
        passages: list[Passage] = []
        start_token = 0

        while start_token < len(offsets):
            end_token = min(start_token + PASSAGE_TOKENS, len(offsets))
            if end_token < len(offsets):
                minimum_end_token = start_token + PASSAGE_TOKENS // 2
                minimum_end_char = offsets[minimum_end_token][0]
                nominal_end_char = offsets[end_token - 1][1]
                boundary = self._last_paragraph_boundary(
                    text,
                    minimum_end_char,
                    nominal_end_char,
                )
                if boundary is not None:
                    aligned_end = bisect.bisect_right(token_starts, boundary)
                    if aligned_end > minimum_end_token:
                        end_token = aligned_end

            start_char = offsets[start_token][0]
            end_char = offsets[end_token - 1][1]
            passage_text = text[start_char:end_char].strip()
            if passage_text:
                stripped_start = text.find(passage_text, start_char, end_char + 1)
                stripped_end = stripped_start + len(passage_text)
                passage_id = f"{document_path}:{stripped_start:07d}-{stripped_end:07d}"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        document_path=document_path,
                        start=stripped_start,
                        end=stripped_end,
                        text=passage_text,
                    )
                )

            if end_token == len(offsets):
                break
            next_start = max(end_token - PASSAGE_OVERLAP_TOKENS, start_token + 1)
            start_token = next_start

        return passages

    @staticmethod
    def _last_paragraph_boundary(text: str, start: int, end: int) -> int | None:
        boundaries = list(PARAGRAPH_BREAK_PATTERN.finditer(text, start, end))
        return boundaries[-1].end() if boundaries else None


class BM25DocumentIndex:
    """Rank one document's passages for all questions about that document."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages = passages
        self.index = BM25Okapi([self.tokenize(passage.text) for passage in passages])

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def rank(self, query: str) -> list[int]:
        scores = self.index.get_scores(self.tokenize(query))
        return sorted(
            range(len(self.passages)),
            key=lambda index: (-float(scores[index]), index),
        )


class LegalBenchRAGPreparer:
    """Transform source documents and spans into reranker supervision."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seed: int,
        smoke_run: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.seed = seed
        self.smoke_run = smoke_run
        self.chunker = TokenPassageChunker(tokenizer)

    def prepare(
        self,
        queries: Sequence[LegalQuery],
        documents: dict[str, str],
        source_manifest: dict[str, Any],
    ) -> PreparedExperiment:
        """Build labeled training pairs and organic evaluation candidate lists.

        Input:
            Validated ``LegalQuery`` objects and the complete document mapping
            returned by ``LegalBenchRAGSource.load_and_validate``.

        Output:
            ``PreparedExperiment`` containing:

            * pair datasets with ``query``, ``passage``, and binary ``label``;
            * validation/test cases containing BM25's top 32 passages in their
              original order, without injecting missing positives; and
            * a manifest describing split, chunk, class, and truncation counts.

            A representative positive pair is::

                {
                    "query": "What law governs this agreement?",
                    "passage": "This Agreement is governed by New York law.",
                    "label": 1.0,
                }

            A hard-negative row uses the same question with a top-ranked but
            non-overlapping passage and ``label=0.0``.
        """

        splits = DocumentGroupSplitter(self.seed).split(queries)
        if self.smoke_run:
            splits = self._smoke_splits(splits)
        split_by_query_id = {
            query.query_id: split_name
            for split_name, split_queries in splits.items()
            for query in split_queries
        }
        selected_queries = [
            query for query in queries if query.query_id in split_by_query_id
        ]
        queries_by_document: dict[str, list[LegalQuery]] = defaultdict(list)
        for query in selected_queries:
            queries_by_document[query.document_path].append(query)

        pair_rows = {"train": [], "validation": []}
        evaluation_cases = {"validation": [], "test": []}
        passage_count = 0
        positive_pairs = Counter()
        negative_pairs = Counter()
        organic_candidate_hits = Counter()
        queries_with_fewer_hard_negatives = Counter()
        max_pair_tokens = 0

        for document_index, document_path in enumerate(
            sorted(queries_by_document), start=1
        ):
            passages = self.chunker.chunk(document_path, documents[document_path])
            if not passages:
                raise ValueError(
                    f"Tokenization produced no passages for {document_path}."
                )
            passage_count += len(passages)
            document_indexer = BM25DocumentIndex(passages)

            for query in queries_by_document[document_path]:
                split_name = split_by_query_id[query.query_id]
                relevant = self._relevant_passages(query, passages)
                positives = self._one_best_passage_per_span(query, passages)
                ranking = document_indexer.rank(query.query)
                negative_indices = [
                    index
                    for index in ranking
                    if passages[index].passage_id not in relevant
                ][:HARD_NEGATIVES_PER_QUERY]
                if len(negative_indices) < HARD_NEGATIVES_PER_QUERY:
                    queries_with_fewer_hard_negatives[split_name] += 1

                if split_name in pair_rows:
                    for passage in positives:
                        pair_rows[split_name].append(
                            {
                                "query": query.query,
                                "passage": passage.text,
                                "label": 1.0,
                            }
                        )
                        positive_pairs[split_name] += 1
                    for index in negative_indices:
                        pair_rows[split_name].append(
                            {
                                "query": query.query,
                                "passage": passages[index].text,
                                "label": 0.0,
                            }
                        )
                        negative_pairs[split_name] += 1

                if split_name in evaluation_cases:
                    candidates = tuple(
                        passages[index] for index in ranking[:EVALUATION_CANDIDATES]
                    )
                    if any(passage.passage_id in relevant for passage in candidates):
                        organic_candidate_hits[split_name] += 1
                    evaluation_cases[split_name].append(
                        EvaluationCase(
                            query=query,
                            candidates=candidates,
                            relevant_passage_ids=frozenset(relevant),
                        )
                    )

                for passage in (
                    *positives,
                    *(passages[index] for index in negative_indices),
                ):
                    pair_length = len(
                        self.tokenizer(
                            query.query,
                            passage.text,
                            add_special_tokens=True,
                            truncation=False,
                        )["input_ids"]
                    )
                    max_pair_tokens = max(max_pair_tokens, pair_length)
                    if pair_length > MAX_PAIR_TOKENS:
                        raise ValueError(
                            f"{query.query_id} creates a {pair_length}-token pair; "
                            f"the limit is {MAX_PAIR_TOKENS}."
                        )

            if document_index % 100 == 0:
                print(
                    f"Prepared {document_index}/{len(queries_by_document)} documents "
                    f"({passage_count:,} passages)"
                )

        rng = random.Random(self.seed)
        rng.shuffle(pair_rows["train"])
        rng.shuffle(pair_rows["validation"])
        split_manifest = self._split_manifest(splits)
        manifest = {
            **source_manifest,
            "smoke_run": self.smoke_run,
            "split": split_manifest,
            "passages": passage_count,
            "passage_tokens": PASSAGE_TOKENS,
            "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "max_observed_pair_tokens": max_pair_tokens,
            "max_pair_tokens": MAX_PAIR_TOKENS,
            "hard_negatives_per_query": HARD_NEGATIVES_PER_QUERY,
            "evaluation_candidates": EVALUATION_CANDIDATES,
            "positive_pairs": dict(positive_pairs),
            "negative_pairs": dict(negative_pairs),
            "organic_candidate_query_hits": dict(organic_candidate_hits),
            "queries_with_fewer_hard_negatives": dict(
                queries_with_fewer_hard_negatives
            ),
        }
        return PreparedExperiment(
            train_dataset=Dataset.from_list(pair_rows["train"]),
            validation_dataset=Dataset.from_list(pair_rows["validation"]),
            validation_cases=evaluation_cases["validation"],
            test_cases=evaluation_cases["test"],
            manifest=manifest,
        )

    @staticmethod
    def _overlap(
        left_start: int, left_end: int, right_start: int, right_end: int
    ) -> int:
        return max(0, min(left_end, right_end) - max(left_start, right_start))

    def _relevant_passages(
        self,
        query: LegalQuery,
        passages: Sequence[Passage],
    ) -> set[str]:
        return {
            passage.passage_id
            for passage in passages
            if any(
                self._overlap(passage.start, passage.end, span.start, span.end) > 0
                for span in query.evidence
            )
        }

    def _one_best_passage_per_span(
        self,
        query: LegalQuery,
        passages: Sequence[Passage],
    ) -> list[Passage]:
        selected: dict[str, Passage] = {}
        for span in query.evidence:
            best = max(
                passages,
                key=lambda passage: self._overlap(
                    passage.start,
                    passage.end,
                    span.start,
                    span.end,
                ),
            )
            if self._overlap(best.start, best.end, span.start, span.end) == 0:
                raise ValueError(
                    f"No passage covers {query.query_id}:{span.start}-{span.end}."
                )
            selected[best.passage_id] = best
        return list(selected.values())

    @staticmethod
    def _split_manifest(splits: dict[str, list[LegalQuery]]) -> dict[str, Any]:
        return {
            split_name: {
                "queries": len(split_queries),
                "documents": len({query.document_path for query in split_queries}),
                "queries_by_domain": dict(
                    Counter(query.domain for query in split_queries)
                ),
                "documents_by_domain": dict(
                    Counter(
                        document.split("/", 1)[0]
                        for document in {query.document_path for query in split_queries}
                    )
                ),
            }
            for split_name, split_queries in splits.items()
        }

    @staticmethod
    def _smoke_splits(
        splits: dict[str, list[LegalQuery]],
    ) -> dict[str, list[LegalQuery]]:
        smoke: dict[str, list[LegalQuery]] = {}
        for split_name, queries in splits.items():
            selected: list[LegalQuery] = []
            per_domain_limit = 16 if split_name == "train" else 4
            for domain in DOMAINS:
                domain_queries = [query for query in queries if query.domain == domain]
                documents = sorted({query.document_path for query in domain_queries})[
                    :SMOKE_DOCUMENTS_PER_DOMAIN
                ]
                domain_selected = [
                    query
                    for query in domain_queries
                    if query.document_path in documents
                ][:per_domain_limit]
                selected.extend(domain_selected)
            expected_limit = SMOKE_QUERIES_PER_SPLIT if split_name != "train" else 64
            smoke[split_name] = selected[:expected_limit]
        return smoke


class RankingMetrics:
    """Evaluate organic retrieval candidates without appending gold passages."""

    def evaluate_bm25(self, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
        rankings = [list(range(len(case.candidates))) for case in cases]
        return self._aggregate(cases, rankings, scores=None)

    def evaluate_model(
        self,
        model: CrossEncoder,
        cases: Sequence[EvaluationCase],
        batch_size: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        pairs: list[tuple[str, str]] = []
        lengths: list[int] = []
        for case in cases:
            pairs.extend(
                (case.query.query, passage.text) for passage in case.candidates
            )
            lengths.append(len(case.candidates))
        predicted = model.predict(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        rankings: list[list[int]] = []
        grouped_scores: list[list[float]] = []
        offset = 0
        for length in lengths:
            scores = [float(value) for value in predicted[offset : offset + length]]
            rankings.append(
                sorted(range(length), key=lambda index: (-scores[index], index))
            )
            grouped_scores.append(scores)
            offset += length
        metrics = self._aggregate(cases, rankings, grouped_scores)
        predictions = self._prediction_records(cases, rankings, grouped_scores)
        return metrics, predictions

    def _aggregate(
        self,
        cases: Sequence[EvaluationCase],
        rankings: Sequence[Sequence[int]],
        scores: Sequence[Sequence[float]] | None,
    ) -> dict[str, Any]:
        per_query: list[dict[str, float | str]] = []
        for case_index, (case, ranking) in enumerate(zip(cases, rankings, strict=True)):
            ordered = [case.candidates[index] for index in ranking]
            relevance = [
                passage.passage_id in case.relevant_passage_ids for passage in ordered
            ]
            reciprocal_rank = next(
                (
                    1.0 / rank
                    for rank, relevant in enumerate(relevance[:RANKING_AT_K], start=1)
                    if relevant
                ),
                0.0,
            )
            relevant_total = len(case.relevant_passage_ids)
            ideal_relevant = min(relevant_total, RANKING_AT_K)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, relevant in enumerate(relevance[:RANKING_AT_K], start=1)
                if relevant
            )
            idcg = sum(
                1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1)
            )
            char_precision, char_recall = self._character_metrics(
                case.query.evidence,
                ordered[:5],
            )
            row: dict[str, float | str] = {
                "query_id": case.query.query_id,
                "domain": case.query.domain,
                "hit@1": float(any(relevance[:1])),
                "hit@5": float(any(relevance[:5])),
                "hit@10": float(any(relevance[:10])),
                "mrr@10": reciprocal_rank,
                "ndcg@10": dcg / idcg if idcg else 0.0,
                "char_precision@5": char_precision,
                "char_recall@5": char_recall,
            }
            if scores is not None:
                row["top_score"] = scores[case_index][ranking[0]]
            per_query.append(row)

        metric_names = (
            "hit@1",
            "hit@5",
            "hit@10",
            "mrr@10",
            "ndcg@10",
            "char_precision@5",
            "char_recall@5",
        )
        overall = {
            metric: float(np.mean([float(row[metric]) for row in per_query]))
            for metric in metric_names
        }
        by_domain = {
            domain: {
                metric: float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in per_query
                            if row["domain"] == domain
                        ]
                    )
                )
                for metric in metric_names
            }
            for domain in DOMAINS
            if any(row["domain"] == domain for row in per_query)
        }
        return {"queries": len(cases), "overall": overall, "by_domain": by_domain}

    def _character_metrics(
        self,
        evidence: Sequence[EvidenceSpan],
        passages: Sequence[Passage],
    ) -> tuple[float, float]:
        gold = self._merge_intervals((span.start, span.end) for span in evidence)
        retrieved = self._merge_intervals(
            (passage.start, passage.end) for passage in passages
        )
        overlap = sum(
            max(0, min(gold_end, found_end) - max(gold_start, found_start))
            for gold_start, gold_end in gold
            for found_start, found_end in retrieved
        )
        gold_length = sum(end - start for start, end in gold)
        retrieved_length = sum(end - start for start, end in retrieved)
        precision = overlap / retrieved_length if retrieved_length else 0.0
        recall = overlap / gold_length if gold_length else 0.0
        return precision, recall

    @staticmethod
    def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
        return merged

    @staticmethod
    def _prediction_records(
        cases: Sequence[EvaluationCase],
        rankings: Sequence[Sequence[int]],
        scores: Sequence[Sequence[float]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for case, ranking, case_scores in zip(cases, rankings, scores, strict=True):
            records.append(
                {
                    "query_id": case.query.query_id,
                    "domain": case.query.domain,
                    "document_path": case.query.document_path,
                    "top_candidates": [
                        {
                            "passage_id": case.candidates[index].passage_id,
                            "start": case.candidates[index].start,
                            "end": case.candidates[index].end,
                            "score": case_scores[index],
                            "relevant": (
                                case.candidates[index].passage_id
                                in case.relevant_passage_ids
                            ),
                        }
                        for index in ranking[:RANKING_AT_K]
                    ],
                }
            )
        return records


class LegalRerankerExperiment:
    """Own model preparation, baseline, full fine-tuning, and evaluation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.metrics = RankingMetrics()

    def run(self) -> None:
        set_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.publish_only:
            HubPublisher(self.config).publish_and_verify()
            return

        source = LegalBenchRAGSource(self.config.data_path)
        source.ensure_available()
        queries, documents, source_manifest = source.load_and_validate()
        self._print_source_summary(source_manifest, queries[0])
        if self.config.validate_only:
            self._write_json(output_dir / "dataset_manifest.json", source_manifest)
            return

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
        )
        prepared = LegalBenchRAGPreparer(
            tokenizer=tokenizer,
            seed=self.config.seed,
            smoke_run=self.config.smoke_run,
        ).prepare(queries, documents, source_manifest)
        self._write_json(output_dir / "dataset_manifest.json", prepared.manifest)
        print(json.dumps(prepared.manifest["split"], indent=2))
        if self.config.prepare_only:
            return

        self._require_training_device()
        model = self._load_model()
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
            f"{len(prepared.train_dataset):,} labeled pairs."
        )

        bm25_metrics = self.metrics.evaluate_bm25(prepared.test_cases)
        base_metrics, _ = self.metrics.evaluate_model(
            model,
            prepared.test_cases,
            batch_size=EVAL_BATCH_SIZE,
        )
        baseline_metrics = {"bm25": bm25_metrics, "untouched_base_model": base_metrics}
        self._write_json(output_dir / "baseline_metrics.json", baseline_metrics)

        validation_evaluator = self._sentence_transformers_evaluator(
            prepared.validation_cases,
            name="legalbenchrag-dev",
        )
        validation_evaluator(model, output_path=str(output_dir / "baseline-eval"))
        training_args = self._training_arguments(validation_evaluator)
        trainer = CrossEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=prepared.train_dataset,
            eval_dataset=prepared.validation_dataset,
            loss=BinaryCrossEntropyLoss(model),
            evaluator=validation_evaluator,
            callbacks=[
                TensorBoardCallback(
                    tb_writer=SummaryWriter(
                        log_dir=str(self.config.output_path / "tensorboard")
                    )
                ),
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                ),
            ],
        )

        started_at = time.perf_counter()
        train_result = trainer.train()
        runtime_seconds = time.perf_counter() - started_at
        trainer.save_state()
        model.save_pretrained(output_dir)
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

        final_metrics, predictions = self.metrics.evaluate_model(
            model,
            prepared.test_cases,
            batch_size=EVAL_BATCH_SIZE,
        )
        comparison = self._comparison(bm25_metrics, base_metrics, final_metrics)
        training_metrics = {
            "train_result": train_result.metrics,
            "runtime_seconds": runtime_seconds,
            "epochs_requested": self.config.epochs,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
        }
        self._write_json(output_dir / "training_metrics.json", training_metrics)
        self._write_json(output_dir / "final_metrics.json", comparison)
        self._write_jsonl(output_dir / "test_predictions.jsonl", predictions)
        self._write_json(
            output_dir / "run_config.json", self._run_config(total_parameters)
        )
        self._write_json(output_dir / "package_versions.json", self._package_versions())

        ModelCardWriter(self.config).write()
        if self.config.publish:
            del trainer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            HubPublisher(self.config).publish_and_verify()

    def _load_model(self) -> CrossEncoder:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return CrossEncoder(
            BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            max_length=MAX_PAIR_TOKENS,
            model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
        )

    def _training_arguments(
        self,
        evaluator: CrossEncoderRerankingEvaluator,
    ) -> CrossEncoderTrainingArguments:
        epochs = SMOKE_EPOCHS if self.config.smoke_run else self.config.epochs
        return CrossEncoderTrainingArguments(
            output_dir=str(self.config.output_path / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            fp16=torch.cuda.is_available(),
            bf16=False,
            gradient_checkpointing=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=1 if self.config.smoke_run else LOGGING_STEPS,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_{evaluator.primary_metric}",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config.seed,
            data_seed=self.config.seed,
            disable_tqdm=False,
        )

    @staticmethod
    def _sentence_transformers_evaluator(
        cases: Sequence[EvaluationCase],
        name: str,
    ) -> CrossEncoderRerankingEvaluator:
        samples = [
            {
                "query": case.query.query,
                "positive": [
                    passage.text
                    for passage in case.candidates
                    if passage.passage_id in case.relevant_passage_ids
                ],
                "documents": [passage.text for passage in case.candidates],
            }
            for case in cases
        ]
        return CrossEncoderRerankingEvaluator(
            samples=samples,
            at_k=RANKING_AT_K,
            always_rerank_positives=False,
            name=name,
            batch_size=EVAL_BATCH_SIZE,
            show_progress_bar=True,
        )

    def _require_training_device(self) -> None:
        if not torch.cuda.is_available() and not self.config.smoke_run:
            raise RuntimeError(
                "The full experiment requires a CUDA GPU. Use --validate-only or "
                "--prepare-only on CPU, or run the default command on a Colab T4."
            )

    @staticmethod
    def _comparison(
        bm25: dict[str, Any],
        base: dict[str, Any],
        final: dict[str, Any],
    ) -> dict[str, Any]:
        ndcg_gain = final["overall"]["ndcg@10"] - base["overall"]["ndcg@10"]
        domain_regressions = {
            domain: final["by_domain"][domain]["ndcg@10"]
            - base["by_domain"][domain]["ndcg@10"]
            for domain in final["by_domain"]
        }
        recommended = ndcg_gain >= MIN_RECOMMENDED_NDCG_GAIN and all(
            delta >= -MIN_RECOMMENDED_NDCG_GAIN for delta in domain_regressions.values()
        )
        return {
            "bm25": bm25,
            "untouched_base_model": base,
            "fine_tuned_model": final,
            "fine_tuned_minus_base_ndcg@10": ndcg_gain,
            "per_domain_ndcg@10_delta": domain_regressions,
            "recommendation_gate": {
                "minimum_overall_ndcg@10_gain": MIN_RECOMMENDED_NDCG_GAIN,
                "maximum_allowed_domain_regression": MIN_RECOMMENDED_NDCG_GAIN,
                "passed": recommended,
            },
        }

    def _run_config(self, total_parameters: int) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "base_model": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "total_parameters": total_parameters,
            "trainable_parameters": total_parameters,
            "full_fine_tuning": True,
            "loss": "BinaryCrossEntropyLoss (BCEWithLogits)",
            "max_pair_tokens": MAX_PAIR_TOKENS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "precision": "float16" if torch.cuda.is_available() else "float32",
            "tracker": "tensorboard",
            "created_at": datetime.now(UTC).isoformat(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }

    @staticmethod
    def _package_versions() -> dict[str, str]:
        packages = (
            "accelerate",
            "datasets",
            "huggingface-hub",
            "numpy",
            "rank-bm25",
            "sentence-transformers",
            "tensorboard",
            "torch",
            "transformers",
        )
        return {package: importlib.metadata.version(package) for package in packages}

    @staticmethod
    def _print_source_summary(manifest: dict[str, Any], example: LegalQuery) -> None:
        print(
            f"Validated {manifest['total_queries']:,} queries, "
            f"{manifest['total_documents']:,} documents, and "
            f"{manifest['evidence_spans']:,} exact evidence spans."
        )
        print(
            json.dumps(
                {
                    "example_query": example.query,
                    "document_path": example.document_path,
                    "evidence_spans": [asdict(span) for span in example.evidence],
                },
                indent=2,
            )
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


class ModelCardWriter:
    """Render the public usage and evidence contract from saved run artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def write(self) -> Path:
        output_dir = self.config.output_path
        manifest = self._read_json(output_dir / "dataset_manifest.json")
        run_config = self._read_json(output_dir / "run_config.json")
        metrics = self._read_json(output_dir / "final_metrics.json")
        training = self._read_json(output_dir / "training_metrics.json")
        packages = self._read_json(output_dir / "package_versions.json")
        verification_path = output_dir / "verification_results.json"
        verification = (
            self._read_json(verification_path) if verification_path.is_file() else None
        )
        repository_id = self._repository_id()
        card = self._render(
            repository_id=repository_id,
            manifest=manifest,
            run_config=run_config,
            metrics=metrics,
            training=training,
            packages=packages,
            verification=verification,
        )
        ModelCard(card)
        destination = output_dir / "README.md"
        destination.write_text(card, encoding="utf-8")
        return destination

    def _render(
        self,
        repository_id: str,
        manifest: dict[str, Any],
        run_config: dict[str, Any],
        metrics: dict[str, Any],
        training: dict[str, Any],
        packages: dict[str, str],
        verification: dict[str, Any] | None,
    ) -> str:
        final = metrics["fine_tuned_model"]["overall"]
        base = metrics["untouched_base_model"]["overall"]
        bm25 = metrics["bm25"]["overall"]
        gate = metrics["recommendation_gate"]
        status_text = (
            "passed the pre-registered usefulness gate"
            if gate["passed"]
            else "did not pass the pre-registered usefulness gate"
        )
        domain_rows = "\n".join(
            "| {domain} | {base:.4f} | {final:.4f} | {delta:+.4f} |".format(
                domain=domain,
                base=metrics["untouched_base_model"]["by_domain"][domain]["ndcg@10"],
                final=metrics["fine_tuned_model"]["by_domain"][domain]["ndcg@10"],
                delta=metrics["per_domain_ndcg@10_delta"][domain],
            )
            for domain in DOMAINS
        )
        package_rows = "\n".join(
            f"| `{name}` | `{version}` |" for name, version in packages.items()
        )
        verification_section = self._verification_section(verification)
        return f"""---
base_model: {BASE_MODEL_ID}
base_model_relation: finetune
library_name: sentence-transformers
pipeline_tag: text-ranking
license: apache-2.0
language:
- en
tags:
- legal
- retrieval
- rag
- reranker
- cross-encoder
- modernbert
- legalbenchrag
---

# LegalBench-RAG Ettin 150M Reranker

This is a fully fine-tuned 150M-parameter cross-encoder for ranking evidence
passages from English contracts and privacy policies. It starts from
[`{BASE_MODEL_ID}`](https://huggingface.co/{BASE_MODEL_ID}) and trains on the
complete LegalBench-RAG release rather than the 776-query mini benchmark.

On the document-held-out organic BM25 candidate test, this run {status_text}.
The gate required at least +{MIN_RECOMMENDED_NDCG_GAIN:.2f} NDCG@10 over the
untouched base model without a domain regression worse than
-{MIN_RECOMMENDED_NDCG_GAIN:.2f}. Treat the table below as the capability
claim; training loss alone is not evidence that retrieval improved.

## Intended use

Use this model as the second stage of a retrieve-and-rerank workflow:

1. Split or index legal documents.
2. Use BM25 or embeddings to retrieve 30-100 candidate passages.
3. Pass the query and candidates to this model.
4. Send the highest-ranked evidence to a human reviewer or grounded generator.

The model emits relevance scores. It does not search a corpus, answer legal
questions, execute actions, or provide legal advice.

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{repository_id}")
query = "When may either party terminate the agreement?"
passages = [
    "Either party may terminate this Agreement with thirty days written notice.",
    "Confidential information must be protected for five years.",
    "Invoices are payable within sixty days after receipt.",
]

ranked = model.rank(query, passages, return_documents=True)
for result in ranked:
    print(float(result["score"]), result["text"])
```

The output is a list ordered from the highest relevance logit to the lowest.
Scores rank candidates for one query; they are not calibrated probabilities.

{verification_section}

## Data and leakage controls

The pinned upstream [LegalBench-RAG release]({UPSTREAM_REPOSITORY}) contains
{manifest["total_queries"]:,} questions, {manifest["total_documents"]:,}
documents, and {manifest["evidence_spans"]:,} expert-annotated evidence spans.
All spans were checked against the exact source substrings before training.

| Domain | Queries | Documents |
|---|---:|---:|
| ContractNLI | {manifest["query_counts"]["contractnli"]:,} | {manifest["document_counts"]["contractnli"]:,} |
| CUAD | {manifest["query_counts"]["cuad"]:,} | {manifest["document_counts"]["cuad"]:,} |
| MAUD | {manifest["query_counts"]["maud"]:,} | {manifest["document_counts"]["maud"]:,} |
| PrivacyQA | {manifest["query_counts"]["privacy_qa"]:,} | {manifest["document_counts"]["privacy_qa"]:,} |

Documents—not individual questions—were assigned to train, validation, and
test. Therefore questions about one contract cannot appear across splits. The
published repository does not redistribute source contracts or passage text.

## Schema transformation

One upstream row contains a query and one or more exact evidence coordinates:

```json
{{
  "query": "What law governs this agreement?",
  "snippets": [{{
    "file_path": "cuad/example.txt",
    "span": [120, 184],
    "answer": "This Agreement is governed by New York law."
  }}]
}}
```

Documents were converted into overlapping token-bounded passages. Each evidence
span selected its highest-overlap passage as a positive. BM25 supplied four
high-ranking non-overlapping passages from the same document as hard negatives:

```json
{{
  "query": "What law governs this agreement?",
  "passage": "This Agreement is governed by New York law.",
  "label": 1.0
}}
```

Evaluation reranks BM25's organic top-{EVALUATION_CANDIDATES}; missing positives
are not inserted into the candidate list.

## Model and objective

This is a standalone cross-encoder, not an adapter. A query and passage attend
to each other jointly and the model emits one logit. Binary cross-entropy with
logits trains positives toward 1 and hard negatives toward 0. Every labeled
pair contributes to the loss; float16 changes memory use and TensorBoard records
progress, but neither changes the objective.

| Configuration | Value |
|---|---|
| Base revision | `{BASE_MODEL_REVISION}` |
| Parameters | {run_config["total_parameters"]:,} trainable / {run_config["total_parameters"]:,} total |
| Epochs requested | {training["epochs_requested"]} |
| Best checkpoint | `{training["best_checkpoint"]}` |
| Pair length | {MAX_PAIR_TOKENS} tokens |
| Passage window / overlap | {PASSAGE_TOKENS} / {PASSAGE_OVERLAP_TOKENS} tokens |
| Batch / accumulation / effective batch | {TRAIN_BATCH_SIZE} / {GRADIENT_ACCUMULATION_STEPS} / {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} |
| Optimizer / schedule | AdamW / linear |
| Learning rate / warmup | {LEARNING_RATE} / {WARMUP_RATIO:.0%} |
| Precision | {run_config["precision"]} |
| Hardware | {run_config["cuda_device"]} |
| Observed training runtime | {training["runtime_seconds"] / 60:.1f} minutes |
| Seed | {run_config["seed"]} |

## Evaluation

All systems use the same held-out documents and organic BM25 candidate lists.
NDCG and MRR measure passage ordering; character precision/recall measure exact
overlap with annotated source coordinates.

| System | NDCG@10 | MRR@10 | Hit@5 | Char recall@5 |
|---|---:|---:|---:|---:|
| BM25 candidate order | {bm25["ndcg@10"]:.4f} | {bm25["mrr@10"]:.4f} | {bm25["hit@5"]:.4f} | {bm25["char_recall@5"]:.4f} |
| Untouched Ettin | {base["ndcg@10"]:.4f} | {base["mrr@10"]:.4f} | {base["hit@5"]:.4f} | {base["char_recall@5"]:.4f} |
| Fine-tuned model | {final["ndcg@10"]:.4f} | {final["mrr@10"]:.4f} | {final["hit@5"]:.4f} | {final["char_recall@5"]:.4f} |

| Domain | Base NDCG@10 | Fine-tuned NDCG@10 | Delta |
|---|---:|---:|---:|
{domain_rows}

## Limitations

- English-only and concentrated on NDAs, commercial contracts, M&A agreements,
  and consumer privacy policies.
- A reranker cannot recover evidence that the first-stage retriever omitted.
- The benchmark uses synthetic question templates around expert annotations;
  ordinary user language may differ.
- Relevance scores are not legal conclusions or calibrated confidence values.
- Validate on your document types, jurisdictions, and retrieval candidate
  distribution before operational use. Human legal review remains necessary.

## Artifacts and reproduction

The repository includes aggregate metrics, per-query rank coordinates without
contract text, the resolved configuration, trainer state, package versions,
and TensorBoard event files under `tensorboard/`.

Training source:
[`scripts/finetune_legalbenchrag_ettin_reranker.py`](https://github.com/LxYuan0420/nlp/blob/main/scripts/finetune_legalbenchrag_ettin_reranker.py)

```bash
uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py
```

| Package | Tested version |
|---|---|
{package_rows}

## Data attribution

LegalBench-RAG builds on ContractNLI, CUAD, MAUD, and PrivacyQA. ContractNLI,
CUAD, and MAUD are distributed under CC BY 4.0; consult every source dataset's
terms and the upstream LegalBench-RAG repository before reuse. The model weights
are released under Apache-2.0, matching the base checkpoint.
"""

    def _repository_id(self) -> str:
        if not (self.config.publish or self.config.publish_only):
            return f"YOUR_USERNAME/{MODEL_REPO_NAME}"
        user = HfApi().whoami()["name"]
        return f"{user}/{MODEL_REPO_NAME}"

    @staticmethod
    def _verification_section(verification: dict[str, Any] | None) -> str:
        if verification is None:
            return ""
        rows = "\n".join(
            "| {query} | {passage} | {score:.4f} |".format(
                query=example["query"].replace("|", "\\|"),
                passage=example["top_passage"].replace("|", "\\|"),
                score=example["top_score"],
            )
            for example in verification["examples"]
        )
        return f"""## Verified examples

These outputs were produced after reloading the published repository in a new
`CrossEncoder` instance.

| Query | Highest-ranked passage | Score |
|---|---|---:|
{rows}"""

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


class HubPublisher:
    """Publish completed artifacts, reload remotely, and verify real inference."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.api = HfApi()

    def publish_and_verify(self) -> None:
        output_dir = self.config.output_path
        missing = [
            name
            for name in REQUIRED_LOCAL_ARTIFACTS
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot publish; missing local artifacts: {missing}"
            )
        repository_id = f"{self.api.whoami()['name']}/{MODEL_REPO_NAME}"
        self.api.create_repo(repository_id, repo_type="model", exist_ok=True)
        self.api.upload_folder(
            repo_id=repository_id,
            repo_type="model",
            folder_path=output_dir,
            ignore_patterns=[
                "checkpoints/**",
                "baseline-eval/**",
                "*.zip",
                "*.download",
            ],
            commit_message="Publish full LegalBench-RAG reranker experiment",
        )
        remote_files = set(self.api.list_repo_files(repository_id, repo_type="model"))
        missing_remote = [
            name for name in REQUIRED_LOCAL_ARTIFACTS if name not in remote_files
        ]
        if missing_remote:
            raise FileNotFoundError(f"Hub upload is incomplete: {missing_remote}")

        remote_model = CrossEncoder(repository_id, max_length=MAX_PAIR_TOKENS)
        examples = [
            {
                "query": "When may either party terminate the agreement?",
                "passages": [
                    "Either party may terminate with thirty days written notice.",
                    "Invoices are due sixty days after receipt.",
                    "Confidential material must be returned on request.",
                ],
                "expected_top": 0,
            },
            {
                "query": "Which law governs the contract?",
                "passages": [
                    "The supplier shall maintain insurance coverage.",
                    "This agreement is governed by the laws of New York.",
                    "Notices must be delivered by registered mail.",
                ],
                "expected_top": 1,
            },
            {
                "query": "How long do confidentiality obligations survive?",
                "passages": [
                    "Payment shall be made in United States dollars.",
                    "Confidentiality obligations survive termination for five years.",
                    "The agreement may be signed in counterparts.",
                ],
                "expected_top": 1,
            },
        ]
        results = []
        for example in examples:
            ranked = remote_model.rank(
                example["query"],
                example["passages"],
                return_documents=True,
            )
            top_index = int(ranked[0]["corpus_id"])
            results.append(
                {
                    "query": example["query"],
                    "expected_top": example["expected_top"],
                    "actual_top": top_index,
                    "top_score": float(ranked[0]["score"]),
                    "top_passage": example["passages"][top_index],
                    "passed": top_index == example["expected_top"],
                }
            )
        if not all(result["passed"] for result in results):
            raise AssertionError(f"Published usage verification failed: {results}")
        verification_path = output_dir / "verification_results.json"
        verification_path.write_text(
            json.dumps({"repository_id": repository_id, "examples": results}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        readme_path = ModelCardWriter(self.config).write()
        self.api.upload_folder(
            repo_id=repository_id,
            repo_type="model",
            folder_path=output_dir,
            allow_patterns=["README.md", "verification_results.json"],
            commit_message="Add verified inference examples",
        )
        if not readme_path.is_file():
            raise FileNotFoundError(
                "Model card regeneration did not produce README.md."
            )
        final_remote_files = set(
            self.api.list_repo_files(repository_id, repo_type="model")
        )
        final_missing = [
            name for name in REQUIRED_REMOTE_ARTIFACTS if name not in final_remote_files
        ]
        if final_missing:
            raise FileNotFoundError(
                f"Final Hub repository is incomplete: {final_missing}"
            )
        print(f"Published and verified https://huggingface.co/{repository_id}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_CACHE_DIR)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--no-push-to-hub", action="store_true")
    args = parser.parse_args()

    modes = sum(
        (args.validate_only, args.prepare_only, args.smoke_run, args.publish_only)
    )
    if modes > 1:
        parser.error(
            "Choose only one of --validate-only, --prepare-only, --smoke-run, "
            "or --publish-only."
        )
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.publish_only and args.no_push_to_hub:
        parser.error("--publish-only conflicts with --no-push-to-hub.")

    return ExperimentConfig(
        output_dir=str(args.output_dir),
        data_dir=str(args.data_dir),
        epochs=args.epochs,
        seed=args.seed,
        publish=not args.no_push_to_hub
        and not args.validate_only
        and not args.prepare_only,
        validate_only=args.validate_only,
        prepare_only=args.prepare_only,
        smoke_run=args.smoke_run,
        publish_only=args.publish_only,
    )


def main() -> None:
    config = parse_args()
    LegalRerankerExperiment(config).run()


if __name__ == "__main__":
    main()
