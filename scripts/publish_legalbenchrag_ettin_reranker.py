#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "huggingface-hub==1.30.0",
#   "sentence-transformers==6.0.1",
#   "torch==2.14.0",
#   "transformers==5.16.1",
# ]
# ///
"""Publish and verify a completed LegalBench-RAG reranker experiment.

This is an experiment-specific, one-off release utility. It does not load the
training dataset or run training. The training script owns data preparation,
model setup, training, evaluation, TensorBoard, and artifact creation; this
utility reads those finished artifacts, writes a data-driven model card,
uploads the output directory, reloads the remote model, and records verified
inference examples.

Run after the full training command has completed successfully:

    uv run --script scripts/publish_legalbenchrag_ettin_reranker.py

Set ``HF_TOKEN`` to a Hugging Face token with model-repository write access.
Use ``--output-dir`` when training wrote to a non-default location.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard
from sentence_transformers import CrossEncoder

MODEL_REPO_NAME = "LegalBenchRAG-Ettin-150M-Reranker"
DEFAULT_OUTPUT_DIR = Path("legalbenchrag-ettin-150m-reranker")
TRAINING_SOURCE_URL = (
    "https://github.com/LxYuan0420/nlp/blob/main/"
    "scripts/finetune_legalbenchrag_ettin_reranker.py"
)
PUBLISHING_SOURCE_URL = (
    "https://github.com/LxYuan0420/nlp/blob/main/"
    "scripts/publish_legalbenchrag_ettin_reranker.py"
)
UPSTREAM_REPOSITORY = "https://github.com/ZeroEntropy-AI/legalbenchrag"

REQUIRED_TRAINING_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "run_config.json",
    "dataset_manifest.json",
    "baseline_metrics.json",
    "final_metrics.json",
    "training_metrics.json",
    "trainer_state.json",
    "package_versions.json",
)
REQUIRED_PUBLISHED_ARTIFACTS = REQUIRED_TRAINING_ARTIFACTS + (
    "README.md",
    "verification_results.json",
)


@dataclass(frozen=True, slots=True)
class PublicationConfig:
    """Immutable inputs for the one-off release operation."""

    output_dir: Path


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """Validated, structured views of a completed full training run."""

    output_dir: Path
    manifest: dict[str, Any]
    run_config: dict[str, Any]
    metrics: dict[str, Any]
    training: dict[str, Any]
    packages: dict[str, str]

    @classmethod
    def load(cls, output_dir: Path) -> ArtifactBundle:
        """Load the artifact directory produced by the training script.

        Input:
            ``output_dir`` contains model weights plus the JSON evidence files
            written after final evaluation.

        Output:
            A typed bundle with decoded manifests, configuration, metrics,
            trainer history, and package versions. Smoke runs are rejected so
            they cannot be mistaken for the full published experiment.
        """

        missing = [
            name
            for name in REQUIRED_TRAINING_ARTIFACTS
            if not (output_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Cannot publish; missing artifacts: {missing}")

        bundle = cls(
            output_dir=output_dir,
            manifest=cls._read_json(output_dir / "dataset_manifest.json"),
            run_config=cls._read_json(output_dir / "run_config.json"),
            metrics=cls._read_json(output_dir / "final_metrics.json"),
            training=cls._read_json(output_dir / "training_metrics.json"),
            packages=cls._read_json(output_dir / "package_versions.json"),
        )
        if bundle.run_config["smoke_run"]:
            raise ValueError(
                "Refusing to publish a smoke-run checkpoint as the full model."
            )
        if not (output_dir / "tensorboard").is_dir():
            raise FileNotFoundError("The completed run has no tensorboard/ directory.")
        return bundle

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


class ModelCardRenderer:
    """Render the public usage contract entirely from observed run artifacts."""

    def __init__(self, bundle: ArtifactBundle, repository_id: str) -> None:
        self.bundle = bundle
        self.repository_id = repository_id

    def write(self, verification: dict[str, Any] | None) -> Path:
        card = ModelCard(self._render(verification))
        card.validate()
        destination = self.bundle.output_dir / "README.md"
        card.save(destination)
        return destination

    def _render(self, verification: dict[str, Any] | None) -> str:
        manifest = self.bundle.manifest
        run = self.bundle.run_config
        metrics = self.bundle.metrics
        training = self.bundle.training
        final = metrics["fine_tuned_model"]["overall"]
        base = metrics["untouched_base_model"]["overall"]
        bm25 = metrics["bm25"]["overall"]
        gate = metrics["recommendation_gate"]
        status = "passed" if gate["passed"] else "did not pass"
        domains = manifest["domains"]
        domain_rows = "\n".join(
            "| {domain} | {base:.4f} | {final:.4f} | {delta:+.4f} |".format(
                domain=domain,
                base=metrics["untouched_base_model"]["by_domain"][domain]["ndcg@10"],
                final=metrics["fine_tuned_model"]["by_domain"][domain]["ndcg@10"],
                delta=metrics["per_domain_ndcg@10_delta"][domain],
            )
            for domain in domains
        )
        data_rows = "\n".join(
            f"| {domain} | {manifest['query_counts'][domain]:,} | "
            f"{manifest['document_counts'][domain]:,} |"
            for domain in domains
        )
        package_rows = "\n".join(
            f"| `{name}` | `{version}` |"
            for name, version in self.bundle.packages.items()
        )
        verification_section = self._verification_section(verification)
        best_checkpoint = Path(training["best_checkpoint"]).name
        gate_gain = gate["minimum_overall_ndcg@10_gain"]
        gate_regression = gate["maximum_allowed_domain_regression"]

        return f"""---
base_model: {run["base_model"]}
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

This is a fully fine-tuned {run["total_parameters"]:,}-parameter cross-encoder
for ranking evidence passages from English contracts and privacy policies. It
starts from [`{run["base_model"]}`](https://huggingface.co/{run["base_model"]})
at revision `{run["base_model_revision"]}` and uses the complete LegalBench-RAG
release rather than the 776-query mini benchmark.

On the document-held-out organic BM25 candidate test, this run **{status}** the
pre-registered usefulness gate. The gate required at least +{gate_gain:.2f}
NDCG@10 over the untouched base without a per-domain regression worse than
-{gate_regression:.2f}. Training loss alone is not treated as evidence that
retrieval improved.

## Intended use

Use this model as the second stage of a retrieve-and-rerank workflow:

1. Split or index legal documents.
2. Use BM25 or embeddings to retrieve 30-100 candidate passages.
3. Pass one query and those candidates to this model.
4. Send the highest-ranked evidence to a reviewer or grounded generator.

It emits relevance scores. It does not search a corpus, answer legal questions,
execute actions, or provide legal advice.

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{self.repository_id}")
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

The return value is ordered from highest to lowest relevance. Scores compare
candidates for one query; they are not calibrated probabilities.

{verification_section}

## Data and leakage controls

The pinned upstream [LegalBench-RAG release]({UPSTREAM_REPOSITORY}) contains
{manifest["total_queries"]:,} questions, {manifest["total_documents"]:,}
documents, and {manifest["evidence_spans"]:,} expert-annotated evidence spans.
All spans were checked against exact source substrings. Documents—not individual
questions—were assigned to train, validation, and test, so questions about one
contract cannot cross splits. The model repository does not redistribute source
contracts or passage text.

| Domain | Queries | Documents |
|---|---:|---:|
{data_rows}

## Schema transformation

One upstream row contains a query and exact evidence coordinates:

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

Documents become overlapping token-bounded passages. Each evidence span selects
its highest-overlap passage as a positive; BM25 supplies up to four high-ranking,
non-overlapping passages from the same document as hard negatives:

```json
{{
  "query": "What law governs this agreement?",
  "passage": "This Agreement is governed by New York law.",
  "label": 1.0
}}
```

Evaluation reranks BM25's organic top-{manifest["evaluation_candidates"]};
missing positives are never inserted into the candidate list.

## Model and objective

This is a standalone cross-encoder, not an adapter. Query and passage attend to
each other jointly and the model emits one score. `BinaryCrossEntropyLoss`
(BCE-with-logits) trains positive pairs toward 1 and hard negatives toward 0.
Every labeled pair contributes; FP16 changes memory/computation and TensorBoard
records progress, but neither changes the objective.

| Configuration | Value |
|---|---|
| Base revision | `{run["base_model_revision"]}` |
| Parameters | {run["trainable_parameters"]:,} trainable / {run["total_parameters"]:,} total |
| Epochs requested | {training["epochs_requested"]} |
| Best checkpoint | `{best_checkpoint}` |
| Pair length | {run["max_pair_tokens"]} tokens |
| Passage window / overlap | {manifest["passage_tokens"]} / {manifest["passage_overlap_tokens"]} tokens |
| Batch / accumulation / effective batch | {run["train_batch_size"]} / {run["gradient_accumulation_steps"]} / {run["effective_batch_size"]} |
| Optimizer / schedule | AdamW / linear |
| Learning rate / warmup | {run["learning_rate"]} / {run["warmup_ratio"]:.0%} |
| Precision | {run["precision"]} |
| Hardware | {run["cuda_device"]} |
| Observed training runtime | {training["runtime_seconds"] / 60:.1f} minutes |
| Seed | {run["seed"]} |

## Evaluation

All systems use the same held-out documents and organic BM25 candidate lists.
NDCG and MRR measure ordering; character precision/recall measures overlap with
the expert source coordinates.

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
- A reranker cannot recover evidence omitted by the first-stage retriever.
- Benchmark question templates may not represent ordinary user phrasing.
- Relevance scores are not legal conclusions or calibrated confidence values.
- Validate on the intended documents, jurisdictions, and retrieval distribution;
  human legal review remains necessary.

## Artifacts and reproduction

The repository includes aggregate metrics, per-query rank coordinates without
contract text, resolved configuration, trainer state, package versions, and
TensorBoard events under `tensorboard/`. Download the repository and run
`tensorboard --logdir tensorboard` to inspect the curves.

Training source: [`finetune_legalbenchrag_ettin_reranker.py`]({TRAINING_SOURCE_URL})<br>
One-off release source: [`publish_legalbenchrag_ettin_reranker.py`]({PUBLISHING_SOURCE_URL})

```bash
uv run --script scripts/finetune_legalbenchrag_ettin_reranker.py
```

| Package | Tested version |
|---|---|
{package_rows}

## Data attribution

LegalBench-RAG builds on ContractNLI, CUAD, MAUD, and PrivacyQA. ContractNLI,
CUAD, and MAUD are distributed under CC BY 4.0; consult each source dataset's
terms and the upstream repository before reuse. Model weights are Apache-2.0,
matching the base checkpoint.
"""

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
        return f"""## Verified remote examples

These outputs were produced after reloading the published repository in a new
`CrossEncoder` instance.

| Query | Highest-ranked passage | Score |
|---|---|---:|
{rows}"""


class HubRelease:
    """Own upload, remote reload, inference checks, and final file verification."""

    def __init__(self, config: PublicationConfig) -> None:
        self.bundle = ArtifactBundle.load(config.output_dir)
        self.api = HfApi()
        self.repository_id = f"{self.api.whoami()['name']}/{MODEL_REPO_NAME}"
        self.card = ModelCardRenderer(self.bundle, self.repository_id)

    def publish_and_verify(self) -> None:
        self.card.write(verification=None)
        self.api.create_repo(self.repository_id, repo_type="model", exist_ok=True)
        self.api.upload_folder(
            repo_id=self.repository_id,
            repo_type="model",
            folder_path=self.bundle.output_dir,
            ignore_patterns=[
                "checkpoints/**",
                "baseline-eval/**",
                "*.zip",
                "*.download",
            ],
            commit_message="Publish full LegalBench-RAG reranker experiment",
        )
        self._require_remote_files(REQUIRED_TRAINING_ARTIFACTS + ("README.md",))

        verification = self._verify_remote_inference()
        verification_path = self.bundle.output_dir / "verification_results.json"
        verification_path.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.card.write(verification)
        self.api.upload_folder(
            repo_id=self.repository_id,
            repo_type="model",
            folder_path=self.bundle.output_dir,
            allow_patterns=["README.md", "verification_results.json"],
            commit_message="Add verified remote inference examples",
        )
        self._require_remote_files(REQUIRED_PUBLISHED_ARTIFACTS)
        print(f"Published and verified https://huggingface.co/{self.repository_id}")

    def _verify_remote_inference(self) -> dict[str, Any]:
        model = CrossEncoder(
            self.repository_id,
            max_length=self.bundle.run_config["max_pair_tokens"],
        )
        examples = (
            (
                "When may either party terminate the agreement?",
                (
                    "Either party may terminate with thirty days written notice.",
                    "Invoices are due sixty days after receipt.",
                    "Confidential material must be returned on request.",
                ),
            ),
            (
                "Which law governs the contract?",
                (
                    "The supplier shall maintain insurance coverage.",
                    "This agreement is governed by the laws of New York.",
                    "Notices must be delivered by registered mail.",
                ),
            ),
            (
                "How long do confidentiality obligations survive?",
                (
                    "Payment shall be made in United States dollars.",
                    "Confidentiality obligations survive termination for five years.",
                    "The agreement may be signed in counterparts.",
                ),
            ),
        )
        results: list[dict[str, Any]] = []
        for query, passages in examples:
            ranked = model.rank(query, passages, return_documents=True)
            corpus_ids = [int(result["corpus_id"]) for result in ranked]
            scores = [float(result["score"]) for result in ranked]
            if sorted(corpus_ids) != list(range(len(passages))):
                raise ValueError(f"Remote rank output omitted candidates: {ranked}")
            if not all(math.isfinite(score) for score in scores):
                raise ValueError(
                    f"Remote rank output contains non-finite scores: {ranked}"
                )
            top_index = corpus_ids[0]
            results.append(
                {
                    "query": query,
                    "top_index": top_index,
                    "top_score": scores[0],
                    "top_passage": passages[top_index],
                }
            )
        return {"repository_id": self.repository_id, "examples": results}

    def _require_remote_files(self, required: tuple[str, ...]) -> None:
        remote = set(self.api.list_repo_files(self.repository_id, repo_type="model"))
        missing = [name for name in required if name not in remote]
        if missing:
            raise FileNotFoundError(f"Hub upload is incomplete: {missing}")


def parse_args() -> PublicationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return PublicationConfig(output_dir=args.output_dir)


def main() -> None:
    HubRelease(parse_args()).publish_and_verify()


if __name__ == "__main__":
    main()
