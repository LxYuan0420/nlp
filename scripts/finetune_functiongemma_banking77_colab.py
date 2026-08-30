#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate==1.14.0",
#   "datasets==5.0.1",
#   "huggingface-hub>=1.4.0,<2",
#   "sentencepiece>=0.2.1,<1",
#   "tensorboard==2.20.0",
#   "trackio==0.37.0",
#   "transformers==5.16.1",
#   "trl==1.12.0",
# ]
# ///
"""Train FunctionGemma to turn banking messages into tool calls.

This tutorial experiment has one deliberately visible pipeline:

1. Prepare data: select ten BANKING77 classes and convert each class label into
   a native FunctionGemma assistant tool call.
2. Prepare the model: load ``google/functiongemma-270m-it`` with FP32 master
   weights and inspect its rendered tool-calling prompt.
3. Configure training: build a TRL ``SFTTrainer`` with deterministic settings,
   TensorBoard logs, and local Trackio tracking.
4. Evaluate: compare exact generated tool selection before and after training.
5. Save and publish: write reproducibility artifacts, generate the model card
   from recorded metrics, and publish the complete result to Hugging Face.

Prerequisites:

* Accept https://huggingface.co/google/functiongemma-270m-it
* Use a CUDA GPU. The default is sized for a free Colab T4.
* To publish, provide a write-capable token through ``HF_TOKEN``, the standard
  Hugging Face token cache, or Colab's ``/content/hf_token`` bridge.

Run the complete experiment and publish it:

    uv run --script scripts/finetune_functiongemma_banking77_colab.py

Inspect only the data transformation on a CPU:

    uv run --script scripts/finetune_functiongemma_banking77_colab.py \
        --validate-only

Run a short local-only GPU smoke test:

    uv run --script scripts/finetune_functiongemma_banking77_colab.py \
        --epochs 0.05 --no-push-to-hub

Recover publication without retraining:

    uv run --script scripts/finetune_functiongemma_banking77_colab.py \
        --publish-only

The default run uses 800 training rows, 200 held-out rows, and four epochs. Its
verified free-T4 reference run took about 18 minutes for training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import trackio
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

BASE_MODEL_ID = "google/functiongemma-270m-it"
DATASET_ID = "mteb/banking77"
MODEL_REPO_NAME = "FunctionGemma-270M-banking77-router"
TRACKIO_SPACE_NAME = "functiongemma-banking77-trackio"
TRACKIO_BUCKET_NAME = "functiongemma-banking77-trackio-data"
TRACKIO_PROJECT = "functiongemma-banking77-routing-full-sft"
DEFAULT_OUTPUT_DIR = Path("functiongemma-banking77-router")
REMOTE_TOKEN_PATH = Path("/content/hf_token")
LOCAL_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
TRL_LOSS_DOCUMENTATION = "https://huggingface.co/docs/trl/v1.12.0/en/sft_trainer"
REQUIRED_LOCAL_ARTIFACTS = (
    "model.safetensors",
    "trainer_state.json",
    "training_metrics.json",
)
REQUIRED_REMOTE_ARTIFACTS = REQUIRED_LOCAL_ARTIFACTS + ("README.md",)

INTENT_DESCRIPTIONS = {
    "card_arrival": "Handle questions about when a newly ordered card will arrive.",
    "card_not_working": "Handle reports that a physical bank card does not work.",
    "cash_withdrawal_not_recognised": "Handle an unrecognized cash withdrawal.",
    "change_pin": "Handle requests to change a card PIN.",
    "compromised_card": "Handle reports that card details may be compromised.",
    "lost_or_stolen_card": "Handle a card reported lost or stolen.",
    "pending_card_payment": "Handle a card payment that is still pending.",
    "terminate_account": "Handle requests to close a bank account.",
    "transfer_not_received_by_recipient": (
        "Handle a transfer the recipient has not received."
    ),
    "verify_my_identity": "Handle questions about completing identity verification.",
}
SELECTED_INTENTS = tuple(INTENT_DESCRIPTIONS)

DEFAULT_SEED = 42
DEFAULT_TRAIN_EXAMPLES_PER_INTENT = 80
DEFAULT_EVAL_EXAMPLES_PER_INTENT = 20
DEFAULT_EPOCHS = 4.0
DEFAULT_GENERATION_EVAL_LIMIT = 100
GENERATION_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 1024
MAX_NEW_TOKENS = 64
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LOGGING_STEPS = 10
LEARNING_RATE = 5.0e-5
WARMUP_FRACTION = 0.1
DEVELOPER_PROMPT = (
    "You route customer requests by calling exactly one banking support tool."
)
FUNCTION_CALL_PATTERN = re.compile(
    r"<start_function_call>call:([a-z0-9_]+)(.*?)<end_function_call>",
    re.DOTALL,
)


def tool_name(intent: str) -> str:
    """Map a BANKING77 intent to its stable application handler name."""

    return f"handle_{intent}"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": tool_name(intent),
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_message": {
                        "type": "string",
                        "description": "The original customer support message.",
                    }
                },
                "required": ["customer_message"],
            },
            "return": {"type": "string"},
        },
    }
    for intent, description in INTENT_DESCRIPTIONS.items()
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment inputs recorded with every completed run."""

    output_dir: str
    epochs: float
    train_examples_per_intent: int
    eval_examples_per_intent: int
    generation_eval_limit: int
    seed: int
    publish: bool


class BankingToolDatasetPreparer:
    """Convert BANKING77 classification rows into tool-calling conversations.

    Input rows come from ``mteb/banking77`` and contain fields such as::

        {"text": "My card is lost", "label_text": "lost_or_stolen_card"}

    Output rows contain the model-native supervision consumed by
    ``SFTTrainer``::

        {
            "messages": [
                {"role": "developer", "content": "..."},
                {"role": "user", "content": "My card is lost"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "handle_lost_or_stolen_card",
                            "arguments": {"customer_message": "My card is lost"},
                        },
                    }],
                },
            ],
            "tools": [<ten JSON function schemas>],
        }
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def prepare(self) -> DatasetDict:
        """Load, balance, and transform the configured BANKING77 subsets.

        Inputs:
            ``self.config`` supplies the random seed and the number of examples
            selected per intent for the source train and test splits.

        Returns:
            A ``DatasetDict`` with ``train`` and ``test`` splits. Source columns
            such as ``text`` and ``label_text`` are replaced by ``messages`` and
            ``tools``. Each assistant message contains exactly one expected
            function call, as illustrated in the class docstring.
        """

        raw = load_dataset(DATASET_ID)
        train_split = raw["train"]
        test_split = raw["test"]
        available_intents = set(train_split.unique("label_text"))
        missing_intents = set(SELECTED_INTENTS).difference(available_intents)
        if missing_intents:
            raise ValueError(
                f"{DATASET_ID} is missing required intents: {sorted(missing_intents)}"
            )

        selected = DatasetDict(
            train=self._balanced_subset(
                train_split,
                self.config.train_examples_per_intent,
                self.config.seed,
            ),
            test=self._balanced_subset(
                test_split,
                self.config.eval_examples_per_intent,
                self.config.seed,
            ),
        )
        return DatasetDict(
            {
                split_name: split.map(
                    self._to_tool_conversation,
                    remove_columns=split.column_names,
                    desc=f"Formatting {split_name} tool calls",
                )
                for split_name, split in selected.items()
            }
        )

    @staticmethod
    def _balanced_subset(
        dataset: Dataset,
        examples_per_intent: int,
        seed: int,
    ) -> Dataset:
        """Select and shuffle an equal number of rows for every intent.

        Args:
            dataset: A raw BANKING77 split containing ``label_text``.
            examples_per_intent: Number of rows retained for each of the ten
                supported intents.
            seed: Base seed. Each intent receives a deterministic offset.

        Returns:
            One shuffled ``Dataset`` containing
            ``examples_per_intent * 10`` source-format rows.
        """

        subsets: list[Dataset] = []
        for offset, intent in enumerate(SELECTED_INTENTS):
            intent_rows = dataset.filter(
                lambda row, expected=intent: row["label_text"] == expected
            )
            if len(intent_rows) < examples_per_intent:
                raise ValueError(
                    f"{DATASET_ID} has {len(intent_rows)} rows for {intent}; "
                    f"requested {examples_per_intent}."
                )
            subsets.append(
                intent_rows.shuffle(seed=seed + offset).select(
                    range(examples_per_intent)
                )
            )
        return concatenate_datasets(subsets).shuffle(seed=seed)

    @staticmethod
    def _to_tool_conversation(row: dict[str, Any]) -> dict[str, Any]:
        """Convert one classification example into one supervised tool call.

        Args:
            row: A BANKING77 row with ``text`` and ``label_text``. For example,
                ``{"text": "Change my PIN", "label_text": "change_pin"}``.

        Returns:
            A dictionary with ``messages`` and ``tools``. The example input
            produces an assistant call whose function name is
            ``handle_change_pin`` and whose ``customer_message`` argument is
            ``"Change my PIN"``.
        """

        customer_message = row["text"]
        intent = row["label_text"]
        return {
            "messages": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": customer_message},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name(intent),
                                "arguments": {"customer_message": customer_message},
                            },
                        }
                    ],
                },
            ],
            "tools": TOOLS,
        }


class ToolRoutingEvaluator:
    """Measure the application behavior: the first generated tool name."""

    @staticmethod
    def expected_tool(sample: dict[str, Any]) -> str:
        """Return the supervised name, such as ``handle_change_pin``.

        Args:
            sample: One transformed row returned by
                ``BankingToolDatasetPreparer.prepare``.
        """

        return sample["messages"][2]["tool_calls"][0]["function"]["name"]

    @staticmethod
    def first_function_call(generated_text: str) -> tuple[str, str]:
        """Extract the first complete call from raw FunctionGemma text.

        Args:
            generated_text: Model output that may continue after the first
                ``<end_function_call>`` marker.

        Returns:
            ``(tool_name, complete_call)``. If no complete call exists, the
            tool name is ``no_function_call`` and the second item is the
            stripped raw output.
        """

        match = FUNCTION_CALL_PATTERN.search(generated_text)
        if match is None:
            return "no_function_call", generated_text.strip()
        return match.group(1), match.group(0)

    @torch.inference_mode()
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Dataset,
        limit: int,
    ) -> tuple[float, list[dict[str, str]]]:
        """Generate held-out calls and compute exact first-tool accuracy.

        Args:
            model: The base or fine-tuned causal language model.
            tokenizer: FunctionGemma tokenizer containing its chat template.
            eval_dataset: Transformed rows with expected assistant tool calls.
            limit: Maximum number of rows evaluated in deterministic order.

        Returns:
            ``(accuracy, predictions)``. Accuracy is a float from 0 to 1. Each
            prediction has this readable shape::

                {
                    "customer_message": "Change my PIN",
                    "expected_tool": "handle_change_pin",
                    "predicted_tool": "handle_change_pin",
                    "first_function_call": "<start_function_call>...",
                }
        """

        sample_count = min(limit, len(eval_dataset))
        samples = eval_dataset.select(range(sample_count))
        model.eval()
        model_device = next(model.parameters()).device
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        results: list[dict[str, str]] = []

        try:
            for start in range(0, sample_count, GENERATION_BATCH_SIZE):
                stop = min(start + GENERATION_BATCH_SIZE, sample_count)
                batch = samples.select(range(start, stop))
                prompts = [
                    tokenizer.apply_chat_template(
                        row["messages"][:2],
                        tools=row["tools"],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for row in batch
                ]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQUENCE_LENGTH,
                    add_special_tokens=False,
                ).to(model_device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
                generations = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=False,
                )
                for row, generation in zip(batch, generations, strict=True):
                    predicted_tool, first_call = self.first_function_call(generation)
                    results.append(
                        {
                            "customer_message": row["messages"][1]["content"],
                            "expected_tool": self.expected_tool(row),
                            "predicted_tool": predicted_tool,
                            "first_function_call": first_call,
                        }
                    )
        finally:
            tokenizer.padding_side = original_padding_side

        correct = sum(row["expected_tool"] == row["predicted_tool"] for row in results)
        return correct / len(results), results

    @staticmethod
    def print_results(
        title: str,
        accuracy: float,
        rows: list[dict[str, str]],
    ) -> None:
        """Print one metric plus three compact input-to-tool examples."""

        print(f"\n=== {title} ===")
        print(f"Held-out exact tool-selection accuracy: {accuracy:.2%}")
        for row in rows[:3]:
            status = (
                "correct" if row["expected_tool"] == row["predicted_tool"] else "wrong"
            )
            print(f"[{status}] {row['customer_message']} -> {row['predicted_tool']}")


class FunctionGemmaExperiment:
    """Orchestrate the educational data, training, evaluation, and save stages."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.tensorboard_dir = self.output_dir / "runs"
        self.evaluator = ToolRoutingEvaluator()

    def run(self, *, validate_only: bool, inspect_model: bool) -> None:
        """Run preparation, training, evaluation, saving, and publication.

        Args:
            validate_only: Stop after loading and transforming the dataset.
            inspect_model: Stop after loading the model and printing one exact
                rendered training prompt. This is the intentionally verbose
                inspection mode; normal training prints compact summaries.

        Returns:
            ``None``. Model files, metrics, predictions, and tracking logs are
            written under ``config.output_dir`` when training completes.
        """

        set_seed(self.config.seed)
        print("\n=== 1. PREPARE DATASET ===")
        dataset = BankingToolDatasetPreparer(self.config).prepare()
        print(
            f"Prepared {len(dataset['train'])} train and {len(dataset['test'])} "
            f"held-out tool-calling rows across {len(SELECTED_INTENTS)} intents."
        )
        if validate_only:
            print("\nDataset validation complete. No model was loaded or published.")
            return

        model, tokenizer = self.load_model()
        self.summarize_model(
            model,
            tokenizer,
            dataset["train"][0],
            show_rendered_prompt=inspect_model,
        )
        self.validate_prompt_lengths(dataset, tokenizer)
        if inspect_model:
            print("\nModel inspection complete. Nothing was trained or published.")
            return

        self._validate_training_environment()
        if self.config.publish:
            HubExperimentPublisher.validate_access()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer = self.build_trainer(model, tokenizer, dataset)

        baseline_accuracy, baseline_predictions = self.evaluator.evaluate(
            trainer.model,
            tokenizer,
            dataset["test"],
            self.config.generation_eval_limit,
        )
        self.evaluator.print_results(
            "4. EVALUATE BEFORE TRAINING",
            baseline_accuracy,
            baseline_predictions,
        )

        print("\n=== 5. TRAIN ===")
        train_result = trainer.train()

        print("\n=== 6. EVALUATE AFTER TRAINING ===")
        eval_metrics = trainer.evaluate()
        final_accuracy, final_predictions = self.evaluator.evaluate(
            trainer.model,
            tokenizer,
            dataset["test"],
            self.config.generation_eval_limit,
        )
        self.evaluator.print_results(
            "FINAL GENERATED TOOL CALLS",
            final_accuracy,
            final_predictions,
        )

        metrics_path = self.save_artifacts(
            trainer=trainer,
            tokenizer=tokenizer,
            train_metrics=train_result.metrics,
            eval_metrics=eval_metrics,
            baseline_accuracy=baseline_accuracy,
            final_accuracy=final_accuracy,
            baseline_predictions=baseline_predictions,
            final_predictions=final_predictions,
        )
        self.publish(metrics_path)

    @staticmethod
    def load_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the trainable model and tokenizer.

        Returns:
            ``(model, tokenizer)``. The model has FP32 master weights; TRL uses
            FP16 autocast during T4 training. The tokenizer owns FunctionGemma's
            tool-aware chat template.
        """

        print("\n=== 2. PREPARE MODEL ===")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        return model, tokenizer

    @staticmethod
    def summarize_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sample: dict[str, Any],
        *,
        show_rendered_prompt: bool,
    ) -> None:
        """Print model metadata and optionally one rendered prompt.

        Args:
            model: Loaded FunctionGemma causal language model.
            tokenizer: Tokenizer used to serialize messages and tool schemas.
            sample: One transformed row with ``messages`` and ``tools``.
            show_rendered_prompt: Print the full serialized prompt only for
                explicit ``--inspect-model`` runs.

        Returns:
            ``None``. A normal run prints one compact summary line. Inspection
            mode additionally prints the exact string given to the tokenizer.
        """

        formatted = tokenizer.apply_chat_template(
            sample["messages"],
            tools=sample["tools"],
            add_generation_prompt=False,
            tokenize=False,
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        token_count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        print(
            f"{BASE_MODEL_ID}: {model.config.model_type}, "
            f"{trainable_parameters:,}/{total_parameters:,} trainable parameters, "
            f"example length {token_count} tokens."
        )
        if show_rendered_prompt:
            print("\nFunctionGemma chat-template output:\n")
            print(formatted)

    @staticmethod
    def validate_prompt_lengths(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Verify all transformed rows fit without target truncation.

        Args:
            dataset: Prepared train and test tool-calling rows.
            tokenizer: FunctionGemma tokenizer used to render and count tokens.

        Returns:
            ``None``. Raises ``ValueError`` when the longest row exceeds
            ``MAX_SEQUENCE_LENGTH``.
        """

        longest = 0
        for split in dataset.values():
            for row in split:
                token_ids = tokenizer.apply_chat_template(
                    row["messages"],
                    tools=row["tools"],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=False,
                )
                longest = max(longest, len(token_ids))
        print(f"Prompt lengths: longest={longest}, limit={MAX_SEQUENCE_LENGTH} tokens.")
        if longest > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "A supervised function call would be truncated. Increase "
                "MAX_SEQUENCE_LENGTH or shorten the tool schemas."
            )

    @staticmethod
    def _validate_training_environment() -> None:
        """Require CUDA at the training boundary and print the assigned GPU."""

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Training requires CUDA. Use --validate-only locally or run the "
                "full experiment on a Colab T4."
            )
        print(f"Training GPU: {torch.cuda.get_device_name(0)}")

    def build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DatasetDict,
    ) -> SFTTrainer:
        """Create the reproducible TRL trainer.

        Args:
            model: FP32 FunctionGemma model to fully fine-tune.
            tokenizer: Serializer for the conversational tool-calling rows.
            dataset: ``DatasetDict`` returned by ``prepare``. ``train`` drives
                optimization and ``test`` supplies epoch validation loss.

        Returns:
            An ``SFTTrainer`` configured for full-sequence ``chunked_nll``.
            It logs to TensorBoard and local Trackio, evaluates once per epoch,
            and saves at most one intermediate checkpoint.
        """

        print("\n=== 3. CONFIGURE TRAINING AND TRACKING ===")
        run_name = (
            f"functiongemma-banking77-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        )
        estimated_optimizer_steps = math.ceil(
            len(dataset["train"])
            * self.config.epochs
            / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        )
        warmup_steps = max(1, math.ceil(estimated_optimizer_steps * WARMUP_FRACTION))
        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            run_name=run_name,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            max_length=MAX_SEQUENCE_LENGTH,
            packing=False,
            # This is a conversational language-modeling dataset, so these
            # settings intentionally supervise the full rendered sequence.
            completion_only_loss=False,
            assistant_only_loss=False,
            # TRL 1.12 defaults to chunked_nll. Pin it explicitly so a future
            # library default cannot silently change the experiment objective.
            # It uses the same labels and cross-entropy as nll, but projects the
            # vocabulary logits for non-ignored positions in smaller chunks
            # instead of materializing one [batch, sequence, vocabulary] tensor.
            loss_type="chunked_nll",
            gradient_checkpointing=False,
            fp16=True,
            bf16=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS,
            logging_first_step=True,
            report_to=["tensorboard", "trackio"],
            project=TRACKIO_PROJECT,
            trackio_space_id=None,
            trackio_static_space_id=False,
            seed=self.config.seed,
            data_seed=self.config.seed,
        )
        print(
            f"epochs={self.config.epochs:g}, learning_rate={LEARNING_RATE}, "
            f"effective_batch={TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, "
            f"warmup_steps={warmup_steps}, loss=chunked_nll."
        )
        print(f"Logs: TensorBoard={self.tensorboard_dir}, Trackio={TRACKIO_PROJECT}.")
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )

    def save_artifacts(
        self,
        *,
        trainer: SFTTrainer,
        tokenizer: PreTrainedTokenizerBase,
        train_metrics: dict[str, Any],
        eval_metrics: dict[str, Any],
        baseline_accuracy: float,
        final_accuracy: float,
        baseline_predictions: list[dict[str, str]],
        final_predictions: list[dict[str, str]],
    ) -> Path:
        """Save the standalone model and all reproducibility evidence.

        Args:
            trainer: Completed trainer containing model weights and log history.
            tokenizer: Tokenizer and chat template paired with the model.
            train_metrics: Runtime and aggregate loss returned by ``train``.
            eval_metrics: Final token-level validation measurements.
            baseline_accuracy: Exact generated tool accuracy before training.
            final_accuracy: Exact generated tool accuracy after training.
            baseline_predictions: Per-request baseline outputs.
            final_predictions: Per-request fine-tuned outputs.

        Returns:
            Path to ``training_metrics.json``. Its top-level keys include
            ``config``, ``training_contract``, ``train_metrics``,
            ``eval_metrics``, both prediction lists, package ``versions``, and
            tracking destinations. The same file later drives model-card
            generation, so published claims come from recorded results.
        """

        print("\n=== 7. SAVE ARTIFACTS ===")
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(self.output_dir)
        trainer.save_state()
        metrics = {
            "base_model": BASE_MODEL_ID,
            "dataset": DATASET_ID,
            "selected_intents": list(SELECTED_INTENTS),
            "config": asdict(self.config),
            "training_contract": {
                "objective": "chunked_nll",
                "objective_math": "causal language-model next-token cross-entropy",
                "loss_tokens": "all non-padding tokens in each rendered conversation",
                "task_metric": "exact first generated tool-name accuracy",
                "warmup_steps": trainer.args.warmup_steps,
                "warmup_fraction": WARMUP_FRACTION,
            },
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "baseline_tool_accuracy": baseline_accuracy,
            "final_tool_accuracy": final_accuracy,
            "evaluated_examples": len(final_predictions),
            "baseline_predictions": baseline_predictions,
            "final_predictions": final_predictions,
            "versions": package_versions(),
            "tracking": {
                "tensorboard_dir": str(self.tensorboard_dir),
                "trackio_project": TRACKIO_PROJECT,
            },
            "model_repo_id": None,
            "trackio_space_id": None,
        }
        metrics_path = self.output_dir / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved model and experiment evidence: {self.output_dir.resolve()}")
        return metrics_path

    def publish(self, metrics_path: Path) -> PublicationResult | None:
        """Generate the card from metrics and optionally publish the output.

        Args:
            metrics_path: ``training_metrics.json`` returned by
                ``save_artifacts``.

        Returns:
            Verified model and Trackio URLs when ``config.publish`` is true;
            otherwise ``None`` after leaving all artifacts local.
        """

        if not self.config.publish:
            print("Publication disabled with --no-push-to-hub.")
            return None

        result = HubExperimentPublisher(self.output_dir).publish(metrics_path)
        print(f"Model: {result.model_url}")
        print(f"Trackio dashboard: {result.trackio_url}")
        return result


@dataclass(frozen=True)
class PublicationResult:
    """Verified Hugging Face destinations for a completed experiment."""

    model_url: str
    trackio_url: str


class ModelCardRenderer:
    """Render a data-driven Hub README from completed experiment metrics."""

    @classmethod
    def render(
        cls,
        metrics: dict[str, Any],
        model_repo_id: str,
        trackio_space_id: str,
    ) -> str:
        """Return a complete model card containing only recorded claims."""

        config = metrics["config"]
        train_metrics = metrics["train_metrics"]
        eval_metrics = metrics["eval_metrics"]
        versions = metrics["versions"]
        baseline_accuracy = float(metrics["baseline_tool_accuracy"])
        final_accuracy = float(metrics["final_tool_accuracy"])
        examples = cls._example_rows(metrics["final_predictions"])
        tools_table = "\n".join(
            f"| `{tool_name(intent)}` | {description} |"
            for intent, description in INTENT_DESCRIPTIONS.items()
        )
        versions_list = "\n".join(
            f"- `{name}=={version}`" for name, version in versions.items()
        )
        example_table = "\n".join(
            f"| {cls._markdown_code(message)} | `{name}` |"
            for message, name, _ in examples
        )
        example_input, _, example_output = examples[0]
        warmup_description = cls._warmup_description(metrics)

        return f"""---
base_model: {BASE_MODEL_ID}
library_name: transformers
license: gemma
language:
- en
datasets:
- {DATASET_ID}
pipeline_tag: text-generation
tags:
- trl
- function-calling
- intent-classification
- banking77
model-index:
- name: {MODEL_REPO_NAME}
  results:
  - task:
      type: text-generation
      name: Banking tool routing
    dataset:
      name: BANKING77 ten-intent held-out slice
      type: {DATASET_ID}
      split: test
    metrics:
    - type: accuracy
      value: {final_accuracy:.4f}
      name: Exact first-tool accuracy
---

# FunctionGemma 270M BANKING77 Router

This is a full fine-tune of
[`{BASE_MODEL_ID}`](https://huggingface.co/{BASE_MODEL_ID}) that turns an
English banking request into one of ten structured support tool calls. It is a
learning experiment, not a production banking system.

## What was trained?

BANKING77 is normally a classification dataset with `text`, integer `label`,
and human-readable `label_text` fields. This experiment does not add a
classification head. It converts `label_text` into the expected assistant tool
call and fine-tunes FunctionGemma to generate that native structure.

```json
{{
  "text": "My card is gone. I think it was stolen.",
  "label_text": "lost_or_stolen_card"
}}
```

becomes a target call to `handle_lost_or_stolen_card`.

## Tool schema

Every tool receives the original customer message. One complete schema is:

```json
{json.dumps(TOOLS[5], indent=2)}
```

| Function | Intended request |
| --- | --- |
{tools_table}

## Use the model

```python
{cls._usage_code(model_repo_id)}
```

The model selects a call; it does not execute the tool. Validate arguments and
dispatch through an explicit allow-listed handler map.

## Observed held-out examples

| Customer input | First generated tool |
| --- | --- |
{example_table}

```text
Input:  {example_input}
Output: {example_output}
```

## Loss function

This run uses TRL `SFTTrainer` with `loss_type="chunked_nll"`. This is the
standard causal-language-model next-token negative log-likelihood, or
cross-entropy, computed in memory-saving chunks:

```text
loss = mean(-log P(correct next token | previous tokens))
```

`assistant_only_loss=False`, and this is a conversational language-modeling
dataset rather than a prompt-completion dataset. Therefore every non-padding
token in the rendered developer prompt, tool declarations, user message, and
assistant call contributes. Padding labels use `-100` and are ignored. Exact
generated tool accuracy is a separate application metric and is not the
differentiable loss. Chunking does not alter this token selection; the label
mask controls which tokens contribute, while `loss_type` controls how the same
calculation is held in memory. See the
[TRL 1.12 SFT documentation]({TRL_LOSS_DOCUMENTATION}).

## Results

| Metric | Value |
| --- | ---: |
| Base-model exact first-tool accuracy | {baseline_accuracy:.2%} |
| Fine-tuned exact first-tool accuracy | {final_accuracy:.2%} |
| Generated evaluation examples | {int(metrics["evaluated_examples"])} |
| Final evaluation loss | {float(eval_metrics["eval_loss"]):.4f} |
| Mean training loss | {float(train_metrics["train_loss"]):.4f} |

Training loss can continue falling while validation loss rises because the
model becomes more confident on repeated training rows without improving
equally on unseen rows. Cross-entropy can increase from a few confidently wrong
tokens even when average token accuracy changes little.

See the [Trackio dashboard](https://huggingface.co/spaces/{trackio_space_id})
for the training curves. TensorBoard events, `trainer_state.json`, and
`training_metrics.json` are included in this repository.

## Training configuration

| Setting | Value |
| --- | --- |
| Training examples | {len(SELECTED_INTENTS) * int(config["train_examples_per_intent"])} |
| Evaluation examples | {len(SELECTED_INTENTS) * int(config["eval_examples_per_intent"])} |
| Epochs | {float(config["epochs"]):g} |
| Learning rate | {LEARNING_RATE} |
| Effective batch size | {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} |
| Maximum sequence length | {MAX_SEQUENCE_LENGTH} |
| Loss | `chunked_nll`, full rendered sequence except padding |
| Warmup | {warmup_description} |
| Seed | {int(config["seed"])} |

Software:

{versions_list}

## Precision and limitations

Load the saved FP32 checkpoint without forcing all weights to FP16. The
verified FP32 Hub reload produced a valid function call, while forced pure FP16
on a T4 produced padding-only output. Training used FP16 autocast around FP32
master weights.

- Only ten BANKING77 intents are supported, not all 77.
- There is no out-of-scope or refusal route.
- Argument quality was not scored separately from tool-name selection.
- Ambiguous, adversarial, multilingual, or unrelated requests may route
  incorrectly.
- Do not use this model for financial decisions without production privacy,
  safety, monitoring, fallback, and human-review controls.

The BANKING77 mirror describes the dataset as CC BY 4.0. FunctionGemma weights
remain subject to the Gemma terms.
"""

    @staticmethod
    def _example_rows(
        predictions: list[dict[str, str]],
        limit: int = 6,
    ) -> list[tuple[str, str, str]]:
        """Select deterministic examples with distinct generated tools."""

        examples: list[tuple[str, str, str]] = []
        seen_tools: set[str] = set()
        for row in predictions:
            predicted_tool = row["predicted_tool"]
            if predicted_tool in seen_tools:
                continue
            raw_output = row.get("first_function_call") or row.get("raw_generation", "")
            end_marker = "<end_function_call>"
            if end_marker in raw_output:
                raw_output = raw_output.split(end_marker, maxsplit=1)[0] + end_marker
            examples.append(
                (row["customer_message"], predicted_tool, raw_output.strip())
            )
            seen_tools.add(predicted_tool)
            if len(examples) == limit:
                break
        if not examples:
            raise ValueError("No final predictions are available for the model card.")
        return examples

    @staticmethod
    def _markdown_code(value: str) -> str:
        """Format a short value safely inside a Markdown table cell."""

        escaped = value.replace("|", "\\|").replace("\n", " ")
        return f"`{escaped}`"

    @staticmethod
    def _warmup_description(metrics: dict[str, Any]) -> str:
        """Describe either the corrected run or the original reference run."""

        contract = metrics.get("training_contract")
        if isinstance(contract, dict) and "warmup_steps" in contract:
            return f"{int(contract['warmup_steps'])} steps"
        return "0.1 steps in the reference run (effectively no warmup)"

    @staticmethod
    def _usage_code(model_repo_id: str) -> str:
        """Create standalone inference code with all ten tool definitions."""

        descriptions = json.dumps(
            {
                tool_name(intent): description
                for intent, description in INTENT_DESCRIPTIONS.items()
            },
            indent=4,
        )
        template = """import re

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "__MODEL_ID__"
TOOL_DESCRIPTIONS = __TOOL_DESCRIPTIONS__


def make_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_message": {"type": "string"},
                },
                "required": ["customer_message"],
            },
            "return": {"type": "string"},
        },
    }


tools = [make_tool(name, description) for name, description in TOOL_DESCRIPTIONS.items()]
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
message = "My card was stolen last night"
inputs = tokenizer.apply_chat_template(
    [
        {
            "role": "developer",
            "content": "You route customer requests by calling exactly one banking support tool.",
        },
        {"role": "user", "content": message},
    ],
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
generated = tokenizer.decode(
    output[0, inputs["input_ids"].shape[1] :],
    skip_special_tokens=False,
)
match = re.search(
    r"<start_function_call>call:([a-z0-9_]+).*?<end_function_call>",
    generated,
    re.DOTALL,
)
if match is None:
    raise ValueError(f"No complete function call: {generated!r}")
print({"tool": match.group(1), "raw_call": match.group(0)})"""
        return template.replace("__MODEL_ID__", model_repo_id).replace(
            "__TOOL_DESCRIPTIONS__",
            descriptions,
        )


class HubExperimentPublisher:
    """Generate the model card, publish artifacts, and verify Hub state."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.api = HfApi(token=load_hf_token())

    @staticmethod
    def validate_access() -> str:
        """Fail before training if publication lacks authenticated Hub access."""

        identity = HfApi(token=load_hf_token()).whoami()
        username = identity.get("name")
        if not isinstance(username, str) or not username:
            raise RuntimeError("Hugging Face did not return an account username.")
        return username

    def publish(self, metrics_path: Path | None = None) -> PublicationResult:
        """Generate a fresh card from metrics, publish, and verify the result."""

        resolved_metrics_path = (
            metrics_path or self.output_dir / "training_metrics.json"
        )
        self._validate_local_artifacts(resolved_metrics_path)
        username = self._authenticated_username()
        model_repo_id = f"{username}/{MODEL_REPO_NAME}"
        trackio_space_id = f"{username}/{TRACKIO_SPACE_NAME}"
        metrics = json.loads(resolved_metrics_path.read_text(encoding="utf-8"))
        metrics["model_repo_id"] = model_repo_id
        metrics["trackio_space_id"] = trackio_space_id
        resolved_metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "README.md").write_text(
            ModelCardRenderer.render(metrics, model_repo_id, trackio_space_id),
            encoding="utf-8",
        )

        print("\n=== 8. GENERATE MODEL CARD, PUBLISH, AND VERIFY ===")
        trackio_url = self._sync_trackio(trackio_space_id, username)
        model_url = self._upload_model(model_repo_id)
        self._verify_remote(model_repo_id, trackio_space_id)
        return PublicationResult(model_url=model_url, trackio_url=trackio_url)

    def _validate_local_artifacts(self, metrics_path: Path) -> None:
        """Fail before remote writes when a completed run is unavailable."""

        required = [self.output_dir / name for name in REQUIRED_LOCAL_ARTIFACTS]
        required.append(metrics_path)
        missing = sorted(str(path) for path in set(required) if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "Cannot publish an incomplete experiment. Missing: "
                + ", ".join(missing)
            )

    def _authenticated_username(self) -> str:
        """Resolve the Hub namespace from the token rather than an email."""

        identity = self.api.whoami()
        username = identity.get("name")
        if not isinstance(username, str) or not username:
            raise RuntimeError("Hugging Face did not return an account username.")
        print(f"Authenticated Hugging Face user: {username}")
        return username

    @staticmethod
    def _sync_trackio(trackio_space_id: str, username: str) -> str:
        """Persist the local Trackio run in a free static Space."""

        synced_space_id = trackio.sync(
            project=TRACKIO_PROJECT,
            space_id=trackio_space_id,
            bucket_id=f"{username}/{TRACKIO_BUCKET_NAME}",
            force=True,
            sdk="static",
        )
        return f"https://huggingface.co/spaces/{synced_space_id}"

    def _upload_model(self, model_repo_id: str) -> str:
        """Upload final artifacts, including the newly generated README."""

        self.api.create_repo(model_repo_id, repo_type="model", exist_ok=True)
        self.api.upload_folder(
            repo_id=model_repo_id,
            repo_type="model",
            folder_path=self.output_dir,
            ignore_patterns=["checkpoint-*", "**/checkpoint-*"],
            commit_message="Publish FunctionGemma banking router experiment",
        )
        return f"https://huggingface.co/{model_repo_id}"

    def _verify_remote(self, model_repo_id: str, trackio_space_id: str) -> None:
        """Check required model files and the static Trackio Space."""

        remote_files = set(self.api.list_repo_files(model_repo_id, repo_type="model"))
        missing = sorted(set(REQUIRED_REMOTE_ARTIFACTS).difference(remote_files))
        if missing:
            raise RuntimeError("Published model is missing: " + ", ".join(missing))
        space = self.api.space_info(trackio_space_id)
        if space.sdk != "static":
            raise RuntimeError(
                f"Expected a static Trackio Space, found sdk={space.sdk!r}."
            )
        print("Verified model files and static Trackio Space on Hugging Face.")


def load_hf_token() -> str:
    """Load a Hub token from supported boundaries without printing it."""

    environment_token = os.environ.get("HF_TOKEN", "").strip()
    if environment_token:
        return environment_token
    for token_path in (REMOTE_TOKEN_PATH, LOCAL_TOKEN_PATH):
        if not token_path.is_file():
            continue
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            if token_path == REMOTE_TOKEN_PATH:
                token_path.unlink()
            return token
    raise RuntimeError(
        "A write-capable Hugging Face token is required. Set HF_TOKEN, use "
        "`hf auth login`, or add an HF_TOKEN secret in Colab."
    )


def package_versions() -> dict[str, str]:
    """Capture the exact training stack for reproduction."""

    packages = (
        "torch",
        "transformers",
        "datasets",
        "trl",
        "tensorboard",
        "trackio",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line boundary inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--train-examples-per-intent",
        type=int,
        default=DEFAULT_TRAIN_EXAMPLES_PER_INTENT,
    )
    parser.add_argument(
        "--eval-examples-per-intent",
        type=int,
        default=DEFAULT_EVAL_EXAMPLES_PER_INTENT,
    )
    parser.add_argument(
        "--generation-eval-limit",
        type=int,
        default=DEFAULT_GENERATION_EVAL_LIMIT,
        help="Held-out rows used for before/after generated tool accuracy.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Keep model artifacts and Trackio data local.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Inspect the transformed dataset without loading the model.",
    )
    parser.add_argument(
        "--inspect-model",
        action="store_true",
        help="Load on CPU and print one rendered FunctionGemma prompt.",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Publish an already completed output directory without retraining.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    for field_name in (
        "train_examples_per_intent",
        "eval_examples_per_intent",
        "generation_eval_limit",
    ):
        if getattr(args, field_name) <= 0:
            parser.error(f"--{field_name.replace('_', '-')} must be positive")
    if args.publish_only and args.no_push_to_hub:
        parser.error("--publish-only cannot be combined with --no-push-to-hub")
    return args


def main() -> None:
    """Expose an educational one-command training and publication workflow."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.publish_only:
        result = HubExperimentPublisher(output_dir).publish()
        print(f"Model: {result.model_url}")
        print(f"Trackio dashboard: {result.trackio_url}")
        return

    config = ExperimentConfig(
        output_dir=str(output_dir),
        epochs=args.epochs,
        train_examples_per_intent=args.train_examples_per_intent,
        eval_examples_per_intent=args.eval_examples_per_intent,
        generation_eval_limit=args.generation_eval_limit,
        seed=args.seed,
        publish=not args.no_push_to_hub,
    )
    FunctionGemmaExperiment(config).run(
        validate_only=args.validate_only,
        inspect_model=args.inspect_model,
    )


if __name__ == "__main__":
    main()
