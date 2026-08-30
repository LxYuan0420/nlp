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
5. Save and publish: write reproducibility artifacts and, by default, delegate
   Hub publication to the separate publication script.

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
verified free-T4 reference run took about 18 minutes for training. The model
card is intentionally maintained outside this training module at
``model_cards/functiongemma-banking77-router/README.md``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
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
TRACKIO_PROJECT = "functiongemma-banking77-routing-full-sft"
DEFAULT_OUTPUT_DIR = Path("functiongemma-banking77-router")

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
    """Own the BANKING77 selection and tool-conversation transformation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def prepare(self) -> DatasetDict:
        """Return deterministic, balanced train and held-out tool-call rows."""

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
    def explain(dataset: DatasetDict) -> None:
        """Print the source schema and one complete transformed training row."""

        sample = dataset["train"][0]
        expected_tool = ToolRoutingEvaluator.expected_tool(sample)
        print("\n=== 1. PREPARE DATASET ===")
        print(f"Source: {DATASET_ID}")
        print("Source schema:")
        print(
            json.dumps(
                {"text": "string", "label": "int", "label_text": "string"}, indent=2
            )
        )
        print(
            f"Balanced subset: {len(dataset['train'])} train / "
            f"{len(dataset['test'])} held-out rows across "
            f"{len(SELECTED_INTENTS)} intents."
        )
        print("\nTransformation:")
        print("  {text, label_text} -> {messages, tools}")
        print("  label_text -> assistant tool_calls[0].function.name")
        print(f"\nCustomer message: {sample['messages'][1]['content']}")
        print(f"Class label becomes tool: {expected_tool}")
        print("\nComplete transformed row:")
        print(json.dumps(sample, indent=2))

    @staticmethod
    def _balanced_subset(
        dataset: Dataset,
        examples_per_intent: int,
        seed: int,
    ) -> Dataset:
        """Select the same number of examples for every supported intent."""

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
        """Convert one classification row to FunctionGemma's native structure."""

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
        """Read the supervised tool name from a transformed dataset row."""

        return sample["messages"][2]["tool_calls"][0]["function"]["name"]

    @staticmethod
    def first_function_call(generated_text: str) -> tuple[str, str]:
        """Return the first complete FunctionGemma call and its function name."""

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
        """Generate on held-out messages and compute exact first-tool accuracy."""

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
        """Print varied inputs and their first generated tool calls."""

        print(f"\n=== {title} ===")
        print(f"Held-out exact tool-selection accuracy: {accuracy:.2%}")
        for row in rows[:5]:
            status = (
                "correct" if row["expected_tool"] == row["predicted_tool"] else "wrong"
            )
            print(f"[{status}] {row['customer_message']}")
            print(f"  expected:  {row['expected_tool']}")
            print(f"  predicted: {row['predicted_tool']}")
            print(f"  output:    {row['first_function_call']}")


class FunctionGemmaExperiment:
    """Orchestrate the educational data, training, evaluation, and save stages."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.tensorboard_dir = self.output_dir / "runs"
        self.evaluator = ToolRoutingEvaluator()

    def run(self, *, validate_only: bool, inspect_model: bool) -> None:
        """Execute the requested stages in the same order they are taught."""

        set_seed(self.config.seed)
        dataset = BankingToolDatasetPreparer(self.config).prepare()
        BankingToolDatasetPreparer.explain(dataset)
        if validate_only:
            print("\nDataset validation complete. No model was loaded or published.")
            return

        model, tokenizer = self._load_model()
        self._explain_model(model, tokenizer, dataset["train"][0])
        self._validate_prompt_lengths(dataset, tokenizer)
        if inspect_model:
            print("\nModel inspection complete. Nothing was trained or published.")
            return

        self._validate_training_environment()
        if self.config.publish:
            from publish_functiongemma_banking77 import validate_hf_access

            username = validate_hf_access()
            print(f"Publication namespace: {username}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer = self._build_trainer(model, tokenizer, dataset)

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

        metrics_path = self._save_artifacts(
            trainer=trainer,
            tokenizer=tokenizer,
            train_metrics=train_result.metrics,
            eval_metrics=eval_metrics,
            baseline_accuracy=baseline_accuracy,
            final_accuracy=final_accuracy,
            baseline_predictions=baseline_predictions,
            final_predictions=final_predictions,
        )
        self._publish_if_requested(metrics_path)

    @staticmethod
    def _load_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load FP32 master weights; Trainer supplies FP16 autocast on the T4."""

        print("\n=== 2. PREPARE MODEL ===")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        return model, tokenizer

    @staticmethod
    def _explain_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sample: dict[str, Any],
    ) -> None:
        """Show parameter counts and the exact serialized training example."""

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
        print(f"Base model: {BASE_MODEL_ID}")
        print(f"Architecture: {model.config.model_type} causal language model")
        print(
            f"Parameters: {total_parameters:,} total / {trainable_parameters:,} trainable"
        )
        print("Method: full fine-tuning, producing one standalone checkpoint")
        print(f"Rendered example length: {token_count} tokens")
        print("\nFunctionGemma chat-template output:")
        print(formatted)

    @staticmethod
    def _validate_prompt_lengths(
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Fail before training if any supervised function call is truncated."""

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
        print(f"Longest rendered row: {longest} tokens")
        print(f"Maximum sequence length: {MAX_SEQUENCE_LENGTH} tokens")
        if longest > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "A supervised function call would be truncated. Increase "
                "MAX_SEQUENCE_LENGTH or shorten the tool schemas."
            )

    @staticmethod
    def _validate_training_environment() -> None:
        """Require CUDA only at the boundary where training begins."""

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Training requires CUDA. Use --validate-only locally or run the "
                "full experiment on a Colab T4."
            )
        print(f"Training GPU: {torch.cuda.get_device_name(0)}")

    def _build_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DatasetDict,
    ) -> SFTTrainer:
        """Create the explicit, reproducible TRL training configuration."""

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
            completion_only_loss=False,
            assistant_only_loss=False,
            loss_type="nll",
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
        print(json.dumps(asdict(self.config), indent=2))
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Warmup: {warmup_steps} steps ({WARMUP_FRACTION:.0%} of the run)")
        print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print("Loss: next-token cross-entropy over every non-padding token")
        print(f"TensorBoard directory: {self.tensorboard_dir}")
        print(f"Trackio project: {TRACKIO_PROJECT} (local during training)")
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )

    def _save_artifacts(
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
        """Save weights, state, metrics, versions, and prediction evidence."""

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
                "objective": "causal language-model next-token cross-entropy",
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

    def _publish_if_requested(self, metrics_path: Path) -> None:
        """Delegate the one-off Hub concerns to the publication module."""

        if not self.config.publish:
            print("Publication disabled with --no-push-to-hub.")
            return

        from publish_functiongemma_banking77 import publish_completed_experiment

        result = publish_completed_experiment(
            output_dir=self.output_dir,
            metrics_path=metrics_path,
        )
        print(f"Model: {result.model_url}")
        print(f"Trackio dashboard: {result.trackio_url}")


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
        from publish_functiongemma_banking77 import publish_completed_experiment

        result = publish_completed_experiment(output_dir=output_dir)
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
