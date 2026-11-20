# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1", "transformers==5.23.0", "peft==0.22.0",
#   "accelerate==1.16.0", "datasets==5.2.0", "tensorboard==2.21.0",
#   "playwright==1.58.0",
# ]
# ///
"""Train a text-only SVG animation LoRA on a CUDA GPU, then measure motion.

Generate data with dataset.py first. Install Chromium with
`python -m playwright install --with-deps chromium` on the Colab runtime.
Run `python train.py --data data --output outputs/smoke --steps 20 --eval-limit 4`
for the smoke test, or omit --steps and --eval-limit for the full experiment.

The base model is frozen; loss applies only to assistant target tokens. Prompt
tokens are masked. Inputs and outputs are never truncated. No Hub write token is
needed. Output includes the adapter, tokenizer, baseline/final predictions,
browser scores, configuration, progress JSONL and TensorBoard events.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import model_info
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from evaluate import MotionEvaluator

SEED = 20261120
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 1024
GENERATION_BATCH_SIZE = 4
LORA_RANK = 16
SAVE_STEPS = 20
SYSTEM = "You animate existing SVG drawings. Preserve every original element, ID and attribute. Insert only native SVG animate or animateTransform children in the supplied motion groups. Return only the complete SVG."


@dataclass(frozen=True)
class Config:
    """Explicit data, resource and training contract for a reproducible run."""

    data: str
    output: str
    model: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "main"
    steps: int = -1
    epochs: float = 2.0
    eval_limit: int = 0
    resume: str | None = None


def messages(row: dict) -> list[dict[str, str]]:
    """Map an example to a text-only chat containing raw SVG and its request."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["input_svg"] + "\n\n" + row["instruction"]},
    ]


class AnimationData:
    """Load the three fixed splits and mask all non-assistant tokens."""

    def __init__(self, directory: Path, tokenizer: PreTrainedTokenizerBase) -> None:
        self.rows = {
            split: [
                json.loads(line)
                for line in (directory / f"{split}.jsonl").read_text().splitlines()
            ]
            for split in ("train", "validation", "test")
        }
        self.tokenizer = tokenizer
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if {r["family"] for r in self.rows[first]} & {
                r["family"] for r in self.rows[second]
            }:
                raise ValueError("Drawing families leak across splits")
        self.hashes = {
            split: hashlib.sha256(
                (directory / f"{split}.jsonl").read_bytes()
            ).hexdigest()
            for split in self.rows
        }

    def prompt(self, row: dict) -> str:
        """Use the model's native chat format with thinking disabled."""
        return self.tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(self, row: dict) -> dict:
        """Return input IDs, attention mask and assistant-only labels, each [T]."""
        prompt = self.prompt(row)
        full = self.tokenizer.apply_chat_template(
            messages(row) + [{"role": "assistant", "content": row["target_svg"]}],
            tokenize=False,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Chat template prompt is not a prefix of the supervised conversation"
            )
        if len(full_ids) > MAX_LENGTH or len(full_ids) <= len(prompt_ids):
            raise ValueError(
                f"Invalid supervised token length for {row['id']}: {len(full_ids)}"
            )
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        }

    def dataset(self, split: str) -> Dataset:
        """Encode the full split without truncation or example filtering."""
        return Dataset.from_list([self.encode(row) for row in self.rows[split]])


class PaddingCollator:
    """Right-pad variable length training rows while masking padding labels."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in features)
        padding_values = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "labels": -100,
        }
        return {
            key: torch.tensor(
                [row[key] + [padding] * (width - len(row[key])) for row in features],
                dtype=torch.long,
            )
            for key, padding in padding_values.items()
        }


class Progress(TrainerCallback):
    """Log observed metrics and atomically package resumable checkpoints."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.writer = SummaryWriter(str(output / "tensorboard"))

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: object,
    ) -> None:
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **(logs or {}),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in record.values()
        ):
            raise RuntimeError(f"Nonfinite training metric: {record}")
        with (self.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        for name, value in record.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, state.global_step)
        self.writer.flush()
        print("PROGRESS " + json.dumps(record), flush=True)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self.writer.close()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self.export()

    def export(self) -> None:
        """Create an archive outside the output tree so downloads can run mid-job."""
        archive = shutil.make_archive(
            str(self.output.parent / "checkpoint-next"), "gztar", self.output
        )
        Path(archive).replace(self.output.parent / "checkpoint-latest.tar.gz")


class Experiment:
    """Own baseline, LoRA fitting, browser evaluation and local artifact saving."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.output = Path(config.output)

    def predictions(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        data: AnimationData,
        split: str,
        label: str,
    ) -> dict:
        """Generate deterministic raw outputs; score all failures without repair."""
        path = self.output / f"{label}-{split}.jsonl"
        rows = data.rows[split]
        if self.config.eval_limit:
            rows = rows[: self.config.eval_limit]
        results = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        if [r["id"] for r in results] != [r["id"] for r in rows[: len(results)]]:
            raise ValueError("Prediction cache does not match this evaluation split")
        model.eval()
        tokenizer.padding_side = "left"
        batch_size = 1 if self.config.eval_limit else GENERATION_BATCH_SIZE
        with MotionEvaluator() as evaluator, path.open("a") as stream:
            for offset in range(len(results), len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                encoded = tokenizer(
                    [data.prompt(row) for row in batch],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                started = time.monotonic()
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.float16),
                ):
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                texts = tokenizer.batch_decode(
                    generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
                )
                batch_seconds = time.monotonic() - started
                for row, text in zip(batch, texts, strict=True):
                    text = text.strip()
                    score = evaluator.score(row, text)
                    result = {
                        "id": row["id"],
                        "prediction": text,
                        "score": asdict(score),
                        "batch_seconds": batch_seconds,
                        "batch_size": len(batch),
                    }
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    results.append(result)
                print(
                    "EVALUATION "
                    + json.dumps(
                        {
                            "label": label,
                            "split": split,
                            "completed": len(results),
                            "total": len(rows),
                            "passed_so_far": sum(
                                r["score"]["motion_pass"] for r in results
                            ),
                            "batch_seconds": batch_seconds,
                        }
                    ),
                    flush=True,
                )
        return {
            "count": len(results),
            "motion_pass_rate": sum(r["score"]["motion_pass"] for r in results)
            / len(results),
        }

    def run(self) -> None:
        """Execute only on CUDA; save enough evidence to reproduce the experiment."""
        if not torch.cuda.is_available():
            raise RuntimeError("A Colab CUDA GPU is required; CPU training is disabled")
        set_seed(SEED)
        self.output.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output / "run.json"
        revision = model_info(self.config.model, revision=self.config.revision).sha
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous["resolved_revision"] != revision:
                raise ValueError(
                    "The base revision changed; use a new output directory"
                )
            if not self.config.resume and (self.output / "progress.jsonl").exists():
                raise ValueError(
                    "Output already contains a run; specify --resume or a new output"
                )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model, revision=revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        data = AnimationData(Path(self.config.data), tokenizer)
        if (
            metadata_path.exists()
            and json.loads(metadata_path.read_text())["data_sha256"] != data.hashes
        ):
            raise ValueError("Dataset changed; use a new output directory")
        training = data.dataset("train")
        validation = data.dataset("validation")
        print(
            "DATA "
            + json.dumps(
                {
                    "counts": {s: len(r) for s, r in data.rows.items()},
                    "max_tokens": max(len(row["input_ids"]) for row in training),
                }
            ),
            flush=True,
        )
        # Keep frozen storage in float32 for T4 numerical stability; autocast
        # is used only during generation. Training stays in float32 after the T4
        # fp16 smoke test produced nonfinite gradients. No custom kernels are required.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.config.model,
            revision=revision,
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).to("cuda")
        metadata = {
            **asdict(self.config),
            "resolved_revision": revision,
            "seed": SEED,
            "gpu": torch.cuda.get_device_name(0),
            "data_sha256": data.hashes,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "datasets",
                    "accelerate",
                    "playwright",
                )
            },
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "training_precision": "float32",
            "generation_precision": "float16 autocast",
            "generation_batch_size": 1
            if self.config.eval_limit
            else GENERATION_BATCH_SIZE,
            "maximum_length": MAX_LENGTH,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        metrics = {}
        # Smoke evaluation uses validation only; the full test set stays untouched.
        split = "validation" if self.config.eval_limit else "test"
        metrics["base"] = self.predictions(model, tokenizer, data, split, "base")
        adapter = LoraConfig(
            r=LORA_RANK,
            lora_alpha=2 * LORA_RANK,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)",
        )
        model = get_peft_model(model, adapter)
        model.print_trainable_parameters()
        model.config.use_cache = False
        progress = Progress(self.output)
        planned_steps = (
            self.config.steps
            if self.config.steps > 0
            else math.ceil(len(training) / 4) * self.config.epochs
        )
        args = TrainingArguments(
            output_dir=str(self.output),
            num_train_epochs=self.config.epochs,
            max_steps=self.config.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=1,
            learning_rate=1e-4,
            warmup_steps=max(1, int(planned_steps * 0.05)),
            lr_scheduler_type="cosine",
            fp16=False,
            activation_checkpointing=True,
            activation_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            logging_nan_inf_filter=False,
            eval_strategy="steps",
            eval_steps=SAVE_STEPS,
            prediction_loss_only=True,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=SEED,
            data_seed=SEED,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=training,
            eval_dataset=validation.select(range(min(20, len(validation)))),
            data_collator=PaddingCollator(tokenizer.pad_token_id),
            callbacks=[progress],
        )
        trainer.train(resume_from_checkpoint=self.config.resume)
        trainer.save_model(str(self.output / "adapter"))
        tokenizer.save_pretrained(self.output / "adapter")
        trainer.save_state()
        model.activation_checkpointing_disable()
        model.config.use_cache = True
        metrics["finetuned"] = self.predictions(
            model, tokenizer, data, split, "finetuned"
        )
        (self.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        progress.export()
        print("RESULT " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/full")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--resume")
    Experiment(Config(**vars(parser.parse_args()))).run()
