#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=1.4.0,<2",
#   "trackio==0.37.0",
# ]
# ///
"""Publish a completed FunctionGemma banking experiment to Hugging Face.

Training and publication have different responsibilities. The training script
creates model weights, tokenizer files, metrics, predictions, TensorBoard logs,
and local Trackio data. This one-off publisher then:

1. validates the completed local artifacts;
2. syncs Trackio to a free static Hugging Face Space;
3. uploads the standalone model without intermediate checkpoints;
4. uploads the reviewed, separately maintained model card; and
5. verifies the required remote files and Space configuration.

Run after a completed local or Colab experiment:

    uv run --script scripts/publish_functiongemma_banking77.py \
        --output-dir functiongemma-banking77-router

The training script calls the same entry point automatically unless
``--no-push-to-hub`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trackio
from huggingface_hub import HfApi

MODEL_REPO_NAME = "FunctionGemma-270M-banking77-router"
TRACKIO_SPACE_NAME = "functiongemma-banking77-trackio"
TRACKIO_BUCKET_NAME = "functiongemma-banking77-trackio-data"
DEFAULT_OUTPUT_DIR = Path("functiongemma-banking77-router")
DEFAULT_MODEL_CARD = (
    Path(__file__).resolve().parents[1]
    / "model_cards"
    / "functiongemma-banking77-router"
    / "README.md"
)
REMOTE_TOKEN_PATH = Path("/content/hf_token")
LOCAL_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
REQUIRED_LOCAL_ARTIFACTS = (
    "model.safetensors",
    "trainer_state.json",
    "training_metrics.json",
)
REQUIRED_REMOTE_ARTIFACTS = (
    "README.md",
    "model.safetensors",
    "trainer_state.json",
    "training_metrics.json",
)


@dataclass(frozen=True)
class PublicationResult:
    """Stable URLs returned to the training script and notebook."""

    model_repo_id: str
    trackio_space_id: str
    model_url: str
    trackio_url: str


class HubExperimentPublisher:
    """Own Hugging Face authentication, upload, and remote verification."""

    def __init__(self, output_dir: Path, model_card_path: Path) -> None:
        self.output_dir = output_dir
        self.model_card_path = model_card_path
        self.token = load_hf_token()
        self.api = HfApi(token=self.token)

    def publish(self, metrics_path: Path | None = None) -> PublicationResult:
        """Publish a complete run and return its verified public locations."""

        resolved_metrics_path = metrics_path or (
            self.output_dir / "training_metrics.json"
        )
        self._validate_local_artifacts(resolved_metrics_path)
        username = self._authenticated_username()
        model_repo_id = f"{username}/{MODEL_REPO_NAME}"
        trackio_space_id = f"{username}/{TRACKIO_SPACE_NAME}"
        trackio_project = self._record_remote_targets(
            resolved_metrics_path,
            model_repo_id,
            trackio_space_id,
        )

        print("\n=== 8. PUBLISH AND VERIFY ===")
        trackio_url = self._sync_trackio(
            trackio_space_id,
            username,
            trackio_project,
        )
        model_url = self._upload_model(model_repo_id)
        self._verify_remote(model_repo_id, trackio_space_id)
        return PublicationResult(
            model_repo_id=model_repo_id,
            trackio_space_id=trackio_space_id,
            model_url=model_url,
            trackio_url=trackio_url,
        )

    def _validate_local_artifacts(self, metrics_path: Path) -> None:
        """Fail before any remote write when the run is incomplete."""

        required_paths = [
            self.output_dir / filename for filename in REQUIRED_LOCAL_ARTIFACTS
        ]
        required_paths.extend((metrics_path, self.model_card_path))
        missing = sorted(
            str(path) for path in set(required_paths) if not path.is_file()
        )
        if missing:
            raise FileNotFoundError(
                "Cannot publish an incomplete experiment. Missing: "
                + ", ".join(missing)
            )

    def _authenticated_username(self) -> str:
        """Resolve the destination namespace from the token, never email text."""

        identity = self.api.whoami()
        username = identity.get("name")
        if not isinstance(username, str) or not username:
            raise RuntimeError("Hugging Face did not return an account username.")
        print(f"Authenticated Hugging Face user: {username}")
        return username

    @staticmethod
    def _record_remote_targets(
        metrics_path: Path,
        model_repo_id: str,
        trackio_space_id: str,
    ) -> str:
        """Keep downloadable metrics linked to the published destinations."""

        metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        tracking = metrics.get("tracking")
        if not isinstance(tracking, dict):
            raise TypeError("training_metrics.json is missing a tracking object.")
        trackio_project = tracking.get("trackio_project")
        if not isinstance(trackio_project, str) or not trackio_project:
            raise ValueError(
                "training_metrics.json is missing tracking.trackio_project."
            )
        metrics["model_repo_id"] = model_repo_id
        metrics["trackio_space_id"] = trackio_space_id
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return trackio_project

    @staticmethod
    def _sync_trackio(
        trackio_space_id: str,
        username: str,
        trackio_project: str,
    ) -> str:
        """Persist the local Trackio project in a free static Space."""

        synced_space_id = trackio.sync(
            project=trackio_project,
            space_id=trackio_space_id,
            bucket_id=f"{username}/{TRACKIO_BUCKET_NAME}",
            force=True,
            sdk="static",
        )
        return f"https://huggingface.co/spaces/{synced_space_id}"

    def _upload_model(self, model_repo_id: str) -> str:
        """Upload final artifacts and then the reviewed standalone model card."""

        self.api.create_repo(model_repo_id, repo_type="model", exist_ok=True)
        self.api.upload_folder(
            repo_id=model_repo_id,
            repo_type="model",
            folder_path=self.output_dir,
            ignore_patterns=["checkpoint-*", "**/checkpoint-*", "README.md"],
            commit_message="Publish FunctionGemma banking router artifacts",
        )
        self.api.upload_file(
            path_or_fileobj=self.model_card_path,
            path_in_repo="README.md",
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Improve FunctionGemma banking router model card",
        )
        return f"https://huggingface.co/{model_repo_id}"

    def _verify_remote(self, model_repo_id: str, trackio_space_id: str) -> None:
        """Prove expected files and a static Space exist after publication."""

        remote_files = set(self.api.list_repo_files(model_repo_id, repo_type="model"))
        missing_files = sorted(set(REQUIRED_REMOTE_ARTIFACTS).difference(remote_files))
        if missing_files:
            raise RuntimeError(
                "Published model is missing required files: " + ", ".join(missing_files)
            )
        space = self.api.space_info(trackio_space_id)
        if space.sdk != "static":
            raise RuntimeError(
                f"Expected a static Trackio Space, found sdk={space.sdk!r}."
            )
        print("Verified model files and static Trackio Space on Hugging Face.")


def load_hf_token() -> str:
    """Load a Hub token from supported boundaries without printing its value."""

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


def validate_hf_access() -> str:
    """Fail fast before training if publication was requested without Hub access."""

    identity = HfApi(token=load_hf_token()).whoami()
    username = identity.get("name")
    if not isinstance(username, str) or not username:
        raise RuntimeError("Hugging Face did not return an account username.")
    return username


def publish_completed_experiment(
    output_dir: Path,
    metrics_path: Path | None = None,
    model_card_path: Path = DEFAULT_MODEL_CARD,
) -> PublicationResult:
    """Public API shared by the training script and standalone CLI."""

    return HubExperimentPublisher(output_dir, model_card_path).publish(metrics_path)


def parse_args() -> argparse.Namespace:
    """Parse publisher boundary inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD)
    return parser.parse_args()


def main() -> None:
    """Publish a completed run without starting another training job."""

    args = parse_args()
    result = publish_completed_experiment(
        output_dir=args.output_dir,
        model_card_path=args.model_card,
    )
    print(f"Model: {result.model_url}")
    print(f"Trackio dashboard: {result.trackio_url}")


if __name__ == "__main__":
    main()
