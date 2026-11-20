# SVG Motion Lab

Fine-tune a small open model to turn a static SVG and a motion instruction into an animated SVG.

[Live comparison](https://huggingface.co/spaces/lxyuan/svg-motion-lab) · [Model adapter](https://huggingface.co/lxyuan/SVG-Motion-Qwen3.5-0.8B-LoRA) · [Dataset](https://huggingface.co/datasets/lxyuan/svg-motion-lab) · [Demo video](https://lxyuan-svg-motion-lab.static.hf.space/comparison.mp4)

Trained on a free Colab Tesla T4. On all 100 held-out synthetic tasks, the untouched model passed **0/100** complete animation checks and the LoRA adapter passed **100/100**. The public demo replays four selected test outputs. All 100 paired predictions remain available in the model repository’s [evaluation evidence](https://huggingface.co/lxyuan/SVG-Motion-Qwen3.5-0.8B-LoRA/tree/main/evidence).

The main run used 100 optimizer steps (0.40 epoch; 400 training examples processed) and took 512 seconds for training. Final validation token loss was 0.00003061. Checkpoint selection used validation loss, not test results. This is a small experiment in learning a fixed animation grammar, not a general SVG animation benchmark.

Budget: free Colab GPU only. Training must fail clearly if CUDA is unavailable.

The dataset contains original procedural vector drawings and deterministic animation targets. Split by drawing family and motion composition, not by randomly shuffling near-duplicate rows. Evaluation must include render validity, preservation of the original drawing, and requested motion at sampled times. Synthetic results describe this task distribution, not general animation quality.

The final comparison uses the same inputs and generation settings with the untouched model and its LoRA adapter. No repair loop is included in either primary comparison arm.

## Project location

This experiment is maintained in `nlp/experiments/svg-motion-lab`. Run the commands below from that directory:

```sh
cd experiments/svg-motion-lab
```

## Local data and reference preview

```sh
python3 dataset.py --output data
uv run --with playwright==1.58.0 playwright install chromium
uv run --with playwright==1.58.0 python -m unittest discover -s tests -v
```

Open `data/reference-preview.html` to inspect the training targets. These are **procedural reference animations**, not model predictions.

Code and newly generated drawings: Apache-2.0. Model weights retain their upstream license.

## Train on a free Colab GPU

Select a GPU runtime and verify `torch.cuda.is_available()` before starting. No paid compute or API-generated training data is needed. The source script deliberately refuses CPU training.

With the repository downloaded into the runtime:

```sh
python dataset.py --output data
pip install transformers==5.23.0 peft==0.22.0 accelerate==1.16.0 datasets==5.2.0 tensorboard==2.21.0 playwright==1.58.0
python -m playwright install --with-deps chromium
python train.py --data data --output outputs/full --steps 100 \
  --revision 2fc06364715b967f1860aea9cf38778875588b17
```

The tested Colab runtime provides PyTorch 2.14.1+cu130. For a separate compatible GPU environment, the script also has pinned `uv` dependencies:

```sh
uv run --script train.py --data data --output outputs/full --steps 100 \
  --revision 2fc06364715b967f1860aea9cf38778875588b17
```

Checkpoints and `outputs/checkpoint-latest.tar.gz` are written every 20 steps. Copy the archive to durable storage while the runtime is alive. Resume using the original output directory and `--resume outputs/full/checkpoint-N`. TensorBoard events live under the run's `tensorboard/` directory.

## Build the replay gallery

The website shows four curated scenes from the recorded evaluation. It does not run inference or accept custom prompts. The model was evaluated on a fixed instruction grammar; open-ended prompt quality has not been established. The static site needs no inference service.

To rebuild from the published predictions without training:

```sh
python dataset.py --output data
uvx --from huggingface_hub hf download lxyuan/SVG-Motion-Qwen3.5-0.8B-LoRA \
  evidence/base-test.jsonl evidence/finetuned-test.jsonl --local-dir outputs/published
python build_demo.py --data data --run outputs/published/evidence --output demo
uv run --with playwright==1.58.0 python verify_demo.py
```

For your own training run, use `--run outputs/full`. `gallery.json` owns the four selected IDs and human-readable captions. The captions paraphrase the original directions; `demo/comparison.json` retains the exact model prompt and untouched predictions. Both views use the same camera crop for display. The full evaluation evidence is unchanged.

Edit the presentation in `web/`, then rebuild. The browser checks cover every scene, paired playback, pause, scrubbing, keyboard interaction, mobile layout and reduced motion. They save screenshots and a short WebM recording under `output/playwright/`.

See [EXPERIMENT.md](EXPERIMENT.md) for the hypothesis, data split, stopping rule and metric. The model receives **SVG source**, not image pixels. The evaluator rejects active content, preserves the original scene tree, and compares rendered transforms and opacity at fixed times. Results on this synthetic grammar do not establish performance on arbitrary artwork or unconstrained prompts.
