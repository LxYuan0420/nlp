# Experiment contract

- User: a developer animating an existing vector drawing.
- Input: raw SVG source plus a text instruction. No raster image or video input.
- Output: the same SVG with native `animate` / `animateTransform` children.
- Hypothesis: a small LoRA improves compositional motion following over the untouched Qwen3.5-0.8B under the same deterministic decoding configuration.
- Resource budget: one free Colab T4, no paid API data generation, no compute purchase.
- Main run: 100 optimizer steps, effective batch 4, learning rate 1e-4, 5 warmup steps, cosine decay, float32 training. Stop on nonfinite metrics or resource failure.
- Checkpoint rule: lowest token loss on the first 20 fixed validation examples, evaluated every 20 steps. Test results never select checkpoints.
- Primary metric: fraction of 100 test tasks with safe valid SVG, unchanged original drawing structure, and correct browser-measured transforms and opacity at all eight sampled times.
- Success target: at least 80/100 complete tasks and at least 20 percentage points above the untouched baseline. A lower result must still be reported as observed.
- Scope: original synthetic drawings, fixed instruction grammar, four motion primitives, and novel combinations. This is not a benchmark of arbitrary SVG animation or natural-language robustness.

## Data

1,000 training, 100 validation, and 100 test rows. Original scene families and motion compositions are disjoint across splits. Each training family covers every training motion composition. Rows are deterministic and exact input/instruction duplicates are removed within each split.

Training families: rocket, ball, diamond, flower, arrow. Validation: kite, cloud. Test: satellite, fish, lantern.

Training covers each motion individually and translation plus rotation. Validation combines translation and scaling. Test combines translation plus opacity, or rotation plus scaling.

All motion is linear, with progress during the first 40% of a cycle, a hold until 60%, then a return. Cycles repeat. The model must copy the scene structure exactly and may insert supported SVG animation nodes. The structural check is deliberately stricter than visual similarity; the motion check compares rendered behavior rather than target code strings.

## Trial record

The first GPU trial exposed a Transformers argument change. The next fp16 attempt stopped on a nonfinite gradient. Float32 training then completed 20 steps in 98.23 seconds with a 4.87 GiB training allocation peak. Four validation examples went from 0/4 passing to 4/4. These results only validated the pipeline and are not the main test result.

The smoke data generator correlated drawing family and motion type. That correlation was removed before the main run, with a regression test. The main run starts from the untouched model on the regenerated dataset; it does not continue from the smoke adapter.

Full run data hashes, model revision, installed package versions, predictions, losses and TensorBoard events are saved with the artifacts. Main generation uses batches of four; the smoke trial used single-example generation.
