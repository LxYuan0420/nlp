---
base_model: google/functiongemma-270m-it
library_name: transformers
license: gemma
language:
- en
datasets:
- mteb/banking77
pipeline_tag: text-generation
tags:
- trl
- function-calling
- intent-classification
- banking77
model-index:
- name: FunctionGemma-270M-banking77-router
  results:
  - task:
      type: text-generation
      name: Banking tool routing
    dataset:
      name: BANKING77 ten-intent held-out slice
      type: mteb/banking77
      split: test
    metrics:
    - type: accuracy
      value: 0.97
      name: Exact first-tool accuracy
---

# FunctionGemma 270M BANKING77 Router

This is a full fine-tune of
[`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it)
that turns an English banking request into one of ten structured support tool
calls. It is a learning experiment, not a production banking system.

The verified free-T4 run improved exact generated tool selection from **51% to
97%** on 100 deterministic held-out requests. Training took 18 minutes 22
seconds for 400 optimizer steps.

## What was trained?

[BANKING77](https://huggingface.co/datasets/mteb/banking77) is normally an
intent-classification dataset:

```json
{
  "text": "My card is gone. I think it was stolen.",
  "label": 43,
  "label_text": "lost_or_stolen_card"
}
```

This experiment does **not** add a 10-class classification head. It converts
the label into the assistant output of a tool-calling conversation:

```json
{
  "messages": [
    {
      "role": "developer",
      "content": "You route customer requests by calling exactly one banking support tool."
    },
    {
      "role": "user",
      "content": "My card is gone. I think it was stolen."
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "handle_lost_or_stolen_card",
            "arguments": {
              "customer_message": "My card is gone. I think it was stolen."
            }
          }
        }
      ]
    }
  ],
  "tools": ["the ten JSON schemas shown below"]
}
```

FunctionGemma's chat template serializes the messages and schemas into the
model's native function-declaration and function-call tokens. At inference
time, the model generates a function call rather than a class ID.

## Tool schema

Every tool uses the same argument contract. Only its name and description
change:

```json
{
  "type": "function",
  "function": {
    "name": "handle_lost_or_stolen_card",
    "description": "Handle a card reported lost or stolen.",
    "parameters": {
      "type": "object",
      "properties": {
        "customer_message": {
          "type": "string",
          "description": "The original customer support message."
        }
      },
      "required": [
        "customer_message"
      ]
    },
    "return": {
      "type": "string"
    }
  }
}
```

Supported tools:

| Function | Intended request |
| --- | --- |
| `handle_card_arrival` | When a newly ordered card will arrive |
| `handle_card_not_working` | A physical card does not work |
| `handle_cash_withdrawal_not_recognised` | An unrecognized cash withdrawal |
| `handle_change_pin` | Change a card PIN |
| `handle_compromised_card` | Card details may be compromised |
| `handle_lost_or_stolen_card` | A card is lost or stolen |
| `handle_pending_card_payment` | A card payment is pending |
| `handle_terminate_account` | Close a bank account |
| `handle_transfer_not_received_by_recipient` | A recipient did not receive a transfer |
| `handle_verify_my_identity` | Complete identity verification |

## Use the model

Install `torch` and `transformers==5.16.1`, then run:

```python
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "lxyuan/FunctionGemma-270M-banking77-router"
DEVELOPER_PROMPT = (
    "You route customer requests by calling exactly one banking support tool."
)
TOOL_DESCRIPTIONS = {
    "handle_card_arrival": "Handle questions about when a newly ordered card will arrive.",
    "handle_card_not_working": "Handle reports that a physical bank card does not work.",
    "handle_cash_withdrawal_not_recognised": "Handle an unrecognized cash withdrawal.",
    "handle_change_pin": "Handle requests to change a card PIN.",
    "handle_compromised_card": "Handle reports that card details may be compromised.",
    "handle_lost_or_stolen_card": "Handle a card reported lost or stolen.",
    "handle_pending_card_payment": "Handle a card payment that is still pending.",
    "handle_terminate_account": "Handle requests to close a bank account.",
    "handle_transfer_not_received_by_recipient": "Handle a transfer the recipient has not received.",
    "handle_verify_my_identity": "Handle questions about completing identity verification.",
}


def make_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
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


TOOLS = [make_tool(name, description) for name, description in TOOL_DESCRIPTIONS.items()]
CALL_PATTERN = re.compile(
    r"<start_function_call>call:(?P<name>[a-z0-9_]+)"
    r"\{customer_message:<escape>(?P<message>.*?)<escape>\}"
    r"<end_function_call>",
    re.DOTALL,
)


def route(customer_message: str) -> dict[str, object]:
    messages = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": customer_message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )
    match = CALL_PATTERN.search(generated)
    if match is None:
        raise ValueError(f"Model did not produce a complete function call: {generated!r}")
    return {
        "name": match.group("name"),
        "arguments": {"customer_message": match.group("message")},
        "raw_call": match.group(0),
    }


# Keep the saved FP32 weights. See the precision note below.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(route("My card was stolen last night"))
```

Expected result:

```python
{
    "name": "handle_lost_or_stolen_card",
    "arguments": {"customer_message": "My card was stolen last night"},
    "raw_call": "<start_function_call>call:handle_lost_or_stolen_card{customer_message:<escape>My card was stolen last night<escape>}<end_function_call>",
}
```

The model only **selects and describes** a call. Your application must validate
the parsed arguments and dispatch the name through an explicit allow-listed
handler map. Do not use `eval`, `globals()`, or execute model-generated code.

## Observed held-out examples

These are first function calls generated by the published checkpoint, not
hand-written target outputs:

| Customer input | Generated tool |
| --- | --- |
| `I am sick of this damn company and want to close out my account.` | `handle_terminate_account` |
| `My card doesn't work.` | `handle_card_not_working` |
| `My card is gone I think it was stolen` | `handle_lost_or_stolen_card` |
| `There is a withdrawal that isn't mind in the app.` | `handle_cash_withdrawal_not_recognised` |
| `Can i change my PIN at the ATM?` | `handle_change_pin` |
| `What are the steps to verify my identity?` | `handle_verify_my_identity` |
| `Why is my payment pending?` | `handle_pending_card_payment` |

For example:

```text
Input:  Why is my payment pending?
Output: <start_function_call>call:handle_pending_card_payment{customer_message:<escape>Why is my payment pending?<escape>}<end_function_call>
```

FunctionGemma may continue generating after the first
`<end_function_call>` when given a large token budget. The usage helper above
intentionally extracts only the first complete call, which is also how this
experiment computes tool accuracy.

## Loss function and results

TRL supervised fine-tuning uses causal-language-model next-token
cross-entropy, also called negative log-likelihood:

```text
loss = mean(-log P(correct next token | all previous tokens))
```

The verified run used a conversational language-model dataset with
`assistant_only_loss=False`. Therefore, every non-padding token in the rendered
developer prompt, tool declarations, user message, and assistant call
contributed to the loss. Padding tokens were ignored. Exact tool accuracy is a
separate generated task metric; it is not the differentiable training loss.

| Epoch | Training loss | Validation loss | Validation token accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.0524 | 0.0468 | 98.9517% |
| 2 | 0.0421 | **0.0428** | 98.9847% |
| 3 | 0.0293 | 0.0430 | 99.0235% |
| 4 | 0.0186 | 0.0492 | 99.0270% |

Training loss continued falling because the model became more confident on the
800 examples it repeatedly saw. Validation loss reached its minimum after
epoch 2 and then rose because that extra confidence did not generalize equally
to the 200 unseen examples. A few confidently wrong next tokens can raise
cross-entropy even while average token accuracy still inches upward. This is
mild overfitting, and it was one reason to run four epochs rather than stop at a
short smoke test.

The last checkpoint remains published because its application-level metric was
strong: **97/100 exact first-tool selections**, compared with **51/100** for the
untouched base model. We did not measure generated tool accuracy at every saved
epoch, so claiming the epoch-2 checkpoint is better for routing would be
unsupported. A follow-up run should measure that metric each epoch and select
the checkpoint by routing accuracy, with validation loss as a secondary signal.

### Reference training configuration

| Setting | Value |
| --- | --- |
| Training / validation rows | 800 / 200 |
| Intents | 10 of 77 |
| Method | Full supervised fine-tune |
| Epochs / optimizer steps | 4 / 400 |
| Per-device / effective batch | 4 / 8 |
| Learning rate | 5e-5 with cosine decay |
| Maximum sequence length | 1,024 |
| Precision | FP32 master weights with FP16 autocast |
| Seed | 42 |

The reference checkpoint's original script passed `warmup_steps=0.1`, which was
effectively no meaningful warmup. The repository script now calculates the
supported integer warmup count as 10% of optimizer steps (40 steps for the
default run). The table above reports the published run, not an unexecuted
corrected rerun.

See the
[Trackio dashboard](https://huggingface.co/spaces/lxyuan/functiongemma-banking77-trackio)
for curves. This repository also includes TensorBoard event files,
`trainer_state.json`, and `training_metrics.json` with the raw before/after
predictions.

## Precision note

Load the model as shown without forcing `torch_dtype=torch.float16`. In the
verified Hub reload test, the saved FP32 checkpoint produced a valid function
call; forcing all weights to pure FP16 on a T4 produced padding-only output.
Training still used FP16 autocast around FP32 master weights.

## Limitations

- Only ten BANKING77 intents are supported, not all 77.
- Every input is forced toward one of those tools; there is no out-of-scope or
  refusal route.
- Argument quality was not scored separately from tool-name selection.
- The data is short English banking text, not real production traffic.
- Similar, ambiguous, adversarial, multilingual, or unrelated requests may be
  routed incorrectly.
- Do not use this model for financial decisions or without privacy, safety,
  monitoring, fallback, and human-review controls.

The BANKING77 mirror describes the dataset as CC BY 4.0. FunctionGemma weights
remain subject to the Gemma terms.
