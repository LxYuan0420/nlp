#!/usr/bin/env python3
"""Detect hallucination with LettuceDetect v2.

A self-contained tutorial for grounded-answer verification.

This script demonstrates the detector only. Each built-in example already
contains a user question, trusted context, and answer to verify. The detector
does not generate the answer; it checks whether the answer is grounded in the
context and reports unsupported spans.

The goal is to make the detector's purpose visible with simple examples:

    question + trusted context + candidate answer
        -> LettuceDetect v2
        -> supported answer or highlighted hallucinated spans

Prerequisite: this script does not start the model server. Start the model first
with vLLM or vLLM Metal using the same command:

    vllm serve KRLabsOrg/lettucedect-v2-qwen-2b \\
        --served-model-name lettucedect-v2-qwen-2b \\
        --port 8000

Then run this tutorial in Terminal 2:

    python3 scripts/lettucedetect_v2_demo.py

Expected output shape:

    Detect hallucination with LettuceDetect v2
    Server: http://localhost:8000/v1
    Model:  lettucedect-v2-qwen-2b

    The examples below keep the answer fixed and ask the model to verify it.
    ============================================================================
    Example: supported
    ----------------------------------------------------------------------------
    Question:
    What is the capital of France?

    Trusted context:
    France is a country in Europe. Its capital is Paris.

    Answer to verify:
    The capital of France is Paris.

    Result: SUPPORTED — no unsupported spans detected.
    ============================================================================
    Example: unsupported
    ----------------------------------------------------------------------------
    Question:
    What is the capital of France and its population?

    Trusted context:
    France is a country in Europe. Its capital is Paris.

    Answer to verify:
    The capital of France is Paris. Its population is 2 million.

    Result: HALLUCINATION DETECTED — 1 span(s).

      Span 1: '2 million'
      Type: unsupported_addition / claim
      Answer offsets: 50:59
    ============================================================================
    Example: contradiction
    ----------------------------------------------------------------------------
    Question:
    What is the capital of France?

    Trusted context:
    France is a country in Europe. Its capital is Paris.

    Answer to verify:
    The capital of France is London.

    Result: HALLUCINATION DETECTED — 1 span(s).

      Span 1: 'London'
      Type: contradiction / entity
      Answer offsets: 25:31

    ============================================================================
    Tutorial complete.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.request import Request, urlopen


MODEL = "lettucedect-v2-qwen-2b"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

SYSTEM_PROMPT = """You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations.

Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory.

Categories (the kinds of unsupported span):
- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)
- fabricated_reference: an entity, name, identifier, or section that is absent from the context
- unsupported_addition: a claim, detail, or behavior the context never states

Subcategories:
- entity: a wrong or invented name, entity, or object
- temporal: an incorrect date, time, duration, or ordering
- numerical: an incorrect number, quantity, or amount
- value: a wrong value, setting, or attribute value
- relational: an incorrect relationship or association between things
- identifier: an invented identifier or name not found in the context
- section: a reference to a section, part, or location that does not exist
- attribute: an invented or incorrect attribute or property
- claim: an added factual claim the context does not support
- behavior: an added or changed action or behavior the context never states
- elaboration: extra detail or elaboration beyond what the context supports
- subjective: an unsupported subjective or evaluative statement
- unspecified: unsupported, with no more specific subtype

Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}."""

EXAMPLES = {
    "supported": {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Its capital is Paris.",
        "answer": "The capital of France is Paris.",
    },
    "unsupported": {
        "question": "What is the capital of France and its population?",
        "context": "France is a country in Europe. Its capital is Paris.",
        "answer": "The capital of France is Paris. Its population is 2 million.",
    },
    "contradiction": {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Its capital is Paris.",
        "answer": "The capital of France is London.",
    },
}


def call_detector(base_url: str, model: str, question: str, context: str, answer: str) -> str:
    user_prompt = (
        f"User request: {question}\n\n{context}\n\nAnswer to verify:\n{answer}"
    )
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result: dict[str, Any] = json.load(response)
    return result["choices"][0]["message"]["content"]


def parse_result(content: str, answer: str) -> dict[str, Any]:
    start = content.index("{")
    result, _ = json.JSONDecoder().raw_decode(content[start:])
    spans = result["hallucinated_spans"]

    for span in spans:
        span["start"] = answer.find(span["text"])
        span["end"] = span["start"] + len(span["text"])
    return result


def print_example(name: str, case: dict[str, str], result: dict[str, Any]) -> None:
    print("=" * 76)
    print(f"Example: {name}")
    print("-" * 76)
    print(f"Question:\n{case['question']}\n")
    print(f"Trusted context:\n{case['context']}\n")
    print(f"Answer to verify:\n{case['answer']}\n")

    spans = result["hallucinated_spans"]
    if not spans:
        print("Result: SUPPORTED — no unsupported spans detected.")
        return

    print(f"Result: HALLUCINATION DETECTED — {len(spans)} span(s).")
    for index, span in enumerate(spans, start=1):
        print(f"\n  Span {index}: {span['text']!r}")
        print(f"  Type: {span['category']} / {span['subcategory']}")
        print(f"  Answer offsets: {span['start']}:{span['end']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "example",
        choices=("all", *EXAMPLES),
        nargs="?",
        default="all",
        help="run one example, or all examples (default)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"vLLM OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"served detector model name (default: {MODEL})",
    )
    args = parser.parse_args()

    names = EXAMPLES if args.example == "all" else {args.example: EXAMPLES[args.example]}
    print("Detect hallucination with LettuceDetect v2")
    print(f"Server: {args.base_url}")
    print(f"Model:  {args.model}")
    print("\nThe examples below keep the answer fixed and ask the model to verify it.")

    for name, case in names.items():
        raw_result = call_detector(args.base_url, args.model, **case)
        result = parse_result(raw_result, case["answer"])
        print_example(name, case, result)

    print("\n" + "=" * 76)
    print("Tutorial complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
