#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "model2vec",
#     "vicinity",
#     "numpy",
# ]
# ///

"""
This script demonstrates advanced uses of the
“minishlab/potion-multilingual-128M” static embedding model:

1. word_neighbors:       Show nearest neighbors of a given token
2. analogy:              a→b :: c→?
3. translate_en_zh:      Match English sentences to Chinese
4. odd_one_out:          Find the semantic outlier in a word list
5. semantic_search:      Find best candidate for a query
6. sentence_similarity:  Compute full pairwise similarity matrix

Run with:
    uv run model2vec_example.py

Example Output:
Reading inline script metadata from `model2vec_example.py`

Nearest neighbors for 'dog':
  ▁dog             (score 0.0000)
  ▁Dog             (score 0.0726)
  ▁dogs            (score 0.0887)
  犬                (score 0.1444)
  狗狗               (score 0.1515)
  狗                (score 0.1792)
  ▁犬               (score 0.1803)
  ▁honden          (score 0.1872)

Nearest neighbors for '猫':
  ▁猫               (score 0.0000)
  猫                (score 0.0708)
  貓                (score 0.1230)
  ▁cats            (score 0.1778)
  ▁katt            (score 0.1927)
  ▁katten          (score 0.2127)
  ▁katzen          (score 0.2272)
  แมว              (score 0.2292)

Analogy: 'man'→'king' :: 'woman'→'▁queen'

Analogy: '北京'→'中国' :: 'Paris'→'▁Paris'

English → Chinese translation:
  'Good morning' → '早上好'  (score 0.671)
  'How are you?' → '你好吗？'  (score 0.366)
  'Thank you very much' → '非常感谢'  (score 0.577)
  'See you later' → '回头见'  (score 0.176)

Odd-one-out detection:
  In ['apple', 'banana', 'orange', '飞机'], the odd one out is: '飞机'

Semantic search for: 'What should I wear in the rain?'
  Best match: 'You should wear a waterproof raincoat.'  (score 0.7941)

Pairwise sentence similarity:
        0       1       2       3       4       5
 0    1.000   0.877   0.097   0.145  -0.099  -0.003  │ 'I saw the man with the telescope'
 1    0.877   1.000   0.079   0.110  -0.079  -0.015  │ 'Using a telescope, I saw the man'
 2    0.097   0.079   1.000   0.758   0.259   0.271  │ "She said he didn't steal the money"
 3    0.145   0.110   0.758   1.000   0.295   0.249  │ "She didn't say he stole the money"
 4   -0.099  -0.079   0.259   0.295   1.000   0.452  │ 'The bank approved my loan'
 5   -0.003  -0.015   0.271   0.249   0.452   1.000  │ 'The river bank was eroded'
"""

from model2vec import StaticModel
from vicinity import Vicinity
import numpy as np

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
index = Vicinity.from_vectors_and_items(model.embedding, model.tokens)

def word_neighbors(token: str, k: int = 8):
    """
    Print the top-k nearest tokens to `token`.
    """
    vector = model.encode(token)
    neighbors, scores = zip(*index.query(vector, k=k)[0])
    print(f"\nNearest neighbors for {token!r}:")
    for word, score in zip(neighbors, scores):
        print(f"  {word:15s}  (score {score:.4f})")

def analogy(a: str, b: str, c: str, k: int = 5):
    """
    Solve analogy: a is to b as c is to ?
    """
    vec = model.encode(b) - model.encode(a) + model.encode(c)
    answers, _ = zip(*index.query(vec, k=k)[0])
    print(f"\nAnalogy: {a!r}→{b!r} :: {c!r}→{answers[0]!r}")

def translate_en_zh(english: list[str], chinese: list[str]):
    """
    For each English sentence, find its best-matching Chinese counterpart.
    """
    print("\nEnglish → Chinese translation:")
    vec_en = model.encode(english)
    vec_zh = model.encode(chinese)
    sim_matrix = vec_en @ vec_zh.T
    for i, src in enumerate(english):
        j = int(sim_matrix[i].argmax())
        score = sim_matrix[i, j]
        print(f"  {src!r} → {chinese[j]!r}  (score {score:.3f})")

def odd_one_out(words: list[str]):
    """
    Identify the word least semantically similar to the rest.
    """
    print("\nOdd-one-out detection:")
    vecs = model.encode(words)
    centroid = vecs.mean(axis=0)
    distances = np.linalg.norm(vecs - centroid, axis=1)
    outlier = words[int(np.argmax(distances))]
    print(f"  In {words!r}, the odd one out is: {outlier!r}")

def semantic_search(query: str, candidates: list[str]):
    """
    Find which candidate best matches the query.
    """
    print(f"\nSemantic search for: {query!r}")
    qvec = model.encode(query)
    cvecs = model.encode(candidates)
    scores = cvecs @ qvec
    best_idx = int(scores.argmax())
    print(f"  Best match: {candidates[best_idx]!r}  (score {scores[best_idx]:.4f})")

def sentence_similarity(sentences: list[str]):
    """
    Compute and display the full pairwise similarity matrix.
    """
    print("\nPairwise sentence similarity:")
    vecs = model.encode(sentences)
    sim = vecs @ vecs.T
    n = len(sentences)
    # Header
    header = " " * 5 + "".join(f"{i:^8d}" for i in range(n))
    print(header)
    for i, row in enumerate(sim):
        row_str = "".join(f"{row[j]:8.3f}" for j in range(n))
        print(f"{i:2d} {row_str}  │ {sentences[i]!r}")


if __name__ == "__main__":
    # Nearest neighbors
    word_neighbors("dog")
    word_neighbors("猫")  # Chinese for “cat”

    # Analogies
    analogy("man", "king", "woman")
    analogy("北京", "中国", "Paris")  # Beijing:China :: Paris:?


    # Translation EN → ZH
    english_sentences = [
        "Good morning",
        "How are you?",
        "Thank you very much",
        "See you later"
    ]
    chinese_sentences = [
        "早上好",
        "你好吗？",
        "非常感谢",
        "回头见"
    ]
    translate_en_zh(english_sentences, chinese_sentences)

    # Odd-one-out
    odd_one_out(["apple", "banana", "orange", "飞机"])  # the odd one is “airplane”

    # Semantic search
    semantic_search(
        "What should I wear in the rain?",
        [
            "You should wear a waterproof raincoat.",
            "It's bright and sunny today.",
            "Make sure to pack your sunscreen.",
            "Don't forget your umbrella.",
        ]
    )

    # Sentence similarity over a set
    sentences = [
        "I saw the man with the telescope",
        "Using a telescope, I saw the man",
        "She said he didn't steal the money",
        "She didn't say he stole the money",
        "The bank approved my loan",
        "The river bank was eroded"
    ]
    sentence_similarity(sentences)
