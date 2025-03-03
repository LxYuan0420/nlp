# -*- coding: utf-8 -*-
"""Semantic Search using Multlingual Distilled Universal Sentence Encoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AOpeByc28RwWfexr6wG9kAA3lrOxCBH4

### 0. Install Sentence Transformer Library
"""

# Install the library using pip
!pip install sentence-transformers

"""### 1. Load the Multilingual DisilBERT model"""

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

"""### 2. Perform Semantic Search"""

# A corpus is a list with documents split by sentences.

english_corpus = [
             'Absence of sanity', 
             'Lack of saneness',
             'A man is eating food.',
             'A man is eating a piece of bread.',
             'The girl is carrying a baby.',
             'A man is riding a horse.',
             'A woman is playing violin.',
             'Two men pushed carts through the woods.',
             'A man is riding a white horse on an enclosed ground.',
             'A monkey is playing drums.',
             'A cheetah is running behind its prey.']

# Each sentence is encoded as a 1-D vector with 768 columns
corpus_embeddings = model.encode(english_corpus, convert_to_tensor=True)

english_queries = ['Nobody has sane thoughts']

for query in english_queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
    hits = hits[0] # Get hits for the first query

    for hit in hits:
        print(f"Matched: {english_corpus[hit['corpus_id']]} with score of {hit['score']}")

chinese_corpus = ['缺乏理智', '缺乏理智'，
              '一个人正在吃东西。'，
              '一个人正在吃一块面包。'，
              '这个女孩怀了一个婴儿。'，
              '一个人在骑马。'，
              '一个女人在拉小提琴。'，
              '两个人推着推车穿过树林。'，
              '一个人在封闭的地面上骑着一匹白马。'，
              '一只猴子正在打鼓。'，
              '一只猎豹正在追赶它的猎物。']

# chinese: translated english corpus and queries

chinese_corpus = ['缺乏理智',
              '缺乏理智',
              '一个人正在吃东西。',
              '一个人正在吃一块面包。',
              '这个女孩怀了一个婴儿。',
              '一个人在骑马。',
              '一个女人在拉小提琴。',
              '两个人推着推车穿过树林。',
              '一个人在封闭的地面上骑着一匹白马。',
              '一只猴子正在打鼓。',
              '一只猎豹正在追赶它的猎物。']

corpus_embeddings  = model.encode(chinese_corpus, convert_to_tensor=True)

chinese_queries = ['没有人有理智的想法', # en: Nobody has sane thoughts
                   '澳大利亚最大的城市悉尼处于 COVID-19 为期 2 周的严格封锁状态' # en: Sydney, Australia's largest city, in 2-week hard COVID-19 lockdown
                   ]

for query in chinese_queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
    hits = hits[0]

    print(f"Query: {query}")
    for hit in hits:
        print(f"Matched: {chinese_corpus[hit['corpus_id']]}  (en: {english_corpus[hit['corpus_id']]}) with score of {hit['score']}")
    print('-'*50)

"""### Reference 


- [Sentence Transformer library](https://github.com/UKPLab/sentence-transformers)
- [Paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.aclweb.org/anthology/D19-1410.pdf)


"""