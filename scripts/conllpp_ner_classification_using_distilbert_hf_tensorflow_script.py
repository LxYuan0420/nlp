# -*- coding: utf-8 -*-
"""ConllPP NER Classification using DistilBERT HF Tensorflow Script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MrCAwKZPFFzm-As8OQ03SjaS5ytXYaYH

### 0. Install and load library
"""

#Restart kernel after installation: Runtime -> Restart runtime

#!pip install -U sentencepiece datasets
#!pip install -U transformers sentencepiece datasets seqeval

import transformers

"""### 1. Fine-tuning DistillBERT model on CONLLPP NER dataset.



"""

!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/tensorflow/token-classification/run_ner.py

"""
import datasets

dataset = datasets.load_dataset("conllpp")
dataset['train'].features['ner_tags']

>> Sequence(feature=ClassLabel(
    num_classes=9, 
    names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], 
    names_file=None, id=None),
    length=-1, 
    id=None)
"""

!python /content/run_ner.py \
  --model_name_or_path distilbert-base-cased \
  --label_all_tokens True \
  --return_entity_level_metrics True \
  --dataset_name conllpp \
  --output_dir /tmp/distilbert-base-cased-finetuned-conllpp-english_tf \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy epoch \
  --num_train_epochs 10

"""Evaluation metrics:
LOC_precision: 0.9689
LOC_recall: 0.9527
LOC_f1: 0.9607
LOC_number: 3635.0000
MISC_precision: 0.8724
MISC_recall: 0.8919
MISC_f1: 0.8821
MISC_number: 1480.0000
ORG_precision: 0.9130
ORG_recall: 0.9171
ORG_f1: 0.9151
ORG_number: 2702.0000
PER_precision: 0.9438
PER_recall: 0.9631
PER_f1: 0.9533
PER_number: 3329.0000
overall_precision: 0.9347
overall_recall: 0.9391
overall_f1: 0.9369
overall_accuracy: 0.9841
Configuration saved in /tmp/distilbert-base-cased-finetuned-conllpp-english_tf/config.json
Model weights saved in /tmp/distilbert-base-cased-finetuned-conllpp-english_tf/tf_model.h5

### 2. Inference using pipeline
"""

!ls /tmp/distilbert-base-cased-finetuned-conllpp-english_tf

# Update NER label2id and id2label in model config file: config.json
"""
  "id2label": {
    "0": "O",
    "1": "B-PER",
    "2": "i-PER",
    "3": "B-ORG",
    "4": "I-ORG",
    "5": "B-LOC",
    "6": "I-LOC",
    "7": "B-MISC",
    "8": "I-MISC"
  },


  "label2id": {
    "B-LOC": 5,
    "B-MISC": 7,
    "B-ORG": 3,
    "B-PER": 1,
    "I-MISC": 8,
    "I-ORG": 4,
    "I-PER": 2,
    "O": 0
  },

"""

from transformers import pipeline
distilbert_ner = pipeline('ner', model="/tmp/distilbert-base-cased-finetuned-conllpp-english_tf/", tokenizer="distilbert-base-cased", aggregation_strategy='first')

article = """
KUALA LUMPUR - Malaysian Prime Minister Muhyiddin Yassin's party said on Thursday (July 8) that his government would continue to function despite Umno withdrawing its backing. 
Amid uncertainty over whether Tan Sri Muhyiddin continues to command majority support without Umno, the largest party in the Perikatan Nasional (PN) ruling pact, 
his Parti Pribumi Bersatu Malaysia said Umno's decision "had no effect on the workings of government"."""


results = distilbert_ner(article)

print("Predicted:")
for tag in results:
    print(f"{tag['entity_group']:<5} as {tag['word']}")


"""
Output
------
Predicted:
ORG   as KUALA LUMPUR
MISC  as Malaysian
PER   as Muhyiddin
PER   as Yassin
PER   as Umno
PER   as Tan
PER   as Sri
PER   as Muhyiddin
ORG   as Umno
ORG   as Perikatan Nasional
ORG   as PN
ORG   as Parti Pribumi Bersatu Malaysia
PER   as Umno
"""
