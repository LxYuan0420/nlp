# -*- coding: utf-8 -*-
"""ConllPP NER Classification using DistilBERT HF Pytorch Script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1luB7ml7YmR9QHhmb1IP9oik_29wDQR5N

### 0. Install and load library
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/huggingface/transformers
# %cd transformers
!python setup.py install

"""### 1. Fine-tuning DistillBERT model on CONLLPP NER dataset.

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

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/transformers/examples/pytorch/token-classification"
!pip install -r requirements.txt

!python run_ner.py --help

!python run_ner.py \
  --task_name ner \
  --model_name_or_path distilbert-base-cased \
  --label_all_tokens True \
  --return_entity_level_metrics True \
  --dataset_name conllpp \
  --output_dir /tmp/distilbert-base-cased-finetuned-conllpp-english_pt \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy epoch \
  --num_train_epochs 10

"""***** train metrics *****
  epoch                    =       10.0
  train_loss               =     0.0274
  train_runtime            = 0:17:37.42
  train_samples            =      14041
  train_samples_per_second =    132.785
  train_steps_per_second   =     16.606
07/25/2021 06:48:02 - INFO - __main__ - *** Evaluate ***
[INFO|trainer.py:522] 2021-07-25 06:48:02,204 >> The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: chunk_tags, tokens, ner_tags, pos_tags, id.
[INFO|trainer.py:2165] 2021-07-25 06:48:02,206 >> ***** Running Evaluation *****
[INFO|trainer.py:2167] 2021-07-25 06:48:02,206 >>   Num examples = 3250
[INFO|trainer.py:2170] 2021-07-25 06:48:02,206 >>   Batch size = 8
 99% 404/407 [00:05<00:00, 67.14it/s]07/25/2021 06:48:09 - INFO - datasets.metric - Removing /root/.cache/huggingface/metrics/seqeval/default/default_experiment-1-0.arrow
100% 407/407 [00:06<00:00, 58.90it/s]
***** eval metrics *****
  epoch                   =       10.0
  eval_LOC_f1             =     0.9602
  eval_LOC_number         =       3635
  eval_LOC_precision      =     0.9606
  eval_LOC_recall         =     0.9598
  eval_MISC_f1            =     0.8862
  eval_MISC_number        =       1480
  eval_MISC_precision     =      0.878
  eval_MISC_recall        =     0.8946
  eval_ORG_f1             =     0.9134
  eval_ORG_number         =       2702
  eval_ORG_precision      =     0.9079
  eval_ORG_recall         =     0.9189
  eval_PER_f1             =     0.9554
  eval_PER_number         =       3329
  eval_PER_precision      =     0.9561
  eval_PER_recall         =     0.9546
  eval_loss               =     0.1273
  eval_overall_accuracy   =     0.9837
  eval_overall_f1         =     0.9375
  eval_overall_precision  =     0.9353
  eval_overall_recall     =     0.9397
  eval_runtime            = 0:00:06.92
  eval_samples            =       3250
  eval_samples_per_second =    468.979
  eval_steps_per_second   =     58.731
07/25/2021 06:48:09 - INFO - __main__ - *** Predict ***
[INFO|trainer.py:522] 2021-07-25 06:48:09,141 >> The following columns in the test set  don't have a corresponding argument in `DistilBertForTokenClassification.forward` and have been ignored: chunk_tags, tokens, ner_tags, pos_tags, id.
[INFO|trainer.py:2165] 2021-07-25 06:48:09,165 >> ***** Running Prediction *****
[INFO|trainer.py:2167] 2021-07-25 06:48:09,165 >>   Num examples = 3453
[INFO|trainer.py:2170] 2021-07-25 06:48:09,165 >>   Batch size = 8
 99% 426/432 [00:04<00:00, 103.25it/s]07/25/2021 06:48:15 - INFO - datasets.metric - Removing /root/.cache/huggingface/metrics/seqeval/default/default_experiment-1-0.arrow
***** predict metrics *****
  predict_LOC_f1             =     0.9239
  predict_LOC_number         =       2991
  predict_LOC_precision      =     0.9167
  predict_LOC_recall         =     0.9311
  predict_MISC_f1            =     0.7755
  predict_MISC_number        =       1319
  predict_MISC_precision     =     0.7995
  predict_MISC_recall        =     0.7528
  predict_ORG_f1             =     0.8992
  predict_ORG_number         =       3629
  predict_ORG_precision      =     0.8942
  predict_ORG_recall         =     0.9041
  predict_PER_f1             =      0.937
  predict_PER_number         =       3006
  predict_PER_precision      =     0.9458
  predict_PER_recall         =     0.9285
  predict_loss               =     0.2619
  predict_overall_accuracy   =     0.9714
  predict_overall_f1         =     0.9018
  predict_overall_precision  =     0.9037
  predict_overall_recall     =        0.9
  predict_runtime            = 0:00:06.73
  predict_samples_per_second =    512.896
  predict_steps_per_second   =     64.168
100% 432/432 [00:07<00:00, 61.41it/s]

### 2. Inference using pipeline
"""

!ls /tmp/distilbert-base-cased-finetuned-conllpp-english_pt

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

# restart runtime if having problem importing pipeline
from transformers import pipeline
distilbert_ner = pipeline('ner', model="/tmp/distilbert-base-cased-finetuned-conllpp-english_pt/", tokenizer="distilbert-base-cased", aggregation_strategy='first')

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
MISC  as -
MISC  as Malaysian
PER   as Muhyiddin
PER   as Yassin
PER   as Umno
PER   as Amid
PER   as Tan
PER   as Sri
PER   as Muhyiddin
ORG   as Umno
ORG   as Perikatan Nasional
ORG   as PN
ORG   as Parti Pribumi Bersatu Malaysia
PER   as Umno
"""