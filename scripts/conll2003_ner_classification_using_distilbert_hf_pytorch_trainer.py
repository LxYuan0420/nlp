# -*- coding: utf-8 -*-
"""Conll2003 NER Classification using DistilBERT HF Pytorch Trainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1583tIXkf9VI2-5rKK3mKuXCN39EW1A7T

### 0. Install and load library
"""

#Restart kernel after installation: Runtime -> Restart runtime

#!pip install -U transformers sentencepiece datasets seqeval

import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np

task = 'ner'
model_checkpoint = 'distilbert-base-uncased'
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric('seqeval')

"""### 1. Load and preprocess dataset """

dataset = load_dataset("conll2003")

# dataset object itself is DatasetDict, which contains train/val/test set
dataset

# take a quick look on one training sample
dataset['train'][0]

label_list = dataset['train'].features['ner_tags'].feature.names

id2label = {idx:label for idx, label in enumerate(label_list)}
label2id = {label:idx for idx, label in id2label.items()}

print(f"Label list: {label_list}")
print(f"id2label: {id2label}")
print(f"Num of label: {len(label_list)}")

example = dataset['train'][420]
print(f"Original sentence token list: {example['tokens']}")

tokenized_input = tokenizer(example['tokens'], is_split_into_words=True)
print(f"tokenized input ids: {tokenized_input['input_ids']}")

tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
print(f"tokenized input: {tokens}")

# to keep track with subtokens from the same word: Huggingface -> Hugg, ##ing, ##face
# in this case: Lathwell -> la, ##th, ##well
print(f"tokenized input word_ids: {tokenized_input.word_ids()}")

# Lathwell is split into 3 subtokens and their word_ids are 4
# We can re-use that word_ids to get the correct ner tag
# in this case the fourth idx of ner_tags ~= 2 
print(f"Orignal ner tag: {example['ner_tags']}")

# if True: label subtokens with the ner tag
# if False: label the first subtoken only and label the rest of subtoken as -100
label_all_token = True

def tokenize_and_align_labels(examples):
    # examples is batch of inputs 
    tokenized_input = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

    labels = []

    for sent_idx, sent_label in enumerate(examples[f'{task}_tags']):

        word_ids = tokenized_input.word_ids(batch_index=sent_idx)
        previous_word_idx = None

        # aligned label ids for current sent idx
        label_ids = []
        for word_idx in word_ids:
            # for special token: [CLS] [SEP]
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(sent_label[word_idx])
            else:
                label_ids.append(sent_label[word_idx] if label_all_token else -100)

            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_input['labels'] = labels
    return tokenized_input

"""Then we will need a data collator that will batch our processed examples together while applying padding to make them all the same size (each pad will be padded to the length of its longest example). There is a data collator for this task in the Transformers library, that not only pads the inputs, but also the labels"""

print(f"example {task} tag: {example[f'{task}_tags']}")

labels = [label_list[i] for i in example[f'{task}_tags']]
print(f"labels: {labels}")

metric.compute(predictions=[labels], references=[labels])

"""So we will need to do a bit of post-processing on our predictions:
- select the predicted index (with the maximum logit) for each token
- convert it to its string label
- ignore everywhere we set a label of -100

The following function does all this post-processing on the result of `Trainer.evaluate` (which is a namedtuple containing predictions and labels) before applying the metric:
"""

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

"""#### Tokenized dataset"""

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

"""### 2. Define TrainingArguments and Trainer"""

# https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id)

args = TrainingArguments(
    f"tesk-{task}",
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    logging_strategy='epoch',
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# predicts on test set

preds = trainer.predict(tokenized_dataset['test'])

preds = (preds[0], preds[1])

results = compute_metrics(preds)
results

# differet way to get test results
predictions, labels, _ = trainer.predict(tokenized_dataset['test'])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

results = metric.compute(predictions=true_predictions, references=true_labels)
results

# save model
trainer.save_model("./models")

"""### 2. Inference using pipeline"""

!ls models/

from transformers import pipeline
distilbert_ner = pipeline('ner', model="./models", aggregation_strategy='first')

article = """
KUALA LUMPUR - Malaysian Prime Minister Muhyiddin Yassin's party said on Thursday (July 8) that his government would continue to function despite Umno withdrawing its backing. 
Amid uncertainty over whether Tan Sri Muhyiddin continues to command majority support without Umno, the largest party in the Perikatan Nasional (PN) ruling pact, 
his Parti Pribumi Bersatu Malaysia said Umno's decision "had no effect on the workings of government"."""


results = distilbert_ner(article)

print("Predicted:")
for tag in results:
    print(f"{tag['entity_group']:<5} as {tag['word']}")


"""
Predicted:
LOC   as kuala lumpur
MISC  as malaysian
PER   as muhyiddin yassin
ORG   as umno
PER   as tan
ORG   as sri
PER   as muhyiddin
ORG   as umno
ORG   as perikatan nasional
ORG   as pn
ORG   as parti pribumi bersatu malaysia
ORG   as umno
"""
