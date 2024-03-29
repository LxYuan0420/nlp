# -*- coding: utf-8 -*-
"""IMDb Review Classification using DistilBert HF Tensorflow Trainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k1zxluUsy-7xDIbp1RLpHWhfJc0N6iJD

### 0. Install Libraries
"""

!pip install -U transformers
!pip install sentencepiece

# Restart kernel after installation: Runtime -> Restart runtime

from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

"""### 1. Download and load dataset"""

# Download dataset
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz

!ls ac*/

!ls ac*/train/

!cat ac*/train/pos/0_9.txt

def read_imdb_split(split_dir):
    """Helper function to read text from txt files located in 
    `split_dir/pos/*.txt` or `split_dir/neg/*.txt`

    @param split_dir: path to train or test directory that contains both pos and neg subdirectory. 

    @returns texts: List of str where each element is a feature (text)
    @returns labels: List of int where each element is a label (positive:1, negative: 0)
    """

    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ['pos', 'neg']:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(1 if label_dir == 'pos' else 0)

    return texts, labels

train_texts, train_labels = read_imdb_split("aclImdb/train")
test_texts, test_labels = read_imdb_split("aclImdb/test")

print(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")
print(f"Train size: {len(train_labels)} | Test size: {len(test_labels)}")

"""### 2. Split and tokenize dataset"""

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

print(train_encodings.keys())

# Each key's value is a list of list. Something like:
# 'input_ids': [[1,2,3], [4,5,6]]
# Refer to the __getitem__ method in the IMDbDataset subclass to see how to access to each element individually.

"""### 3. Create tf Dataset"""

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)) 
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))        
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

"""### 4. Prepare training arguments and start training"""

training_args = TFTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    #logging_steps=10,
    save_strategy="epoch",
    logging_strategy="epoch",
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")


trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

"""**Training completed. Do not forget to share your model on huggingface.co/models =)**

```
TrainOutput(global_step=7500, training_loss=0.20122720197954524, metrics={'train_runtime': 3150.3136, 'train_samples_per_second': 19.046, 'train_steps_per_second': 2.381, 'total_flos': 1.23411474432e+16, 'train_loss': 0.20122720197954524, 'epoch': 3.0})```

### 5. Predicts on Test set
"""

preds = trainer.predict(test_dataset)

predictions = preds[0].argmax(-1)

from sklearn.metrics import classification_report

print(classification_report(preds[1], # labels from test_dataset
                            predictions, 
                            ))

"""### 6. Save model"""

trainer.save_model("./models")

!ls models/

#config.json tf_model.h5

"""### 7. Inference using pipeline"""

from transformers import pipeline

review_pipeline = pipeline("text-classification", model="./models", tokenizer=tokenizer, return_all_scores=True)

positive_test_case = "Awesome movie. Love it so much!"

predictions = review_pipeline(positive_test_case)

predictions

#[[{'label': 'LABEL_0', 'score': 0.0015208885306492448},
#  {'label': 'LABEL_1', 'score': 0.9984791278839111}]]

negative_test_case = "Bad movie and storyline. I hate it so much!"

predictions = review_pipeline(negative_test_case)

predictions
#[[{'label': 'LABEL_0', 'score': 0.9911043643951416},
# {'label': 'LABEL_1', 'score': 0.008895594626665115}]]

"""### Inference using model.from_pretrained()"""

reload_model = TFDistilBertForSequenceClassification.from_pretrained("./models")

test_cases = [positive_test_case, negative_test_case]
encoded_test_cases = tokenizer(test_cases, truncation=True, padding=True, return_tensors='tf')

outputs = reload_model(encoded_test_cases)

predictions = tf.argmax(outputs.logits, axis=-1)

for sent, pred in zip(test_cases, predictions):
    print(f"Sentence: {sent} | Predicted: {'Positive' if pred == 1 else 'Negative'}")

"""### Reference


1.   Tokenizer: https://huggingface.co/transformers/main_classes/tokenizer.html
2.   Pipeline: https://huggingface.co/transformers/main_classes/pipelines.html#the-pipeline-abstraction
3.   Load model after trainer.train(): https://discuss.huggingface.co/t/how-to-test-my-text-classification-model-after-training-it/6689/2
4.   Fine-tuning with custom datasets: https://huggingface.co/transformers/master/custom_datasets.html
5.   trainer.predict(): https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.predict

"""