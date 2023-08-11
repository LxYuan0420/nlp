"""
A SpanMarker Model Training Script. You can manually edit the code to modify
the dataset, model, and parameters. Run the script using the following
command:

(env)$ python spanmarker_train.py | tee -a spanmarker_train.log

"""

from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer

from span_marker import SpanMarkerModel, Trainer


def main() -> None:
    # Load the dataset, ensure "tokens" and "ner_tags" columns, and get a list of labels
    dataset = "Babelscape/multinerd"
    train_dataset = load_dataset(dataset, split="train")
    eval_dataset = (
        load_dataset(dataset, split="validation").shuffle().select(range(3000))
    )
    labels = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-ANIM",
        "I-ANIM",
        "B-BIO",
        "I-BIO",
        "B-CEL",
        "I-CEL",
        "B-DIS",
        "I-DIS",
        "B-EVE",
        "I-EVE",
        "B-FOOD",
        "I-FOOD",
        "B-INST",
        "I-INST",
        "B-MEDIA",
        "I-MEDIA",
        "B-MYTH",
        "I-MYTH",
        "B-PLANT",
        "I-PLANT",
        "B-TIME",
        "I-TIME",
        "B-VEHI",
        "I-VEHI",
    ]

    # Initialize a SpanMarker model using a pretrained BERT-style encoder
    # Note: Not all encoders work though, they must allow for position_ids as
    # an input argument, which disqualifies DistilBERT, T5, DistilRoBERTa,
    # ALBERT & BART. Furthermore, using uncased models is generally not
    # recommended, as the capitalisation can be very useful to find named
    # entities.
    model_name = "bert-base-multilingual-cased"

    # Tokenizer and data collator will be automatically loaded in the
    # SpanMarker Model Class: https://github.com/tomaarsen/SpanMarkerNER/blob/main/span_marker/modeling.py#L95
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SpanMarkerModel.from_pretrained(
        model_name,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=8,
    )

    # Prepare the 🤗 transformers training arguments
    args = TrainingArguments(
        output_dir="span-marker-bert-base-multilingual-cased-multinerd",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        push_to_hub=True,
    )

    # Initialize the trainer using our model, training args & dataset, and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.push_to_hub()
    model.tokenizer.push_to_hub("span-marker-bert-base-multilingual-cased-multinerd")
    # trainer.save_model("models/span_marker_mbert_base_multinerd/checkpoint-final")

    test_dataset = load_dataset(dataset, split="test")
    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", metrics)

    trainer.create_model_card(language="multilingual", license="apache-2.0")


if __name__ == "__main__":
    main()
