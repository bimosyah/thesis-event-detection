import argparse
import numpy as np
import os
import pandas as pd

from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from utils import load_config, set_seed

cfg = load_config()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_and_eval(train_csv, val_csv, test_csv, output_dir, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer"]["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["transformer"]["model_name"], num_labels=2)

    train_df = pd.read_csv(train_csv)[["message", "lable"]]
    val_df = pd.read_csv(val_csv)[["message", "lable"]]
    test_df = pd.read_csv(test_csv)[["message", "lable"]]

    # Rename 'lable' to 'labels' for Trainer compatibility
    train_df = train_df.rename(columns={'lable': 'labels'})
    val_df = val_df.rename(columns={'lable': 'labels'})
    test_df = test_df.rename(columns={'lable': 'labels'})

    ds_train = Dataset.from_pandas(train_df)
    ds_val = Dataset.from_pandas(val_df)
    ds_test = Dataset.from_pandas(test_df)

    def preprocess(batch):
        return tokenizer(batch["message"], truncation=True, padding="max_length",
                         max_length=cfg["transformer"]["max_length"])

    ds_train = ds_train.map(preprocess, batched=True)
    ds_val = ds_val.map(preprocess, batched=True)
    ds_test = ds_test.map(preprocess, batched=True)
    ds_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    ds_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"seed_{seed}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg["transformer"]["batch_size"],
        per_device_eval_batch_size=cfg["transformer"]["eval_batch_size"],
        learning_rate=cfg["transformer"]["learning_rate"],
        num_train_epochs=cfg["transformer"]["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )

    trainer.train()
    preds = trainer.predict(ds_test)
    metrics = preds.metrics
    # save predictions for stats
    preds_lables = np.argmax(preds.predictions, axis=1)
    out_df = pd.DataFrame({"message": test_df["message"], "lable": test_df["labels"], "pred": preds_lables})
    os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(os.path.join(output_dir, f"preds_seed_{seed}.csv"), index=False)
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, f"metrics_seed_{seed}.csv"), index=False)
    print("Done seed", seed, "metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_eval(args.train, args.val, args.test, args.out, seed=args.seed)
