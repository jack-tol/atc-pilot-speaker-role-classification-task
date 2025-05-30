import warnings, numpy as np, torch, random
from datasets import load_dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    logging as hf_logging
)
from sklearn.metrics import precision_recall_fscore_support

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

dataset = load_dataset("USERNAME/DATASET_NAME")
label2id = {"atc": 0, "pilot": 1}
dataset = dataset.map(lambda ex: {"label": label2id[ex["class"]]})

tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
tokenized = dataset.map(lambda b: tokenizer(b["text"], padding=True, truncation=True, max_length=256), batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor): logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = (preds == labels).mean()
    return {k: float(v) for k, v in {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}.items()}

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./atc-pilot-speaker-role-classification",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=600,
        save_total_limit=10,
        num_train_epochs=4,
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
        disable_tqdm=False
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

eval_results = trainer.evaluate(eval_dataset=tokenized["validation"])
print("\nFinal Evaluation on Validation Set:")
for k, v in eval_results.items():
    print(f"{k:<20}: {v:.4f}")