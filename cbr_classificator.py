# -*- coding: utf-8 -*-
!pip install -q transformers datasets scikit-learn torch accelerate sentence-transformers

import json
import os
import random
import re
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

import pickle
import shutil

# ============= CONFIGURATION =============
MODEL_NAME = "DeepPavlov/rubert-base-cased"
JSON_PATH = "/content/labeled_data_1.json"
SYNTH_JSON_PATH = "/content/Syntetic_data_cbr.json"
MAX_LENGTH = 128
SEED = 42
STAGE1_DIR = "/content/cbr_stage1_encoder"
STAGE1_BEST_DIR = "/content/cbr_stage1_encoder_best"
STAGE2_DIR = "/content/cbr_stage2_classifier"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("GPU available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= LABEL NORMALIZATION =============
LABEL_NORMALIZATION = {
    "topic": {
        "inflation": "Inflation",
        "Inflation": "Inflation",
        "key_rate": "Interest rate",
        "interest_rate": "Interest rate",
        "Interest rate": "Interest rate",
        "monetary_conditions": "Monetary conditions",
        "Monetary conditions": "Monetary conditions",
        "credit": "Credit",
        "Credit": "Credit",
        "inflation_expectations": "Expectations",
        "expectations": "Expectations",
        "Expectations": "Expectations",
        "economic_activity": "Economic activity",
        "Economic activity": "Economic activity",
        "labor_market": "Labor market",
        "Labor market": "Labor market",
        "external_conditions": "External conditions",
        "External conditions": "External conditions",
        "fiscal_policy": "Fiscal policy",
        "Fiscal policy": "Fiscal policy",
        "exchange_rate": "Exchange rate",
        "Exchange rate": "Exchange rate",
        "financial_stability": "Financial sector & Macroprudential policy",
        "Financial sector & Macroprudential policy": "Financial sector & Macroprudential policy",
        "metadata": "Other",
        "Other": "Other",
    },
    "stance": {
        "forward-looking": "forward-looking",
        "backward-looking": "backward-looking",
        "metadata": "Other",
        "Other": "Other",
    },
    "sentiment": {
        "hawkish": "hawkish",
        "dovish": "dovish",
        "neutral": "neutral",
        "risk-highlighting": "risk-highlighting",
        "confidence-building": "confidence-building",
        "metadata": "Other",
        "Other": "Other",
    }
}

def normalize_label(value, dim):
    if value is None:
        return None
    value = str(value).strip()
    return LABEL_NORMALIZATION[dim].get(value, value)

# ============= DATA LOADING =============
def load_manual_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for task in raw:
        if not task.get("annotations") or len(task["annotations"]) == 0:
            continue

        sentence = task["data"].get("text", "").strip()
        if not sentence:
            continue

        res_list = task["annotations"][0].get("result", [])

        topics = next(
            (r["value"]["choices"] for r in res_list if r["from_name"] == "topic"),
            []
        )
        stance = next(
            (r["value"]["choices"][0] for r in res_list if r["from_name"] == "stance"),
            None
        )
        sentiment = next(
            (r["value"]["choices"][0] for r in res_list if r["from_name"] == "sentiment"),
            None
        )

        if not topics or stance is None or sentiment is None:
            continue

        topic = topics[0]
        topic = normalize_label(topic, "topic")
        stance = normalize_label(stance, "stance")
        sentiment = normalize_label(sentiment, "sentiment")

        samples.append({
            "text": sentence,
            "topic": topic,
            "stance": stance,
            "sentiment": sentiment,
            "source": "manual",
        })

    return samples

def load_synthetic_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for row in raw:
        text = str(row.get("text", "")).strip()
        topic = normalize_label(row.get("topic"), "topic")
        stance = normalize_label(row.get("stance"), "stance")
        sentiment = normalize_label(row.get("sentiment"), "sentiment")

        if not text or topic is None or stance is None or sentiment is None:
            continue

        samples.append({
            "text": text,
            "topic": topic,
            "stance": stance,
            "sentiment": sentiment,
            "source": "synthetic",
        })

    return samples

manual_samples = load_manual_data(JSON_PATH)
synthetic_samples = load_synthetic_data(SYNTH_JSON_PATH)

print("Manual samples:", len(manual_samples))
print("Synthetic samples:", len(synthetic_samples))
print("Manual example:", manual_samples[0])
print("Synthetic example:", synthetic_samples[0])

# ============= LABEL ENCODING =============
all_samples_for_labels = manual_samples + synthetic_samples

topic_le = LabelEncoder()
stance_le = LabelEncoder()
sentiment_le = LabelEncoder()

topic_le.fit([x["topic"] for x in all_samples_for_labels])
stance_le.fit([x["stance"] for x in all_samples_for_labels])
sentiment_le.fit([x["sentiment"] for x in all_samples_for_labels])

print("\nSentiment classes:", sentiment_le.classes_)
print("Number of sentiment classes:", len(sentiment_le.classes_))

# ============= DATA SPLIT =============
manual_texts = [x["text"] for x in manual_samples]
manual_topics = topic_le.transform([x["topic"] for x in manual_samples])
manual_stances = stance_le.transform([x["stance"] for x in manual_samples])
manual_sentiments = sentiment_le.transform([x["sentiment"] for x in manual_samples])

X_train_manual, X_tmp, y_topic_train_manual, y_topic_tmp, y_stance_train_manual, y_stance_tmp, y_sent_train_manual, y_sent_tmp = train_test_split(
    manual_texts,
    manual_topics,
    manual_stances,
    manual_sentiments,
    test_size=0.3,
    random_state=SEED,
    stratify=manual_topics,
)

X_val, X_test, y_topic_val, y_topic_test, y_stance_val, y_stance_test, y_sent_val, y_sent_test = train_test_split(
    X_tmp,
    y_topic_tmp,
    y_stance_tmp,
    y_sent_tmp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_topic_tmp,
)

print("\nManual split sizes:")
print("train:", len(X_train_manual), "val:", len(X_val), "test:", len(X_test))

synthetic_texts = [x["text"] for x in synthetic_samples]
synthetic_topics = topic_le.transform([x["topic"] for x in synthetic_samples])
synthetic_stances = stance_le.transform([x["stance"] for x in synthetic_samples])
synthetic_sentiments = sentiment_le.transform([x["sentiment"] for x in synthetic_samples])

X_train = X_train_manual + synthetic_texts
y_topic_train = np.concatenate([y_topic_train_manual, synthetic_topics])
y_stance_train = np.concatenate([y_stance_train_manual, synthetic_stances])
y_sent_train = np.concatenate([y_sent_train_manual, synthetic_sentiments])

print("Final train size with synthetic:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# Shuffle training data
train_idx = np.arange(len(X_train))
np.random.shuffle(train_idx)

X_train = [X_train[i] for i in train_idx]
y_topic_train = y_topic_train[train_idx]
y_stance_train = y_stance_train[train_idx]
y_sent_train = y_sent_train[train_idx]

# ============= STAGE 1: SENTENCE ENCODER =============
print("\n" + "="*60)
print("STAGE 1: Training Sentence Encoder (sentiment-based)")
print("="*60)

def build_positive_pairs_by_sentiment_balanced(texts, sentiments, target_pairs=10000):
    """Generate balanced positive pairs based on sentiment"""
    groups = defaultdict(list)

    for text, sent in zip(texts, sentiments):
        groups[sent].append(text)

    # Remove duplicates within each group
    for key in groups:
        groups[key] = list(dict.fromkeys(groups[key]))

    # Calculate pairs per class
    n_classes = len(groups)
    pairs_per_class = target_pairs // n_classes

    pairs = []

    for _, items in groups.items():
        if len(items) < 2:
            continue

        class_pairs = []

        # If not enough pairs, take all combinations
        if len(items) * (len(items) - 1) // 2 <= pairs_per_class:
            class_pairs = list(combinations(items, 2))
        else:
            # Otherwise sample randomly
            while len(class_pairs) < pairs_per_class:
                a, b = random.sample(items, 2)
                if (a, b) not in class_pairs and (b, a) not in class_pairs:
                    class_pairs.append((a, b))

        for a, b in class_pairs:
            pairs.append(InputExample(texts=[a, b]))

    return pairs

train_pairs = build_positive_pairs_by_sentiment_balanced(X_train, y_sent_train)
val_pairs = build_positive_pairs_by_sentiment_balanced(X_val, y_sent_val)

print("Train pairs (sentiment only):", len(train_pairs))
print("Val pairs (sentiment only):", len(val_pairs))

def evaluate_embeddings_separation(model, texts, labels, label_names, device='cuda'):
    """
    Evaluate embedding quality:
    - Intra-class similarity (higher is better)
    - Inter-class distance (higher is better)
    """
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_numpy=True, device=device, show_progress_bar=False)

    # Group by class
    class_embeddings = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        class_embeddings[label].append(emb)

    # Intra-class similarity
    intra_similarities = []
    for label, embs in class_embeddings.items():
        if len(embs) < 2:
            continue
        embs = np.array(embs)
        sim_matrix = cosine_similarity(embs)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        intra_similarities.extend(sim_matrix[mask])

    avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0

    # Inter-class distance
    centroids = {}
    for label, embs in class_embeddings.items():
        centroids[label] = np.mean(embs, axis=0)

    labels_list = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in labels_list])

    inter_distances = []
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            dist = 1 - cosine_similarity([centroid_matrix[i]], [centroid_matrix[j]])[0, 0]
            inter_distances.append(dist)

    avg_inter_distance = np.mean(inter_distances) if inter_distances else 0

    separation_score = avg_inter_distance - (1 - avg_intra_similarity)

    return {
        'intra_class_similarity': avg_intra_similarity,
        'inter_class_distance': avg_inter_distance,
        'separation_score': separation_score,
        'class_counts': {label_names[k]: len(v) for k, v in class_embeddings.items()}
    }

# Create sentence transformer model
word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=MAX_LENGTH)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode="mean"
)

st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)

train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(st_model)

warmup_steps = int(len(train_dataloader) * 3 * 0.1)

# Create validation evaluator
val_sentences1 = [pair.texts[0] for pair in val_pairs[:min(500, len(val_pairs))]]
val_sentences2 = [pair.texts[1] for pair in val_pairs[:min(500, len(val_pairs))]]
val_scores = [1.0] * len(val_sentences1)

evaluator = EmbeddingSimilarityEvaluator(
    val_sentences1,
    val_sentences2,
    val_scores,
    name='sentiment_validation'
)

# Training with early stopping
best_score = -1
patience = 3
patience_counter = 0

for epoch in range(10):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/10")
    print(f"{'='*60}")

    st_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps if epoch == 0 else 0,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),
    )

    print("\nEvaluating embedding separation on validation set...")
    val_metrics = evaluate_embeddings_separation(
        st_model,
        X_val,
        y_sent_val,
        sentiment_le.classes_,
        device=DEVICE
    )

    print(f"  Intra-class similarity: {val_metrics['intra_class_similarity']:.4f}")
    print(f"  Inter-class distance:   {val_metrics['inter_class_distance']:.4f}")
    print(f"  Separation score:       {val_metrics['separation_score']:.4f}")

    current_score = val_metrics['separation_score']

    if current_score > best_score:
        best_score = current_score
        patience_counter = 0
        st_model.save(STAGE1_BEST_DIR)
        print(f"  New best model saved! (score: {best_score:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        break

# Load best model
print(f"\n{'='*60}")
print("Loading best encoder model...")
print(f"{'='*60}")
st_model = SentenceTransformer(STAGE1_BEST_DIR, device=DEVICE)

# Final test evaluation
print("\nFinal test evaluation (Stage 1):")
test_metrics = evaluate_embeddings_separation(
    st_model,
    X_test,
    y_sent_test,
    sentiment_le.classes_,
    device=DEVICE
)

print(f"  Test Intra-class similarity: {test_metrics['intra_class_similarity']:.4f}")
print(f"  Test Inter-class distance:   {test_metrics['inter_class_distance']:.4f}")
print(f"  Test Separation score:       {test_metrics['separation_score']:.4f}")

# Save final model
os.makedirs(STAGE1_DIR, exist_ok=True)

tokenizer_stage1 = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_stage1.save_pretrained(STAGE1_DIR)

transformer_module = st_model[0]
transformer_module.auto_model.save_pretrained(STAGE1_DIR)

print(f"\nStage 1 encoder saved to: {STAGE1_DIR}")

# ============= STAGE 2: MULTI-TASK CLASSIFIER =============
print("\n" + "="*60)
print("STAGE 2: Training Multi-Task Classifier")
print("="*60)

def make_ds(texts, y_topic, y_stance, y_sent):
    return Dataset.from_dict({
        "text": texts,
        "labels_topic": y_topic,
        "labels_stance": y_stance,
        "labels_sentiment": y_sent,
    })

ds = DatasetDict({
    "train": make_ds(X_train, y_topic_train, y_stance_train, y_sent_train),
    "validation": make_ds(X_val, y_topic_val, y_stance_val, y_sent_val),
    "test": make_ds(X_test, y_topic_test, y_stance_test, y_sent_test),
})

tokenizer = AutoTokenizer.from_pretrained(STAGE1_BEST_DIR)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

ds = ds.map(tokenize, batched=True)
ds.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels_topic",
        "labels_stance",
        "labels_sentiment",
    ],
)

def class_weights(y, n_classes):
    """Calculate balanced class weights"""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

topic_weights = class_weights(y_topic_train, len(topic_le.classes_))
stance_weights = class_weights(y_stance_train, len(stance_le.classes_))
sentiment_weights = class_weights(y_sent_train, len(sentiment_le.classes_))

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_path, n_topic, n_stance, n_sentiment, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        h = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.intermediate = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.topic_head = nn.Linear(h, n_topic)
        self.stance_head = nn.Linear(h, n_stance)
        self.sentiment_head = nn.Linear(h, n_sentiment)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_topic=None,
        labels_stance=None,
        labels_sentiment=None,
        **kwargs,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        x = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        x = self.dropout(x)
        x = self.intermediate(x)

        logits_topic = self.topic_head(x)
        logits_stance = self.stance_head(x)
        logits_sentiment = self.sentiment_head(x)

        loss = None
        if labels_topic is not None:
            device = logits_topic.device

            loss_topic = nn.CrossEntropyLoss(weight=topic_weights.to(device))(logits_topic, labels_topic)
            loss_stance = nn.CrossEntropyLoss(weight=stance_weights.to(device))(logits_stance, labels_stance)
            loss_sentiment = nn.CrossEntropyLoss(weight=sentiment_weights.to(device))(logits_sentiment, labels_sentiment)

            loss = loss_topic + loss_stance + loss_sentiment

        return {
            "loss": loss,
            "logits_topic": logits_topic,
            "logits_stance": logits_stance,
            "logits_sentiment": logits_sentiment,
        }

class MultiTaskTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"].detach() if outputs["loss"] is not None else None

        logits = (
            outputs["logits_topic"].detach(),
            outputs["logits_stance"].detach(),
            outputs["logits_sentiment"].detach(),
        )
        labels = (
            inputs["labels_topic"].detach(),
            inputs["labels_stance"].detach(),
            inputs["labels_sentiment"].detach(),
        )
        return loss, logits, labels

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if hasattr(self.model.encoder, "config"):
            self.model.encoder.config.save_pretrained(output_dir)

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, metric_name='mean_f1_macro'):
        self.metric_name = metric_name
        self.best_metric = -1
        self.best_model_path = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None:
            return

        current_metric = metrics.get(f'eval_{self.metric_name}', -1)

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_model_path = os.path.join(args.output_dir, 'best_model')

            os.makedirs(self.best_model_path, exist_ok=True)

            print(f"\n  New best model! {self.metric_name}: {current_metric:.4f}")
            print(f"    Saving to {self.best_model_path}")

            if model is not None:
                state_dict = model.state_dict()
                state_dict = {k: v.contiguous() for k, v in state_dict.items()}
                torch.save(state_dict, os.path.join(self.best_model_path, "pytorch_model.bin"))

                if hasattr(model.encoder, "config"):
                    model.encoder.config.save_pretrained(self.best_model_path)

def compute_metrics(eval_pred):
    """Compute multi-task metrics"""
    logits, labels = eval_pred
    logits_topic, logits_stance, logits_sentiment = logits
    y_topic, y_stance, y_sentiment = labels

    pred_topic = np.argmax(logits_topic, axis=1)
    pred_stance = np.argmax(logits_stance, axis=1)
    pred_sentiment = np.argmax(logits_sentiment, axis=1)

    topic_f1 = f1_score(y_topic, pred_topic, average="macro")
    stance_f1 = f1_score(y_stance, pred_stance, average="macro")
    sentiment_f1 = f1_score(y_sentiment, pred_sentiment, average="macro")

    return {
        "topic_acc": accuracy_score(y_topic, pred_topic),
        "stance_acc": accuracy_score(y_stance, pred_stance),
        "sentiment_acc": accuracy_score(y_sentiment, pred_sentiment),
        "topic_f1_macro": topic_f1,
        "stance_f1_macro": stance_f1,
        "sentiment_f1_macro": sentiment_f1,
        "mean_f1_macro": (topic_f1 + stance_f1 + sentiment_f1) / 3,
    }

model = MultiTaskModel(
    STAGE1_BEST_DIR,
    n_topic=len(topic_le.classes_),
    n_stance=len(stance_le.classes_),
    n_sentiment=len(sentiment_le.classes_),
    dropout=0.3,
)

args = TrainingArguments(
    output_dir=STAGE2_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mean_f1_macro",
    greater_is_better=True,
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED,
)

save_best_callback = SaveBestModelCallback(metric_name='mean_f1_macro')

trainer = MultiTaskTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
    callbacks=[save_best_callback],
)

trainer.train()

# ============= FINAL TEST EVALUATION =============
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

best_model_path = save_best_callback.best_model_path

if best_model_path and os.path.exists(best_model_path):
    print(f"\nLoading best model from: {best_model_path}")
    print(f"  Best validation {save_best_callback.metric_name}: {save_best_callback.best_metric:.4f}")

    best_model = MultiTaskModel(
        STAGE1_BEST_DIR,
        n_topic=len(topic_le.classes_),
        n_stance=len(stance_le.classes_),
        n_sentiment=len(sentiment_le.classes_),
        dropout=0.3,
    )

    state_dict = torch.load(os.path.join(best_model_path, "pytorch_model.bin"))
    best_model.load_state_dict(state_dict)

    test_args = TrainingArguments(
        output_dir=STAGE2_DIR,
        per_device_eval_batch_size=32,
        eval_strategy="no",
        report_to="none",
        seed=SEED,
    )

    test_trainer = MultiTaskTrainer(
        model=best_model,
        args=test_args,
        compute_metrics=compute_metrics,
    )

    test_results = test_trainer.evaluate(ds["test"])

    print("\n" + "="*60)
    print("TEST SET RESULTS (Best Model)")
    print("="*60)
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"  {metric_name:20s}: {value:.4f}")
else:
    print("\nNo best model found, using final model")
    test_results = trainer.evaluate(ds["test"])

    print("\n" + "="*60)
    print("TEST SET RESULTS (Final Model)")
    print("="*60)
    for key, value in test_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            print(f"  {metric_name:20s}: {value:.4f}")

print("\nTraining complete!")
print(f"  Stage 1 model: {STAGE1_BEST_DIR}")
print(f"  Stage 2 model: {best_model_path if best_model_path else STAGE2_DIR}")

# ============= EXPORT MODEL ARTIFACTS =============
EXPORT_DIR = "/content/cbr_model_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("\n" + "="*60)
print("EXPORTING MODEL ARTIFACTS")
print("="*60)

# Stage 1 encoder
stage1_export = os.path.join(EXPORT_DIR, "stage1_encoder")
shutil.copytree(STAGE1_BEST_DIR, stage1_export, dirs_exist_ok=True)
print(f"Stage 1 encoder saved: {stage1_export}")

# Stage 2 classifier
stage2_export = os.path.join(EXPORT_DIR, "stage2_classifier")
os.makedirs(stage2_export, exist_ok=True)

if save_best_callback.best_model_path and os.path.exists(save_best_callback.best_model_path):
    shutil.copy(
        os.path.join(save_best_callback.best_model_path, "pytorch_model.bin"),
        os.path.join(stage2_export, "pytorch_model.bin")
    )
    print(f"Stage 2 classifier (best model) saved: {stage2_export}")
else:
    torch.save(
        model.state_dict(),
        os.path.join(stage2_export, "pytorch_model.bin")
    )
    print(f"Stage 2 classifier (final model) saved: {stage2_export}")

shutil.copy(
    os.path.join(STAGE1_BEST_DIR, "config.json"),
    os.path.join(stage2_export, "config.json")
)

# Tokenizer
tokenizer_export = os.path.join(EXPORT_DIR, "tokenizer")
os.makedirs(tokenizer_export, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(STAGE1_BEST_DIR)
tokenizer.save_pretrained(tokenizer_export)
print(f"Tokenizer saved: {tokenizer_export}")

# Label encoders
label_encoders = {
    "topic": topic_le,
    "stance": stance_le,
    "sentiment": sentiment_le,
}

with open(os.path.join(EXPORT_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

print(f"Label encoders saved: label_encoders.pkl")

# Label mappings (JSON format for readability)
label_mappings = {
    "topic": {i: label for i, label in enumerate(topic_le.classes_)},
    "stance": {i: label for i, label in enumerate(stance_le.classes_)},
    "sentiment": {i: label for i, label in enumerate(sentiment_le.classes_)},
}

with open(os.path.join(EXPORT_DIR, "label_mappings.json"), "w", encoding="utf-8") as f:
    json.dump(label_mappings, f, ensure_ascii=False, indent=2)

print(f"Label mappings saved: label_mappings.json")

# Model configuration
model_config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "n_topic": len(topic_le.classes_),
    "n_stance": len(stance_le.classes_),
    "n_sentiment": len(sentiment_le.classes_),
    "dropout": 0.3,
    "best_val_f1": save_best_callback.best_metric if save_best_callback.best_model_path else None,
}

with open(os.path.join(EXPORT_DIR, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

print(f"Model config saved: model_config.json")

# Test metrics
with open(os.path.join(EXPORT_DIR, "test_metrics.json"), "w") as f:
    clean_metrics = {k.replace('eval_', ''): v for k, v in test_results.items()}
    json.dump(clean_metrics, f, indent=2)

print(f"Test metrics saved: test_metrics.json")

# Create archive
print("\nCreating archive...")
shutil.make_archive("/content/cbr_model_export", 'zip', EXPORT_DIR)
print(f"Archive created: /content/cbr_model_export.zip")

print("\n" + "="*60)
print("EXPORT STRUCTURE:")
print("="*60)

for root, dirs, files in os.walk(EXPORT_DIR):
    level = root.replace(EXPORT_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        size = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
        print(f'{subindent}{file} ({size:.2f} MB)')

print("\n" + "="*60)
print("COMPLETE! Download: /content/cbr_model_export.zip")
print("="*60)

# ============= INFERENCE ON UNLABELED DATASET =============
print("\n" + "="*60)
print("LABELING UNLABELED DATASET")
print("="*60)

def split_into_sentences(text):
    """Simple sentence splitting"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def predict_sentences(texts, batch_size=32):
    """Predict labels for list of sentences"""
    results = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            pred_topic = torch.argmax(outputs["logits_topic"], dim=1).cpu().numpy()
            pred_stance = torch.argmax(outputs["logits_stance"], dim=1).cpu().numpy()
            pred_sentiment = torch.argmax(outputs["logits_sentiment"], dim=1).cpu().numpy()

            prob_topic = torch.softmax(outputs["logits_topic"], dim=1).cpu().numpy()
            prob_stance = torch.softmax(outputs["logits_stance"], dim=1).cpu().numpy()
            prob_sentiment = torch.softmax(outputs["logits_sentiment"], dim=1).cpu().numpy()

            for j, text in enumerate(batch_texts):
                results.append({
                    "sentence": text,
                    "topic": str(topic_le.classes_[pred_topic[j]]),
                    "topic_confidence": float(prob_topic[j, pred_topic[j]]),
                    "stance": str(stance_le.classes_[pred_stance[j]]),
                    "stance_confidence": float(prob_stance[j, pred_stance[j]]),
                    "sentiment": str(sentiment_le.classes_[pred_sentiment[j]]),
                    "sentiment_confidence": float(prob_sentiment[j, pred_sentiment[j]]),
                })

    return results

def parse_date(date_str):
    """Parse Russian date format to ISO"""
    months_ru = {
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
    }

    try:
        match = re.search(r'(\d+)\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day, month_name, year = match.groups()
            month = months_ru.get(month_name.lower())
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"
    except:
        pass

    return None

def label_cbr_dataset(input_jsonl_path, output_json_path, batch_size=32):
    """Label CBR dataset from JSONL format"""
    # Read JSONL
    data = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    labeled_data = []
    total_sentences = 0

    print(f"Processing {len(data)} documents...")

    for doc_idx, doc in enumerate(data):
        if (doc_idx + 1) % 10 == 0:
            print(f"  Processed {doc_idx + 1}/{len(data)} documents...")

        text = doc.get("text_main", "")
        if not text:
            continue

        sentences = split_into_sentences(text)
        if not sentences:
            continue

        predictions = predict_sentences(sentences, batch_size=batch_size)
        total_sentences += len(predictions)

        # Aggregate metrics
        topic_counts = Counter([p["topic"] for p in predictions])
        stance_counts = Counter([p["stance"] for p in predictions])
        sentiment_counts = Counter([p["sentiment"] for p in predictions])

        published_date = parse_date(doc.get("published_at", ""))

        labeled_doc = {
            "doc_id": doc.get("doc_id"),
            "section": doc.get("section"),
            "category": doc.get("category"),
            "title": doc.get("title"),
            "published_at": doc.get("published_at"),
            "published_date": published_date,
            "sentences": predictions,
            "aggregated": {
                "total_sentences": len(predictions),
                "topic_counts": dict(topic_counts),
                "stance_counts": dict(stance_counts),
                "sentiment_counts": dict(sentiment_counts),
            }
        }

        labeled_data.append(labeled_doc)

    # Save results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {output_json_path}")
    print(f"  Total documents: {len(labeled_data)}")
    print(f"  Total sentences: {total_sentences}")

    # Overall statistics
    all_topics = Counter()
    all_stances = Counter()
    all_sentiments = Counter()

    for doc in labeled_data:
        all_topics.update(doc["aggregated"]["topic_counts"])
        all_stances.update(doc["aggregated"]["stance_counts"])
        all_sentiments.update(doc["aggregated"]["sentiment_counts"])

    print(f"\nTopic distribution (top 5):")
    for topic, count in all_topics.most_common(5):
        print(f"  {topic:30s}: {count:5d} ({count/total_sentences*100:.1f}%)")

    print(f"\nStance distribution:")
    for stance, count in all_stances.most_common():
        print(f"  {stance:30s}: {count:5d} ({count/total_sentences*100:.1f}%)")

    print(f"\nSentiment distribution:")
    for sentiment, count in all_sentiments.most_common():
        print(f"  {sentiment:30s}: {count:5d} ({count/total_sentences*100:.1f}%)")

    return labeled_data

# Run labeling
INPUT_JSONL = "/content/inference_data_cbr.jsonl"
OUTPUT_JSON = "/content/cbr_labeled_predictions.json"

labeled_data = label_cbr_dataset(
    input_jsonl_path=INPUT_JSONL,
    output_json_path=OUTPUT_JSON,
    batch_size=32
)