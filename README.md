# Bank of Russia Monetary Policy Communication NLP

This repository contains the code for a master's thesis project on sentence-level classification of Bank of Russia monetary policy communication.

The project classifies central bank communication along three dimensions:

- **Topic**: the main economic theme of the sentence.
- **Temporal orientation**: whether the sentence is forward-looking, backward-looking, or other.
- **Policy sentiment**: hawkish, dovish, neutral, risk-highlighting, confidence-building, or other.

The resulting sentence-level predictions are aggregated into communication-based monetary policy indicators, including Net Policy Sentiment.

## Repository structure

```text
├── configs/
│   ├── labels.json
│   ├── classical_baselines.yaml
│   ├── final_rubert.yaml
│   └── dapt.yaml
│
├── data/
│   ├── manual_clean_encoded.csv
│   ├── synthetic_clean_encoded.csv
│   ├── combined_clean_encoded.csv
│   ├── label2id.json
│   └── id2label.json
│
├── notebooks/
│   ├── 00_collect_cbr_html_texts.py
│   ├── 01_prepare_labeled_data_and_splits.py
│   ├── 02_classical_baselines.py
│   ├── 03_transformer_single_vs_multitask.py
│   ├── 04_encoder_comparison.py
│   ├── 05_collect_dapt_corpus.py
│   ├── 06_clean_dapt_corpus.py
│   ├── 07_train_dapt_rubert.py
│   ├── 08_final_rubert_models.py
│   └── 09_error_analysis_and_indices.py
│
├── .gitignore
└── README.md
