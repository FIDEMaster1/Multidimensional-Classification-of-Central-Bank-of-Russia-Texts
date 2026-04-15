# Multi-Label Classification of CBR Communications

Two-stage NLP pipeline for classifying Central Bank of Russia (CBR) texts by **topic** (12 classes), **stance** (3 classes), and **sentiment** (6 classes), followed by visualization and macroeconomic analysis.

The project is inspired by recent literature showing that central bank communication is a **powerful policy tool** that shapes expectations, financial markets, and macroeconomic outcomes :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}.

---

## 📌 Pipeline Overview

### Stage 1: Fine-tune Encoder (RuBERT)
Contrastive learning is used to build high-quality sentence embeddings tailored to central bank language.

### Stage 2: Multi-Task Classifier
A neural classifier predicts:
- Topic
- Stance
- Sentiment

### Stage 3: Visualization & Analysis
- Time-series aggregation
- Communication indices
- Macroeconomic comparison

---

## 🏷 Labels

### Topic (12)
- Inflation  
- Interest rate  
- Monetary conditions  
- Credit  
- Expectations  
- Economic activity  
- Labor market  
- External conditions  
- Fiscal policy  
- Exchange rate  
- Financial sector  
- Other  

### Stance (3)
- Forward-looking  
- Backward-looking  
- Other  

### Sentiment (6)
- Hawkish  
- Dovish  
- Neutral  
- Risk-highlighting  
- Confidence-building  
- Other  
