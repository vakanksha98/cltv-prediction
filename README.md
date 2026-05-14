---
title: CLTV Prediction
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10
app_file: app.py
pinned: false
---

# Customer Lifetime Value Prediction

An interactive demo of ML-based CLTV prediction using RFM (Recency, Frequency, Monetary) analysis and XGBoost classification.

## Features

- **RFM Analysis** - Visualize customer segmentation
- **XGBoost Classification** - Predict high-CLTV customers (98% accuracy)
- **Churn Prediction** - Identify at-risk customers
- **Interactive Charts** - Explore customer segments

## Model Performance

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| XGBoost | 0.998 | 98% |
| Random Forest | 0.996 | - |

## How It Works

```
Customer Data → RFM Analysis → CLTV Modeling → Classification → Churn Prediction
```

## Tech Stack

- Python · XGBoost · scikit-learn · lifetimes · Gradio

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur