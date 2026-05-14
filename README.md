# CLTV Prediction - Customer Lifetime Value Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FG6009?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

Customer Lifetime Value (CLTV) prediction using machine learning and RFM (Recency, Frequency, Monetary) analysis.

## Overview

This project analyzes online retail data to predict customer lifetime value, segment customers, and identify churn risk. It uses both regression (CLTV prediction) and classification (high-CLTV identification) approaches.

## Features

- **RFM Analysis** - Recency, Frequency, Monetary segmentation
- **CLTV Calculation** - Beta-Geometric/Gamma-Gamma models
- **Customer Classification** - Random Forest & XGBoost classifiers
- **Churn Prediction** - 90-day inactivity risk assessment
- **Feature Importance** - Identify key CLTV drivers

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| ML Models | scikit-learn, XGBoost |
| Advanced CLTV | lifetimes (BetaGeoFitter, GammaGammaFitter) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

## Key Concepts

### RFM Metrics
| Metric | Description |
|--------|-------------|
| Recency (R) | Days since last purchase |
| Frequency (F) | Number of transactions |
| Monetary (M) | Total spending amount |

### Customer Segments
| Segment | RFM Profile | Description |
|---------|-------------|-------------|
| Champions | 44 | Recent, frequent, high-value |
| Loyal | 34, 43 | Consistent spenders |
| Potential Loyalists | 14, 24 | Developing relationships |
| At Risk | 31, 32 | Declining engagement |

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Classifier | AUC-ROC | 0.998 |
| XGBoost Classifier | Accuracy | 98% |
| XGBoost Classifier | F1-Score | 0.96 |
| Random Forest Classifier | AUC-ROC | 0.996 |

## Project Structure

```
cltv-prediction/
├── CLTV-4.ipynb              # Main analysis notebook
├── online+retail.zip         # Dataset (extracted on use)
├── README.md                  # This file
└── requirements.txt          # Dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/vakanksha98/cltv-prediction.git
cd cltv-prediction

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lifetimes>=0.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## Usage

1. Open `CLTV-4.ipynb` in Jupyter or Google Colab
2. Extract `online+retail.zip` if needed
3. Run cells sequentially

### Workflow
```
Load & Clean Data
       ↓
Feature Engineering (RFM, AOV)
       ↓
RFM Analysis & Segmentation
       ↓
CLTV Modeling (BetaGeoFitter, GammaGammaFitter)
       ↓
Classification (High CLTV Prediction)
       ↓
Churn Prediction
       ↓
Model Evaluation & Comparison
```

## Business Applications

- **Targeted Marketing** - Focus on Champions & Loyal segments
- **Retention** - Early identification of "At Risk" customers
- **Resource Allocation** - Prioritize high-CLTV segments
- **Churn Prevention** - Proactive engagement strategies
- **Pricing Strategy** - Personalized offers by CLTV tier

## Portfolio Presentation

This project demonstrates:
- End-to-end ML pipeline (data → features → model → evaluation)
- RFM analysis methodology
- Advanced CLTV modeling (BG-NBD, Gamma-Gamma)
- Ensemble methods (XGBoost vs Random Forest)
- Business impact translation (data science to business value)

## Future Improvements

- [ ] Add time-series forecasting for CLTV trends
- [ ] Deploy as Streamlit dashboard
- [ ] Add cohort analysis
- [ ] Integrate with CRM for real-time scoring
- [ ] A/B testing framework for retention campaigns

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur

## License

MIT License - See [LICENSE](LICENSE) for details.