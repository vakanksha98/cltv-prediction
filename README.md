# CLTV Prediction - Customer Lifetime Value Analysis

## Overview
Customer Lifetime Value (CLTV) prediction project using machine learning and RFM (Recency, Frequency, Monetary) analysis. The notebook demonstrates data processing, RFM segmentation, CLTV calculation, customer classification, and churn prediction.

## Key Concepts & Techniques

### 1. **Data Loading & Cleaning**
- **Dataset**: OnlineRetail.csv (Online Retail transaction data)
- **Columns**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Cleaning Steps**:
  - Remove missing CustomerID values
  - Remove negative quantities (returns/cancellations)
  - Handle missing descriptions
  - Convert InvoiceDate to datetime format

### 2. **Feature Engineering**
- **TotalPrice**: Quantity × UnitPrice (revenue per transaction)
- **AOV (Average Order Value)**: Total Monetary Value ÷ Frequency
- **RFM Metrics**:
  - **Recency (R)**: Days since last purchase
  - **Frequency (F)**: Number of transactions
  - **Monetary (M)**: Total spending amount

### 3. **RFM Analysis & Scoring**
- Quartile-based scoring (1-4 scale) for R, F, M
- Combined RFM Score (e.g., "444" = Best customer)
- **Customer Segments**:
  - **Champions** (RFM: 44) - Recent, frequent, high-value
  - **Loyal** (RFM: 34, 43) - Consistent spenders
  - **Potential Loyalists** (RFM: 14, 24) - Developing relationships
  - **Recent Customers** (RFM: 41, 42) - New but active
  - **At Risk** (RFM: 31, 32) - Declining engagement
  - **Others** - Remaining customers

### 4. **CLTV Calculation**

#### Method 1: Simple Formula
```
CLTV = AOV × Purchase Frequency × Gross Margin
```

#### Method 2: Beta-Gamma-Fitter (Probabilistic)
- **BetaGeoFitter**: Predicts future purchase transactions
- **GammaGammaFitter**: Estimates customer lifetime value
- Incorporates customer purchasing patterns & monetary dynamics
- Parameters:
  - `penalizer_coef` (0.1): Stabilizes convergence
  - `time` (6): 6-month prediction horizon
  - `freq` ('D'): Daily frequency
  - `discount_rate` (0.01): 1% discount for future cash flows

### 5. **Machine Learning Models**

#### Regression (CLTV Prediction)
- **Random Forest Regressor**
  - 100 estimators
  - Features: Recency, Frequency, Monetary, AOV
  - RMSE, MAE, R² score evaluation
  
- **XGBoost Regressor**
  - 200 estimators, learning_rate=0.1, max_depth=5
  - Better gradient boosting for complex patterns
  - Performance metrics comparison

#### Classification (High CLTV Prediction)
- **Target**: High_CLTV (Top 30% of customers)
- **Models**:
  - **Random Forest Classifier**: 100 trees
    - Precision: 0.97, Recall: 0.89, F1: 0.93
    - AUC: 0.996
  - **XGBoost Classifier**: 200 trees
    - Precision: 0.99, Recall: 0.93, F1: 0.96
    - AUC: 0.998 (Better performance)

### 6. **Feature Importance**
- Identifies which RFM metrics drive CLTV predictions
- Random Forest vs XGBoost importance comparison
- Frequency & Monetary typically most important

### 7. **Churn Prediction**
- **Definition**: No purchase for 90+ days
- Predicts customer churn risk
- Enables proactive retention strategies

## Libraries Used
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Models**: scikit-learn (RF, train_test_split)
- **Advanced Models**: XGBoost, lifetimes (BetaGeoFitter, GammaGammaFitter)
- **Metrics**: Classification_report, confusion_matrix, ROC-AUC

## Key Findings

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Class | Accuracy | 98% |
| XGBoost Class | AUC-ROC | 0.998 |
| RF Regressor | R² | 0.089 |
| XGB Regressor | R² | 0.096 |

## Workflow Summary

1. **Load & Clean** → Remove nulls, invalid data
2. **Engineer Features** → Create TotalPrice, AOV
3. **RFM Analysis** → Calculate R, F, M metrics
4. **Segmentation** → Categorize customers
5. **CLTV Modeling** → Predict customer value
6. **Classification** → Identify high-value customers
7. **Churn Analysis** → Flag at-risk customers
8. **Evaluate** → Compare multiple models

## How to Use

1. **Open CLTV-4.ipynb** in Jupyter
2. **Prepare data**: Ensure OnlineRetail.csv is available
3. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lifetimes
   ```
4. **Run notebook cells** sequentially
5. **Visualize results**: Check segment distributions & model performance

## Output Files
- `cltv-2.pdf`: Visualization report
- RFM dataframe with customer segments
- CLTV predictions with confidence intervals
- Feature importance plots

## Business Applications

- **Targeted Marketing**: Focus on Champions & Loyal customers
- **Retention**: Identify "At Risk" customers early
- **Resource Allocation**: Prioritize high-CLTV segments
- **Churn Prevention**: Proactive engagement for churned customers
- **Pricing Strategy**: Personalized offers based on CLTV tiers

## Key Metrics Explained

- **RMSE**: Lower is better (average prediction error)
- **MAE**: Absolute average error in CLTV estimation
- **R² Score**: Explains variance in CLTV (0-1 scale)
- **AUC-ROC**: Measures classification performance (0.5-1.0)
- **Precision**: Accuracy of positive predictions
- **Recall**: Capture rate of true positives

## Contact
**Author**: Akanksha Verma
- GitHub: [@vakanksha98](https://github.com/vakanksha98)
- Email: akanksha.v.official@gmail.com

## License
MIT License
