# 🏠 House Prices: Advanced Regression & Deep Learning Techniques

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20XGBoost%20|%20TensorFlow-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%205%25-blueviolet)

## 📌 Project Overview
This project aims to predict residential home prices in Ames, Iowa using advanced machine learning and deep learning techniques. The solution implements a rigorous **CRISP-DM** methodology, extensive **Feature Engineering**, and a **Stacked Ensemble** architecture combining Linear Models, Gradient Boosting Machines, and Neural Networks.

## 🛠️ Tech Stack & Methodology
* **Data Analysis:** Pandas, NumPy, SciPy (Skewness, Box-Cox)
* **Visualization:** Matplotlib, Seaborn
* **Preprocessing:** RobustScaler, OneHotEncoder, LabelEncoder
* **Machine Learning:**
    * *Regularized Linear Models:* Lasso, ElasticNet
    * *Kernel Methods:* Kernel Ridge Regression (KRR)
    * *Boosting:* XGBoost, LightGBM, GradientBoosting
* **Deep Learning:** TensorFlow/Keras (Dense Neural Network with Dropout & BatchNormalization)
* **Ensemble Strategy:** StackingCVRegressor + Weighted Blending

## 📊 Key Insights & Feature Engineering
* **Outlier Removal:** Identified and removed anomalies in `GrLivArea` to improve model generalization.
* **Target Transformation:** Applied `Log(1+x)` transformation to `SalePrice` to correct skewness and satisfy normality assumptions.
* **Feature Imputation:** Used domain knowledge to fill missing values (e.g., `PoolQC`=None implies No Pool, `LotFrontage` imputed by Neighborhood median).
* **New Features:** Created `TotalSF` (Total Square Footage) combining basement and upper floors.

## 🧠 Model Architecture
The final prediction is a weighted blend of four robust pipelines:
$$FinalPred = 0.60 \times StackedModel + 0.15 \times XGB + 0.15 \times LGBM + 0.10 \times NeuralNet$$

## 📂 Project Structure
```bash
├── data/               # Raw and processed data
├── notebooks/          # Jupyter Notebooks for EDA and Modeling
├── src/                # Modular python scripts
├── submissions/        # Generated CSV files for Kaggle
└── README.md           # Project documentation