# 🌍 Air Quality Index (AQI) Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-green.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end Machine Learning data science project to predict Air Quality Index (PM 2.5 concentration) using Linear Regression based on fundamental climate and atmospheric features.

## 📌 Problem Statement

Air pollution is a major environmental risk to health. Predicting PM 2.5 levels effectively allows for better environmental monitoring and proactive safety measures. This project utilizes historical climate and atmospheric records including Temperature, Humidity, Wind Speed, and Atmospheric Pressure to predict the PM 2.5 concentration level.

## 📂 Project Structure

This project follows professional data science template practices:

```text
├── data/
│   ├── raw/                 # Raw dataset (AQI_Data.csv)
│   └── processed/           # Processed/Cleaned data
├── models/                  # Saved pickled models (.pkl)
├── notebooks/               # Jupyter notebooks for EDA and draft modeling
├── reports/                 
│   └── figures/             # Visualizations generated during EDA & Evaluation
├── src/                     # Source code files
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Exploratory Data Analysis (EDA)

Understanding feature distributions and correlations before jumping into predictive modeling is critical.

### 1. PM 2.5 Target Distribution
We analyzed the distribution of the target variable to understand data skewness and density.

![PM2.5 Distribution](reports/figures/pm25_distribution.png)

### 2. Feature Correlation Heatmap
By exploring the linear relationships between variables, we identified the strongest driving properties for PM 2.5 concentrations, which heavily influenced the base assumptions of our linear regression approach.

![Feature Correlation](reports/figures/correlation_heatmap.png)

---

## ⚙️ Model Development & Evaluation

We applied an algorithmic approach via **Linear Regression**, mapping climate conditions (T, TM, Tm, SLP, H, VV, V, VM) to PM 2.5 levels.

### Results
After validating our approach on a 20% holdout test set, the metric evaluations are:

| Metric | Score |
| --- | --- |
| **Root Mean Squared Error (RMSE)** | `56.71` |
| **Mean Absolute Error (MAE)** | `42.57` |
| **R-Squared ($R^2$)** | `0.4866` |

> *Note: While the R-squared reflects a moderate baseline model, further non-linear models (Random Forest, XGBoost) and advanced feature engineering could stretch performance further. This fulfills the objective of a solid linear baseline setup.*

### Actual vs. Predicted Target Performance
![Actual vs Predicted](reports/figures/actual_vs_predicted.png)

*(Points closer to the red diagonal line indicate perfectly accurate predictions).*

---

## 🚀 Getting Started

To reproduce this project locally, simply follow these steps.

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Air-Quality-Index-Predictor.git
cd Air-Quality-Index-Predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain the Model
Execute the continuous training script, which will automatically clean the data, generate graphs within the reports directory, extract metrics, and persist the new model to `models/`:
```bash
cd src
python train.py
```

### 3. Run Inference
You can directly input physical atmospheric features into the CLI and receive an instantaneous PM 2.5 prediction:
```bash
python predict.py --T 15.0 --H 80.0 --VM 12.5
```
**Example Output:**
```text
Predicted PM 2.5: 142.34
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
