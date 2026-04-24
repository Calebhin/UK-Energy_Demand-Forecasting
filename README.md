# 🇬🇧 UK National Electricity Demand Forecasting
## Weather-Enriched Machine Learning and Deep Learning Models

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🔗 Live Dashboard
👉 **[https://uk-energy-dashboard.streamlit.app](https://uk-energy-dashboard.streamlit.app)**

---

## 📋 Project Overview
This repository contains the full implementation of an MSc Data Science
and Artificial Intelligence dissertation project at Bournemouth University.
The study systematically develops and benchmarks weather-enriched
forecasting models for UK national electricity demand using the National
Grid ESO dataset spanning 2009 to 2025.

---

## ❓ Research Question
To what extent can the integration of high-resolution meteorological
variables and calendar features with hybrid deep-learning architectures
improve medium-term and long-term UK national electricity demand
forecasting accuracy compared to current published benchmarks?

---

## 📊 Key Results

### Validation Set (2021)
| Model | MAE (MW) | RMSE (MW) | sMAPE (%) |
|---|---|---|---|
| **XGBoost + Lags ★** | **333** | **457** | **1.1** |
| LSTM | 385 | 520 | 1.4 |
| TFT (Improved) | 703 | 1,013 | 2.6 |
| Prophet (Improved) | 2,330 | 2,890 | 8.2 |
| Prophet (Basic) | 2,567 | 3,201 | 9.3 |
| XGBoost (Basic) | 3,799 | 4,280 | 13.3 |
| N-BEATSx | 6,340 | 7,513 | 26.4 |

### Test Set (2022-2025)
| Model | MAE (MW) | RMSE (MW) | sMAPE (%) |
|---|---|---|---|
| **XGBoost + Lags ★** | **365** | **501** | **1.5** |
| LSTM | 707 | 927 | 3.0 |
| TFT | 1,143 | 1,527 | 4.7 |
| Prophet (Improved) | 3,110 | 3,841 | 15.5 |
| N-BEATSx | 4,114 | 4,926 | 14.7 |

★ Champion model confirmed by Diebold-Mariano and Wilcoxon tests
(n = 20,994, all p < 0.0001)

---

## 📁 Repository Structure

    UK-Energy-Dashboard/
    ├── app.py                             
    ├── xgboost_demand_model.json          
    ├── lstm_demand_model.keras            
    ├── requirements.txt                   
    ├── README.md                          
    ├── notebooks/
    │   └── UK_Energy_Forecast_Model.ipynb 
    └── results/
        ├── validation_leaderboard.csv     
        ├── test_leaderboard.csv           
        ├── ablation_results.csv           
        ├── shap_summary_beeswarm.png      
        ├── shap_bar.png                   
        ├── shap_dependence_lag_1.png      
        ├── shap_dependence_lag_24.png     
        ├── shap_dependence_month.png      
        ├── shap_waterfall_worst.png       
        ├── ablation_study.png             
        └── ablation_smape.png             

---

## 🚀 Getting Started

### 1. Clone the repository

    git clone https://github.com/calebhin/UK-Energy-Dashboard.git
    cd UK-Energy-Dashboard

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Download the data
The dataset is too large for GitHub. Download from:
- **Demand data:** [National Grid ESO — Kaggle](https://www.kaggle.com/datasets/tomfarnell/national-grid-energy-consumption-2009-2025)
- **Weather data:** [ERA5 — Copernicus CDS](https://cds.climate.copernicus.eu/)

### 4. Run the notebook

    jupyter notebook notebooks/UK_Energy_Forecast_Model.ipynb

### 5. Launch the dashboard locally

    streamlit run app.py

---

## 🤖 Models Implemented
| Model | Type | Val MAE | Test MAE |
|---|---|---|---|
| Prophet (Basic) | Statistical baseline | 2,567 MW | — |
| Prophet (Improved) | Statistical improved | 2,330 MW | 3,110 MW |
| XGBoost (Basic) | ML baseline | 3,799 MW | — |
| XGBoost + Lags | ML Champion ★ | 333 MW | 365 MW |
| LSTM | Deep learning | 385 MW | 707 MW |
| N-BEATSx | Deep learning exog | 6,340 MW | 4,114 MW |
| TFT | Transformer | 703 MW | 1,143 MW |

---

## 📈 Statistical Validation

- **Diebold-Mariano tests:** all p < 0.0001 confirming XGBoost superiority
- **Wilcoxon signed-rank tests:** all p < 0.0001 corroborating DM results
- **Sample size:** n = 20,994 hourly observations (2022-2025)

---

## 🔍 SHAP Interpretability

| Feature | Mean SHAP (MW) | Interpretation |
|---|---|---|
| lag_1 | 5,367 | Previous hour demand — dominant predictor |
| lag_168 | 934 | Same hour last week — weekly cycle |
| hour | 801 | Time of day — daily demand pattern |
| lag_24 | 313 | Yesterday same hour — daily cycle |
| rolling_mean_24 | 137 | Recent trend context |
| dayofweek | 127 | Weekday vs weekend |
| month | 125 | Seasonal effect |
| rolling_std_24 | 54 | Demand volatility |
| is_weekend | 6 | Weekend indicator |

---

## 🌐 Dashboard Features

- 24-hour demand forecast with uncertainty bands
- Model switcher between XGBoost and LSTM
- Weather scenario exploration (temperature impact)
- Real-time prediction using champion XGBoost model
- Model performance metrics display

**Live URL:** https://uk-energy-dashboard.streamlit.app

---

## 📚 Data Sources

- Farnell, T. (2024). National Grid energy consumption 2009-2025. Kaggle.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. QJRMS, 146(730).

---

## 👤 Author

**Caleb Adejare**
MSc Data Science and Artificial Intelligence
Bournemouth University
Supervisor: Kevin Wilson

---

## 📄 License

This project is licensed under the MIT License.