# Stock Price Prediction Project

This project focuses on predicting stock prices using historical data. It includes **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Machine Learning Models** such as Random Forest, XGBoost, and LSTM. The goal is to predict the stock's closing price 5 trading days into the future.

---

## **Google Colab Notebook**
You can access the complete workflow and code in the Google Colab notebook:
(https://colab.research.google.com/drive/1qmvSttuH8tD7LcoOXG7wPVb0_gN6-rYH?usp=sharing)

---

## **Project Overview**
1. **Exploratory Data Analysis (EDA):**
   - Visualized key patterns and relationships in the data.
   - Analyzed trends, seasonality, and anomalies.
2. **Feature Engineering:**
   - Created meaningful features from the time series data (e.g., moving averages, Bollinger Bands, lag features).
   - Selected the top 10 features based on importance scores from the Random Forest model.
3. **Model Development:**
   - Trained and evaluated three models: **Random Forest**, **XGBoost**, and **LSTM**.
   - Compared their performance using metrics like RMSE, MAE, and R² Score.
4. **Model Selection:**
   - Selected **Random Forest** as the final model due to its superior performance.
   - Discussed the potential of **LSTM** for future improvements.

---

## **Results**
- **Random Forest (Top 10 Features):**
  - RMSE: 2.9175
  - MAE: 1.7363
  - R² Score: 0.9968
- **XGBoost (Top 10 Features):**
  - RMSE: 3.0709
  - MAE: 1.8883
  - R² Score: 0.9965
- **LSTM:**
  - RMSE: 6.4892
  - MAE: 4.4041
  - R² Score: 0.9568





