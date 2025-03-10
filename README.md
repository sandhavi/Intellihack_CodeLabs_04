# Stock Price Prediction System

## 📌 Overview
This project implements a **Stock Price Prediction System** using an **LSTM (Long Short-Term Memory) model**. The system fetches market data, processes it, makes predictions, and visualizes the insights for financial analysts and traders.

## 🚀 Features
- **Stock Price Prediction:** Uses historical market data to predict the future closing price.
- **Technical Indicators:** Incorporates moving averages, Bollinger Bands, and cumulative returns.
- **User Interface:** Built with **Streamlit** for an interactive experience.
- **Scalable Architecture:** Supports batch and real-time data ingestion.
- **Model Monitoring & Updates:** Ensures accuracy with periodic retraining.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```sh
https://github.com/sandhavi/Intellihack_CodeLabs_04.git
cd stock_prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app/main.py
```

---

## 📊 System Architecture

### 1️⃣ Data Collection & Ingestion
- **Sources:** Live market data from APIs (e.g., Alpha Vantage, Yahoo Finance) & CSV files.
- **Batch & Streaming:** Supports both **real-time** and **historical** data processing.
- **Storage:** Saves processed data in a **SQLite/PostgreSQL** database for future use.

### 2️⃣ Data Processing Pipeline
- **Preprocessing:** Cleans missing values, normalizes data using **MinMaxScaler**.
- **Feature Engineering:** Computes **Moving Averages, Bollinger Bands, and Lag Features**.
- **Storage Architecture:** Raw data and transformed features are stored for future reference.

### 3️⃣ Model Operations
- **Training:** The **LSTM model** is trained on historical stock data.
- **Evaluation:** Assessed using **RMSE & MAPE**.
- **Deployment:** The trained model is stored as a `.keras` file and loaded dynamically.
- **Monitoring:** Performance is tracked, and the model is retrained periodically.

### 4️⃣ Insight Delivery
- **Web App:** Uses **Streamlit** for an interactive UI.
- **Visualizations:** Displays stock trends, predicted prices, and technical indicators.
- **Alerts:** Future enhancement includes email/SMS notifications for price changes.

### 5️⃣ System Considerations
- **Scalability:** Can be deployed on **AWS/GCP** for real-time trading.
- **Reliability:** Uses **error handling** and **logging** for robustness.
- **Latency:** Optimized preprocessing & inference for fast results.
- **Cost:** Balances cloud-based vs. local deployment based on user requirements.

---

## 🔄 Data Flow Explanation
1️⃣ **Market Data Collection:**
   - Real-time API fetch or CSV upload.
   
2️⃣ **Preprocessing:**
   - Clean data, normalize features, generate lagged values.
   
3️⃣ **Model Inference:**
   - Data is fed into the **LSTM model** to predict stock prices.
   
4️⃣ **Results Display:**
   - Predictions & insights are displayed in the Streamlit app.

---

## 🛑 Challenges & Solutions
### 1️⃣ Handling Missing Data
- **Solution:** Impute missing values using **interpolation & forward-fill techniques**.

### 2️⃣ Model Drift Over Time
- **Solution:** Periodic **retraining** with new data & evaluation.

### 3️⃣ Real-Time Data Processing
- **Solution:** Implement **Kafka** or **WebSockets** for live updates.

### 4️⃣ Data Security & Privacy
- **Solution:** Use **secure API keys & encrypted storage**.

---

## 📈 Future Improvements
- **Deploy on Cloud (AWS/GCP)** for real-time trading.
- **Enhance Feature Engineering** with sentiment analysis & macroeconomic indicators.
- **Incorporate Reinforcement Learning** for optimized trading strategies.
- **Improve UI with Dash/React** for better user experience.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork, submit PRs, or open issues.

---



## 🔗 Connect
- **GitHub:** [your-username](https://github.com/ovindumandith)
- **LinkedIn:** [your-profile](https://www.linkedin.com/in/ovindu-gunatunga/)

