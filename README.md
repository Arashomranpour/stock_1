<div align="center">

# 📈 Stock Price Prediction with LSTM

**Download live market data with yfinance, forecast prices with a deep-learning LSTM model and explore them in a Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## ✨ Overview

- 📥 Historical prices come from **Yahoo Finance** (`yfinance`) from **2019-01-01 until today** (default ticker `AAPL`).
- 🧠 `stock_1.ipynb` scales the closing price with `MinMaxScaler`, builds sliding-window sequences and trains a stacked **LSTM** network (Keras, 50 epochs; final training loss ≈ 0.008). The trained model is saved as `AAPL_stock.h5`.
- 🌐 `arash.py` is the **Streamlit app** "Stock Prediction using Deep Learning":
  - enter any stock ticker,
  - view summary statistics of the data,
  - closing price vs time with **100-day** and **200-day moving averages**,
  - predicted vs original prices.

> ⚠️ Educational project - not financial advice.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/stock_1.git
cd stock_1
pip install -r requirements.txt
streamlit run arash.py
```

## 📁 Project Structure

```
.
├── stock_1.ipynb     # Data prep, LSTM training
├── arash.py          # Streamlit dashboard
├── AAPL_stock.h5     # Trained model
└── requirements.txt
```

## 🛠️ Tech Stack

`Keras / TensorFlow` · `yfinance` · `scikit-learn` · `pandas` · `Matplotlib` · `Streamlit`
