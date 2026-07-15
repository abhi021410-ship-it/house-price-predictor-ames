# 🏠 House Price Predictor (Ames Dataset)

A machine‑learning web app that predicts house prices based on **location, size, rooms, and amenities**.  
Built with Python, scikit‑learn, Flask, and a modern dark‑themed UI.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## 📖 Overview

This project uses the **Ames Housing Dataset** (1460 residential homes in Ames, Iowa) to train a Random Forest regression model.  
The model takes **11 user‑friendly features** and returns an estimated sale price in dollars.

A clean, step‑by‑step web interface lets anyone adjust the house characteristics and see the prediction instantly.

---

## 🎯 Features

- **Step‑by‑step form** – answer one question at a time, like a configurator.
- **Dark premium UI** – glass‑morphism effect, smooth background slideshow.
- **Real‑time prediction** – Flask backend serves the model and returns results in JSON.
- **Trained on real data** – Ames Housing dataset, test R² ≈ 0.88.
- **Ready to deploy** – includes `Procfile` and `gunicorn` for one‑click hosting on Render.

---

## ❓ What questions does it ask?

The app guides you through **11 steps**:

| Step | Question | Type | Example |
|------|----------|------|---------|
| 1 | Neighborhood (location) | Dropdown | North Ames, College Creek, Old Town… |
| 2 | Above ground living area (sqft) | Number | 1500 |
| 3 | Total basement area (sqft) | Number | 800 |
| 4 | Bedrooms (above ground) | Number | 3 |
| 5 | Full bathrooms | Number | 2 |
| 6 | Half bathrooms | Number | 1 |
| 7 | Kitchens | Number | 1 |
| 8 | Fireplaces | Number | 1 |
| 9 | Garage capacity (cars) | Number | 2 |
| 10 | Garage area (sqft) | Number | 400 |
| 11 | Central Air Conditioning | Dropdown | Yes / No |

After the last question, you click **Predict Price** and the estimated value appears in the centre of the screen.

---

## 🧠 How the model works

- **Dataset**: [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (`train.csv`)
- **Preprocessing**:  
  - Numeric features: missing values filled with median, then scaled (StandardScaler)  
  - Categorical features (Neighborhood, CentralAir): missing values filled with most frequent, then one‑hot encoded
- **Model**: Random Forest Regressor (100 trees, `random_state=42`)
- **Performance**:  
  - Train R² = 0.970  
  - Test R²  = 0.881

The entire pipeline (preprocessing + model) is saved with `joblib` and loaded by the Flask server.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask, flask-cors |
| Machine Learning | scikit-learn, pandas, numpy |
| Model serialization | joblib |
| Frontend | HTML5, CSS3 (glassmorphism, slideshow), vanilla JavaScript |
| Deployment | Render (or any WSGI host) + gunicorn |

---

## 🚀 Run locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/house-price-predictor-ames.git
   cd house-price-predictor-ames
