# 💻 Computer Price Predictor

🔗 **[Try the live app] https://computer-price-predictor---multiplelinearregression-jokcgwuuos.streamlit.app/

A Streamlit app that predicts computer prices using multiple regression, and explains how the underlying model makes its predictions.

## What it does

- **Predict tab** — enter RAM, speed, hard drive size, and screen size to get an estimated price with a 95% prediction interval.
- **How It Works tab** — compares a simple 4-feature model against a fuller 6-feature model (adding advertising spend and market trend), showing:
  - Coefficients and which ones behave unintuitively
  - Residual plots (with plain-language explanations)
  - Overall accuracy (MAE, RMSE)
  - Whether the model predicts better for low, mid, or high-priced computers

## Why

Built as part of a multiple regression assignment analyzing the Computers.csv dataset. The goal was to go beyond a static notebook analysis and turn the same models into something interactive — anyone can test predictions and see the model's accuracy and limitations without reading raw regression output.

## Tech stack

- Python
- Streamlit
- statsmodels (OLS regression)
- pandas, numpy
- scikit-learn (evaluation metrics)
- matplotlib

## Project structure
