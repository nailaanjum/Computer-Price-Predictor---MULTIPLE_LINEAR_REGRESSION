import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Page config (must be first Streamlit command) ----------
st.set_page_config(page_title="Computer Price Predictor", page_icon="💻", layout="centered")

# ---------- 1. LOAD ----------
@st.cache_data
def load_data(path="Data/Computers.csv"):
    df = pd.read_csv(path)
    return df

# ---------- 2. PREPARE ----------
def prepare_features(df, cols):
    X = sm.add_constant(df[cols])
    y = df["price"]
    return X, y

# ---------- 3. TRAIN ----------
@st.cache_resource
def fit_model(_df, cols):
    X, y = prepare_features(_df, cols)
    model = sm.OLS(y, X).fit()
    return model

# ---------- 4. EVALUATE ----------
def get_metrics(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return {"MAE": mae, "RMSE": rmse}

def get_residual_df(model, X, y):
    preds = model.predict(X)
    return pd.DataFrame({"predicted": preds, "residual": model.resid, "actual": y})

def flag_unintuitive_signs(model, expected_signs):
    flags = []
    for feature, expected in expected_signs.items():
        coef = model.params.get(feature)
        if coef is None:
            continue
        actual = "+" if coef > 0 else "-"
        if actual != expected:
            flags.append(f"**{feature}**: you'd expect price to go *up* with more {feature}, "
                          f"but the model says it actually goes *down* (coefficient: {coef:.2f})")
    return flags

def price_range_breakdown(residual_df, bins=3):
    residual_df = residual_df.copy()
    residual_df["price_bin"] = pd.qcut(residual_df["actual"], bins, labels=["Low", "Mid", "High"])
    return residual_df.groupby("price_bin", observed=True)["residual"].agg(["mean", "std", "count"])

# ---------- 5. PREDICTION HELPER ----------
def predict_single(model, ram, speed, hd, screen):
    input_row = pd.DataFrame({
        "const": [1], "ram": [ram], "speed": [speed],
        "hd": [hd], "screen": [screen]
    })
    pred_summary = model.get_prediction(input_row).summary_frame(alpha=0.05)
    return pred_summary

# ---------- 6. STREAMLIT LAYOUT ----------
st.title("💻 Computer Price Predictor")

df = load_data()

tab1, tab2 = st.tabs(["🔮 Predict", "📊 How It Works"])

with tab1:
    st.write("Estimate a computer's price based on its specs.")

    ram = st.slider("🧠 RAM (MB)", 2, 64, 8)
    speed = st.slider("⚡ Speed (MHz)", 25, 100, 50)
    hd = st.slider("💾 Hard Drive (MB)", 80, 2100, 500)
    screen = st.slider("🖥️ Screen (inches)", 14, 17, 15)

    base_model = fit_model(df, ["ram", "speed", "hd", "screen"])
    result = predict_single(base_model, ram, speed, hd, screen)

    st.metric("💰 Predicted Price", f"${result['mean'][0]:,.2f}")
    st.caption(
        f"We're fairly confident the real price would fall somewhere between "
        f"${result['obs_ci_lower'][0]:,.0f} and ${result['obs_ci_upper'][0]:,.0f}."
    )

with tab2:
    st.write(
        "Curious how the prediction above was made? This tab shows what's under the hood — "
        "no stats background needed."
    )

    model_choice = st.radio(
        "Choose a model to explore",
        ["Simple model (4 features)", "Full model (6 features)"],
        horizontal=True
    )
    cols = ["ram", "speed", "hd", "screen"] if "Simple" in model_choice else \
           ["ram", "speed", "hd", "screen", "ads", "trend"]

    model = fit_model(df, cols)
    X, y = prepare_features(df, cols)

    st.subheader("🎯 What drives the price?")
    st.caption(
        "Each number below shows how much the predicted price changes when that "
        "feature goes up by one unit, holding everything else constant."
    )
    st.dataframe(model.params.rename("effect on price"))

    flags = flag_unintuitive_signs(model, {"ram": "+", "speed": "+", "hd": "+", "screen": "+"})
    if flags:
        st.warning(
            "A few things came out backwards from what we'd normally expect:\n\n"
            + "\n\n".join(flags)
        )
    else:
        st.success("All features behave the way you'd expect — more of each raises the price.")

    st.subheader("📈 How wrong is the model, and where?")
    st.caption(
        "This plot compares the price the model predicted (x-axis) to how far off it was "
        "(y-axis). Points scattered evenly around the middle line mean the model is "
        "unbiased. A pattern — like a curve, or points fanning out — means the model "
        "struggles in that price range."
    )
    resid_df = get_residual_df(model, X, y)
    fig, ax = plt.subplots()
    ax.scatter(resid_df["predicted"], resid_df["residual"], alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Error (actual − predicted)")
    st.pyplot(fig)

    st.subheader("💵 Overall accuracy")
    metrics = get_metrics(model, X, y)
    col1, col2 = st.columns(2)
    col1.metric("Average error (MAE)", f"${metrics['MAE']:,.2f}")
    col2.metric("Typical error size (RMSE)", f"${metrics['RMSE']:,.2f}")

    st.subheader("🏷️ Does it predict better for cheap vs. expensive computers?")
    st.caption(
        "We split computers into Low, Mid, and High price groups and checked the average "
        "error in each. If one group has a much bigger average error, the model is less "
        "reliable for that price range."
    )
    st.dataframe(price_range_breakdown(resid_df))