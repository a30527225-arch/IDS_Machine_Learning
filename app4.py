import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="IDS USING MACHINE LEARNING", layout="wide")

# ==============================
# CSS (ULTRA UI)
# ==============================
st.markdown("""
<style>
[data-testid="stSidebar"] { width: 220px; }
[data-testid="stSidebar"] > div:first-child { width: 220px; }

.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1e1e2f, #2c2c54);
    color: white;
    text-align: center;
}

.marquee {
    width: 100%;
    white-space: nowrap;
    animation: marquee 12s linear infinite;
    color: #00C853;
    font-weight: bold;
}

@keyframes marquee {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD
# ==============================
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")
features = joblib.load("models/features.joblib")

# ==============================
# HEADER
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#00C853;'>IDS USING MACHINE LEARNING</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div class="marquee">
🚀 MUHAMMAD ASIF RIAZ | MUHAMMAD SHAHID | MUHAMMAD USMAN | HAMMAD | LIQAT ALI 🚀
</div><hr>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "🔍 Live Prediction", "📁 CSV Upload", "📊 Analytics", "🧠 Explain AI", "📉 Confusion Matrix", "ℹ️ About"]
)

# ==============================
# RISK
# ==============================
def risk(conf):
    if conf > 0.9:
        return "🔴 High Risk"
    elif conf > 0.7:
        return "🟠 Medium Risk"
    return "🟢 Low Risk"

# ==============================
# DASHBOARD
# ==============================
if menu == "🏠 Dashboard":
    col1, col2, col3 = st.columns(3)

    col1.markdown('<div class="card">🤖 Model<br>Random Forest</div>', unsafe_allow_html=True)
    col2.markdown('<div class="card">📡 Detection<br>Normal / Attack</div>', unsafe_allow_html=True)
    col3.markdown('<div class="card">🟢 Status<br>Active</div>', unsafe_allow_html=True)

# ==============================
# LIVE PREDICTION + RANDOM
# ==============================
elif menu == "🔍 Live Prediction":
    st.subheader("🔍 Live AI Detection")

    if st.button("🎲 Generate Random Data"):
        st.session_state["rand"] = np.random.randint(100, 50000, size=len(features))

    inputs = []
    cols = st.columns(3)

    for i, f in enumerate(features):
        val = 0.0
        if "rand" in st.session_state:
            val = float(st.session_state["rand"][i])

        with cols[i % 3]:
            v = st.number_input(f, value=val)
            inputs.append(v)

    if st.button("🚀 Predict"):
        arr = scaler.transform([inputs])

        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0]
        conf = np.max(prob)

        if pred == 1:
            st.error(f"🚨 Attack ({conf*100:.2f}%) | {risk(conf)}")
        else:
            st.success(f"🟢 Normal ({conf*100:.2f}%) | {risk(conf)}")

        # Explain (local importance)
        st.subheader("🧠 Why this prediction?")
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-5:]

        for i in reversed(top_idx):
            st.write(f"{features[i]} → {imp[i]:.4f}")

# ==============================
# CSV
# ==============================
elif menu == "📁 CSV Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        X = scaler.transform(df[features])
        preds = model.predict(X)
        probs = model.predict_proba(X).max(axis=1)

        df["Prediction"] = ["Attack" if p else "Normal" for p in preds]
        df["Confidence"] = probs * 100

        st.dataframe(df)
        st.download_button("⬇ Download", df.to_csv(index=False), "results.csv")

# ==============================
# ANALYTICS
# ==============================
elif menu == "📊 Analytics":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        X = scaler.transform(df[features])
        preds = model.predict(X)

        labels = ["Attack" if p else "Normal" for p in preds]
        counts = pd.Series(labels).value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig2)

# ==============================
# EXPLAIN AI
# ==============================
elif menu == "🧠 Explain AI":
    imp = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values(by="Importance", ascending=False)

    st.dataframe(df_imp)

    fig, ax = plt.subplots()
    ax.barh(df_imp["Feature"][:10], df_imp["Importance"][:10])
    st.pyplot(fig)

# ==============================
# CONFUSION MATRIX
# ==============================
elif menu == "📉 Confusion Matrix":
    file = st.file_uploader("Upload labeled CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "Label" in df.columns:
            y_true = df["Label"].apply(lambda x: 1 if x != "BENIGN" else 0)
            X = scaler.transform(df[features])
            y_pred = model.predict(X)

            cm = confusion_matrix(y_true, y_pred)

            st.write("Confusion Matrix:")
            st.write(cm)

# ==============================
# ABOUT
# ==============================
elif menu == "ℹ️ About":
    st.markdown("""
## 💀 GOD MODE IDS

### 🚀 Features
- AI Detection
- Random Input Generator
- CSV Analysis
- Visualization
- Explainable AI
- Confusion Matrix

### 🎯 Objective
Detect cyber threats using intelligent machine learning.

---

👨‍💻 **Muhammad Asif Riaz**  
F22BDATS1M02032
""")