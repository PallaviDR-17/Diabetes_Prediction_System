import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# -------------------------------
# Sidebar: About Section
# -------------------------------
st.sidebar.markdown("""
### ℹ️ About the App
This intelligent system uses a **Logistic Regression** model trained on the **PIMA Diabetes dataset** 
to predict whether a person is likely to have diabetes.
""")

# -------------------------------
# Theme Selection
# -------------------------------
st.sidebar.markdown("### 🎨 Choose Theme")
theme = st.sidebar.radio("", ("🌞 Light Mode", "🌙 Dark Mode"))

# Apply CSS dynamically
if theme == "🌙 Dark Mode":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0E1117 !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #161B22 !important;
            color: white !important;
        }
        div[data-testid="stMarkdownContainer"] {
            color: white !important;
        }
        input, select, textarea {
            background-color: #1E1E1E !important;
            color: white !important;
        }
        h1, h2, h3, h4, label {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #F5F7FA !important;
            color: black !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #E9ECEF !important;
            color: black !important;
        }
        div[data-testid="stMarkdownContainer"] {
            color: black !important;
        }
        h1, h2, h3, h4, label {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("💉 Diabetes Prediction System")

# -------------------------------
# Input Fields
# -------------------------------
st.markdown("### 🧾 Enter Patient Details")

col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies = st.number_input("Pregnancies", 0, 17, 3)
    Glucose = st.number_input("Glucose Level", 0, 200, 120)
with col2:
    BloodPressure = st.number_input("Blood Pressure", 0, 122, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
with col3:
    Insulin = st.number_input("Insulin Level", 0, 846, 79)
    BMI = st.number_input("BMI", 0.0, 67.1, 25.0)

DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
Age = st.number_input("Age", 1, 120, 33)

# -------------------------------
# Load and Prepare Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale & Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# -------------------------------
# MODEL EVALUATION (in terminal)
# -------------------------------
y_pred = model.predict(X_scaled)
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("📊 MODEL PERFORMANCE METRICS ")
print("-------------------------------------------")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"🎯 Precision: {prec:.4f}")
print(f"🔁 Recall:    {rec:.4f}")
print(f"🏆 F1 Score:  {f1:.4f}")
print("-------------------------------------------\n")

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("---")
predict_btn = st.button("🔍 Predict")

if predict_btn:
    # Prepare input
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]
    
    # Display Result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    color = "red" if result == "Diabetic" else "green"
    
    st.markdown(f"### 🧠 Prediction Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
    
    # Probability Bar Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bg_color = "#0E1117" if theme == "🌙 Dark Mode" else "white"
    text_color = "white" if theme == "🌙 Dark Mode" else "black"

    ax.bar(["Non-Diabetic", "Diabetic"], probability, color=["#2ecc71", "#e74c3c"])
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.tick_params(colors=text_color)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability", color=text_color)
    for i, val in enumerate(probability):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", color=text_color, fontsize=12, weight="bold")
    st.pyplot(fig)

    # -------------------------------
    # Dynamic ROC Curve Simulation
    # -------------------------------
    st.markdown("📈  ROC Curve")
    
    # Generate small variations around user input for simulation
    samples = np.tile(input_scaled, (200, 1))
    noise = np.random.normal(0, 0.2, samples.shape)
    X_test_simulated = samples + noise
    y_test_simulated = np.random.randint(0, 2, 200)
    
    y_pred_proba = model.predict_proba(X_test_simulated)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_simulated, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', color=text_color)
    ax2.set_ylabel('True Positive Rate', color=text_color)
    ax2.set_title('Receiver Operating Characteristic', color=text_color)
    ax2.legend(loc="lower right", facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
    fig2.patch.set_facecolor(bg_color)
    ax2.set_facecolor(bg_color)
    ax2.tick_params(colors=text_color)
    ax2.spines['bottom'].set_color(text_color)
    ax2.spines['left'].set_color(text_color)
    st.pyplot(fig2)

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Diabetes Prediction System ")
