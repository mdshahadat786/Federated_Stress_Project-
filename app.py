import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIG (Original) ---
st.set_page_config(page_title="Stress Detection", layout="centered")

st.title("Student Stress Detection System")
st.write("Please share your daily routine and feelings.")

# Load model (For single prediction)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("System is ready! ")
except:
    st.error("Model file not found ")
    st.stop()

# --- 2. INPUT SECTION (Original) ---
st.header("Enter Student Details")
col1, col2 = st.columns(2)
with col1:
    study = st.slider("Study Hours", 0, 15, 5)
    sleep = st.slider("Sleep Hours", 0, 15, 7)
    social = st.slider("Social Media Usage", 0, 10, 2)
    activity = st.slider("Physical Activity", 0, 10, 1)
    extra = st.slider("Extracurricular Activities", 0, 10, 1)
with col2:
    pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    family = st.slider("Family Support (1-5)", 1, 5, 4)
    screen = st.slider("Total Screen Time", 0, 15, 4)
    financial = st.slider("Financial Stress (1-5)", 1, 5, 2)
    examfear = st.slider("Exam Anxiety (1-5)", 1, 5, 2)
    timemanagement = st.slider("Time Management (1-5)", 1, 5, 3)

# Session state to store current level for the graph
if 'current_level' not in st.session_state:
    st.session_state.current_level = None

# --- 3. PREDICTION & LIVE METRICS ---
if st.button("Predict Stress & Analyze Performance"):
    # --- ADDED: 24 HOUR CONDITION (As requested) ---
    total_hours = study + sleep + social + activity + extra
    
    if total_hours > 24:
        st.warning(f" **Data Alert:** You have logged {total_hours} hours. A day only has 24 hours. Please adjust your sliders.")
    else:
        # Single Prediction Logic
        data_input = np.array([[study, sleep, social, pressure, family, activity, screen, extra, financial, examfear, timemanagement]])
        
        # Stress score calculation (Original)
        raw_score = (pressure * 10) + (examfear * 10) + (financial * 10) - (sleep * 5) - (family * 5)
        stress_score = max(0, min(100, raw_score + 40)) 

        st.markdown("---")
        st.subheader("Analysis Results")
        
        if stress_score <= 30:
            st.session_state.current_level = "Low"
            st.success(f"Stress Level: Low ({stress_score}%)")
            st.write("You are doing well!")
        elif stress_score <= 65:
            st.session_state.current_level = "Moderate"
            st.warning(f"Stress Level: Moderate ({stress_score}%)")
            st.write("You are under some pressure. Take it easy.")
        else:
            st.session_state.current_level = "High"
            st.error(f"Stress Level: High ({stress_score}%)")
            st.write("Please take a break and seek support.")
        
        st.progress(int(stress_score))

        # --- SUGGESTIONS CODE ---
        st.subheader(" Personalized Suggestions")
        if sleep < 6: 
            st.write("• Maintain a proper sleep schedule (at least 7-8 hours).")
        if study > 8: 
            st.write("• Take regular breaks during study time.")
        if pressure > 3: 
            st.write("• Break your academic tasks into smaller steps.")
        if financial > 3: 
            st.write("• Discuss financial concerns with a trusted person.")
        if examfear > 3: 
            st.write("• Practice mock tests to build confidence.")
        if timemanagement < 3: 
            st.write("• Follow a structured daily schedule.")

        # --- 4. PERFORMANCE SECTION ---
        st.markdown("---")
        st.header(" Model Performance Report")
        st.write("Performance metrics are calculated by re-training the model on a random subset of your data.")

        try:
            # Load CSV
            df = pd.read_csv("stress.csv")
            X = df.drop("Stress", axis=1)
            y = df["Stress"]

            # 1. DATA SPLIT 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            # 2. LIVE TRAINING
            live_model = RandomForestClassifier(n_estimators=50, max_depth=5) 
            live_model.fit(X_train, y_train)
            y_pred = live_model.predict(X_test)

            # 3. METRICS CALCULATION
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

            # Percentage Tables
            perf_df = pd.DataFrame({
                "Metric Name": ["Model Accuracy", "Precision Score", "Recall Score", "F1 Score"],
                "Value (%)": [f"{acc:.2f}%", f"{prec:.2f}%", f"{rec:.2f}%", f"{f1:.2f}%"]
            })
            st.table(perf_df)

            # 4. CONFUSION MATRIX
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=False)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        except Exception as e:
            st.info("Performance data (stress.csv) not available for live analysis.")

# --- STRESS LEVEL OVERVIEW GRAPH ---
st.markdown("---")
st.subheader("Stress Level Overview")

if st.session_state.current_level:
    levels = {"Low": 0, "Moderate": 0, "High": 0}
    levels[st.session_state.current_level] = 1

    chart_data = pd.DataFrame(
        list(levels.items()),
        columns=["Stress Level", "Value"]
    )
    st.bar_chart(chart_data.set_index("Stress Level"))
else:
    st.info("Run prediction to see graph")