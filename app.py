import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load models and vectorizer
task_classifier = joblib.load("task_classifier.pkl")
priority_model = joblib.load("priority_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# App UI
st.set_page_config(page_title="AI Task Management Dashboard", layout="wide")
st.title("📊 AI-Powered Task Management System")

st.markdown("This app classifies tasks and predicts their priority using AI models.")

# Sidebar input
st.sidebar.header("📝 Enter Task Details")
task_description = st.sidebar.text_area("Task Description", height=150)

# Prediction output area
if st.sidebar.button("Predict"):
    if task_description.strip() == "":
        st.warning("Please enter a task description.")
    else:
        # Predict task category
        task_category = task_classifier.predict([task_description])[0]

        # Vectorize input for priority model
        X_vectorized = tfidf.transform([task_description])
        predicted_priority = priority_model.predict(X_vectorized)[0]

        st.subheader("✅ Prediction Results")
        st.success(f"**Task Category:** {task_category}")
        st.info(f"**Predicted Priority:** {predicted_priority}")

        # Store predictions in session state for analysis
        if "predictions" not in st.session_state:
            st.session_state["predictions"] = []

        st.session_state["predictions"].append({
            "Description": task_description,
            "Category": task_category,
            "Priority": predicted_priority
        })

# Show dashboard visuals if predictions exist
if "predictions" in st.session_state and len(st.session_state["predictions"]) > 0:
    df = pd.DataFrame(st.session_state["predictions"])

    st.markdown("---")
    st.header("📈 Dashboard Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Category Distribution")
        cat_counts = df["Category"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        st.subheader("Priority Level Count")
        fig2, ax2 = plt.subplots()
        df["Priority"].value_counts().plot(kind="bar", ax=ax2, color="skyblue")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Priority")
        st.pyplot(fig2)

    with st.expander("📋 View Prediction History"):
        st.dataframe(df[::-1], use_container_width=True)

