import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# Load models and vectorizer
task_classifier = joblib.load("task_classifier.pkl")
priority_model = joblib.load("priority_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Set up page
st.set_page_config(page_title="AI Task Management Dashboard", layout="wide")
st.title("📊 AI-Powered Task Management System")

st.markdown("Automatically classify tasks and predict their priority using AI models.")

# Sidebar Inputs
st.sidebar.header("📝 Enter Task Details")
task_description = st.sidebar.text_area("Task Description", height=150)

departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Product", "Support"]
department = st.sidebar.selectbox("Department", departments)

assignees = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
assignee = st.sidebar.selectbox("Assign To", assignees)

deadline = st.sidebar.date_input("Deadline", min_value=datetime.today())

# Prediction Button
if st.sidebar.button("Predict"):
    if task_description.strip() == "":
        st.warning("Please enter a task description.")
    else:
        # Predict task category
        task_category = task_classifier.predict([task_description])[0]

        # Predict priority
        X_vectorized = tfidf.transform([task_description])
        predicted_priority = priority_model.predict(X_vectorized)[0]

        st.subheader("✅ Prediction Results")
        st.success(f"**Task Category:** {task_category}")
        st.info(f"**Predicted Priority:** {predicted_priority}")

        # Store results in session
        if "predictions" not in st.session_state:
            st.session_state["predictions"] = []

        st.session_state["predictions"].append({
            "Description": task_description,
            "Category": task_category,
            "Priority": predicted_priority,
            "Department": department,
            "Assignee": assignee,
            "Deadline": deadline.strftime("%Y-%m-%d")
        })

# Show dashboard
if "predictions" in st.session_state and len(st.session_state["predictions"]) > 0:
    df = pd.DataFrame(st.session_state["predictions"])

    st.markdown("---")
    st.header("📊 Task Dashboard Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Category Distribution")
        cat_counts = df["Category"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        st.subheader("Priority Distribution")
        fig2, ax2 = plt.subplots()
        df["Priority"].value_counts().plot(kind="bar", ax=ax2, color="skyblue")
        ax2.set_xlabel("Priority")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    st.markdown("### 📋 Task Assignment Table")
    st.dataframe(df[::-1], use_container_width=True)
