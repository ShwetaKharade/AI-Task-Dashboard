import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load models and vectorizer
with open("task_classifier.pkl", "rb") as f:
    task_classifier = pickle.load(f)
with open("priority_model.pkl", "rb") as f:
    priority_model = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# App title
st.title("AI Task Classification and Prioritization")

# Task description input
task_description = st.text_input("Enter Task Description", "Generate report for project")

# User Workload slider (scaled 0–20)
user_workload = st.slider("User Workload (0-1)", 0.0, 20.0, 10.0)

# Behavior score slider
behavior_score = st.slider("Behavior Score (0-1)", 0.0, 1.0, 0.68)

# Completion status
completion_status = st.selectbox("Completion Status", ["Not Started", "In Progress", "Completed"])

# Estimated duration
estimated_duration = st.number_input("Estimated Duration (minutes)", min_value=1, value=30)

# Days until due
days_until_due = st.number_input("Days Until Due", min_value=0, value=3)

# On button click: make predictions
if st.button("Classify and Prioritize"):
    # Text feature
    text_vector = tfidf.transform([task_description])
    task_category = task_classifier.predict(text_vector)[0]

    # Priority prediction
    input_features = pd.DataFrame([{
        "user_workload": user_workload / 20,  # Scale to 0–1
        "behavior_score": behavior_score,
        "completion_status": completion_status,
        "estimated_duration": estimated_duration,
        "days_until_due": days_until_due,
        "task_category": task_category
    }])

    # One-hot encode or encode category if necessary
    # (Assuming the model pipeline handles preprocessing internally)

    predicted_priority = priority_model.predict(input_features)[0]
    priority_map = {0: "Low", 1: "Medium", 2: "High"}
    priority_label = priority_map.get(predicted_priority, "Unknown")

    # Display results
    st.success(f"**Predicted Task Category (from text):** {task_category}")
    st.success(f"**Predicted Priority Level: {predicted_priority} ({priority_label})**")

    # --- Optional Dashboard Visuals ---
    st.subheader("Task Summary Dashboard")

    # Pie chart for workload/behavior split
    fig1, ax1 = plt.subplots()
    ax1.pie([user_workload, 20 - user_workload], labels=["Current Workload", "Available Capacity"],
            autopct='%1.1f%%', colors=["red", "lightgrey"], startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie([behavior_score, 1 - behavior_score], labels=["Behavior Score", "Remaining"],
            autopct='%1.1f%%', colors=["green", "lightgrey"], startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

    # Table summary
    summary_df = pd.DataFrame({
        "Metric": ["Task Category", "Priority Level", "User Workload", "Behavior Score", "Completion Status", "Duration (min)", "Days Until Due"],
        "Value": [task_category, f"{predicted_priority} ({priority_label})", f"{user_workload}/20", f"{behavior_score}", completion_status, estimated_duration, days_until_due]
    })
    st.dataframe(summary_df)

