import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load models
task_classifier = joblib.load("task_classifier.pkl")
priority_model = joblib.load("priority_model.pkl")

# Page config
st.set_page_config(page_title="AI Task Dashboard", layout="wide")

# --- Two Column Layout ---
left_col, right_col = st.columns([1, 3])

# --- LEFT COLUMN: Task Input ---
with left_col:
    st.markdown("### 📝 Enter Task Description")
    task_description = st.text_area("Type your task below:", placeholder="e.g. Prepare monthly report", height=150)

# --- RIGHT COLUMN: Inputs + Output ---
with right_col:
    st.title("🔍 Task Classifier & Priority Predictor")

    # Inputs
    user_workload = st.slider("User Workload (0–20)", 0.0, 20.0, 10.0)
    behavior_score = st.slider("Behavior Score (0–1)", 0.0, 1.0, 0.7)
    completion_status_options = ["Not Started", "In Progress", "Completed"]
    completion_status = st.selectbox("Completion Status", completion_status_options)
    estimated_duration = st.number_input("Estimated Duration (minutes)", min_value=1, value=30)
    days_until_due = st.number_input("Days Until Due", min_value=0, value=3)

    if st.button("🚀 Classify and Prioritize"):
        if isinstance(task_description, str) and task_description.strip():
            try:
                # Predict task category
                cleaned_desc = task_description.strip().lower()
                task_category = task_classifier.predict([cleaned_desc])[0]

                # Prepare features
                input_df = pd.DataFrame([{
                    "user_workload": user_workload / 20,
                    "user_behavior_score": behavior_score,
                    "completion_status": completion_status,
                    "estimated_duration_min": estimated_duration,
                    "days_until_due": days_until_due
                }])

                # Encode categorical
                le = LabelEncoder()
                le.fit(completion_status_options)
                input_df["completion_status"] = le.transform(input_df["completion_status"])

                # Predict priority
                predicted_priority = priority_model.predict(input_df)[0]
                priority_map = {0: "Low", 1: "Medium", 2: "High"}
                priority_label = priority_map.get(predicted_priority, "Unknown")

                # Output Results
                st.success(f"**Predicted Task Category:** {task_category}")
                st.success(f"**Predicted Priority Level:** {predicted_priority} ({priority_label})")

                # Charts
                chart1, chart2 = st.columns(2)

                with chart1:
                    fig1, ax1 = plt.subplots()
                    ax1.pie([user_workload, 20 - user_workload],
                            labels=["Workload", "Free Capacity"],
                            autopct='%1.1f%%', colors=["red", "lightgrey"], startangle=90)
                    ax1.axis("equal")
                    st.pyplot(fig1)

                with chart2:
                    fig2, ax2 = plt.subplots()
                    ax2.pie([behavior_score, 1 - behavior_score],
                            labels=["Behavior Score", "Remaining"],
                            autopct='%1.1f%%', colors=["green", "lightgrey"], startangle=90)
                    ax2.axis("equal")
                    st.pyplot(fig2)

                # Summary Table
                st.markdown("### 📋 Summary")
                summary_df = pd.DataFrame({
                    "Metric": [
                        "Task Description", "Task Category", "Priority Level",
                        "User Workload", "Behavior Score", "Completion Status",
                        "Estimated Duration (min)", "Days Until Due"
                    ],
                    "Value": [
                        task_description, task_category, f"{predicted_priority} ({priority_label})",
                        f"{user_workload}/20", f"{behavior_score}", completion_status,
                        estimated_duration, days_until_due
                    ]
                })
                st.dataframe(summary_df)

            except Exception as e:
                st.error("❌ An error occurred during prediction.")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a valid task description.")
