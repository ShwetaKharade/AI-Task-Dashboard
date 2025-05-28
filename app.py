import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load models
task_classifier = joblib.load("task_classifier.pkl")
priority_model = joblib.load("priority_model.pkl")

# Page configuration
st.set_page_config(page_title="AI Task Prioritizer", layout="wide")
st.title("🧠 AI Task Classification & Prioritization Dashboard")

# Two-column layout
col1, col2 = st.columns([1, 2])

# ----- LEFT SIDE: User Input -----
with col1:
    st.header("📥 Input Task Details")

    task_description = st.text_area("Task Description", placeholder="e.g., Schedule client meeting and send follow-up email")

    user_workload = st.slider("User Workload (0–20)", 0.0, 20.0, 10.0)
    behavior_score = st.slider("Behavior Score (0–1)", 0.0, 1.0, 0.68)
    completion_status_options = ["Not Started", "In Progress", "Completed"]
    completion_status = st.selectbox("Completion Status", completion_status_options)

    estimated_duration = st.number_input("Estimated Duration (minutes)", min_value=1, value=30)
    days_until_due = st.number_input("Days Until Due", min_value=0, value=3)

    predict_button = st.button("🚀 Classify and Prioritize")

# ----- RIGHT SIDE: Output Dashboard -----
with col2:
    if predict_button:
        if isinstance(task_description, str) and task_description.strip():
            try:
                cleaned_description = task_description.strip().lower()
                task_category = task_classifier.predict([cleaned_description])[0]

                input_features = pd.DataFrame([{
                    "user_workload": user_workload / 20,
                    "user_behavior_score": behavior_score,
                    "completion_status": completion_status,
                    "estimated_duration_min": estimated_duration,
                    "days_until_due": days_until_due
                }])

                # Consistent encoding
                le_status = LabelEncoder()
                le_status.fit(completion_status_options)
                input_features["completion_status"] = le_status.transform(input_features["completion_status"])

                predicted_priority = priority_model.predict(input_features)[0]
                priority_map = {0: "High", 1: "Medium", 2: "Low"}
                priority_label = priority_map.get(predicted_priority, "Unknown")

                # Output section
                st.header("📊 Prediction Results")
                st.success(f"**Task Category:** {task_category}")
                st.success(f"**Priority Level:** {predicted_priority} ({priority_label})")

                # Charts
                st.subheader("📈 Visual Summary")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    fig1, ax1 = plt.subplots()
                    ax1.pie([user_workload, 20 - user_workload],
                            labels=["Current Workload", "Available Capacity"],
                            autopct='%1.1f%%', colors=["red", "lightgrey"], startangle=90)
                    ax1.axis("equal")
                    st.pyplot(fig1)

                with chart_col2:
                    fig2, ax2 = plt.subplots()
                    ax2.pie([behavior_score, 1 - behavior_score],
                            labels=["Behavior Score", "Remaining"],
                            autopct='%1.1f%%', colors=["green", "lightgrey"], startangle=90)
                    ax2.axis("equal")
                    st.pyplot(fig2)
                # Additional Visuals
                st.markdown("### 📈 Additional Insights")
                extra_col1, extra_col2 = st.columns(2)

# Bar chart: Urgency
                with extra_col1:
                    urgency_score = max(0, 10 - days_until_due)  # Inverted scale: lower days = higher urgency
                    fig3, ax3 = plt.subplots()
                    ax3.barh(["Urgency"], [urgency_score], color='orange')
                    ax3.set_xlim(0, 10)
                    ax3.set_xlabel("Urgency (0=Low, 10=High)")
                    st.pyplot(fig3)

# Duration vs Standard Day
                with extra_col2:
                    duration_hours = estimated_duration / 60
                    fig4, ax4 = plt.subplots()
                    ax4.bar(["Duration (hr)"], [duration_hours], color='purple')
                    ax4.axhline(y=8, color='gray', linestyle='--', label="8-hr Work Day")
                    ax4.set_ylim(0, 10)
                    ax4.legend()
                    st.pyplot(fig4)

                # Summary Table
                st.subheader("🧾 Task Summary")
                summary_df = pd.DataFrame({
                    "Metric": [
                        "Task Category", "Priority Level", "User Workload",
                        "Behavior Score", "Completion Status",
                        "Duration (min)", "Days Until Due"
                    ],
                    "Value": [
                        task_category, f"{predicted_priority} ({priority_label})", f"{user_workload}/20",
                        f"{behavior_score}", completion_status,
                        estimated_duration, days_until_due
                    ]
                })
                st.dataframe(summary_df)

            except Exception as e:
                st.error("❌ An error occurred while processing your request.")
                st.exception(e)
        else:
            st.warning("⚠️ Please enter a valid task description.")
