import streamlit as st
import joblib
import numpy as np

# Load models
task_classifier = joblib.load("task_classifier.pkl")
priority_model = joblib.load("priority_model.pkl")
tfidf_vectorizer = joblib.load("tfidf.pkl")

st.title("🧠 AI Task Classification and Prioritization")

task_desc = st.text_input("Enter Task Description", "Generate report for project")
user_workload = st.slider("User Workload (0-20)", 0.0, 20.0, 10.0)
behavior_score = st.slider("Behavior Score (0-1)", 0.0, 1.0, 0.5)
completion_status = st.selectbox("Completion Status", ["Not Started", "In Progress", "Completed"])
duration = st.number_input("Estimated Duration (minutes)", min_value=1, value=30)
days_until_due = st.number_input("Days Until Due", min_value=0, value=3)

if st.button("Classify and Prioritize"):
    # 1. Predict category
    text_vector = tfidf_vectorizer.transform([task_desc])
    task_category = task_classifier.predict(text_vector)[0]

    # 2. Prepare additional features for priority model
    # Convert completion_status to numeric value
    status_map = {"Not Started": 0, "In Progress": 1, "Completed": 2}
    completion_status_num = status_map[completion_status]

    # Concatenate with tfidf features
    # Convert sparse matrix to dense
    X_text = text_vector.toarray()
    X_meta = np.array([[user_workload, behavior_score, completion_status_num, duration, days_until_due]])
    X_combined = np.hstack([X_text, X_meta])

    # 3. Predict priority
    predicted_priority = priority_model.predict(X_combined)[0]

    st.success(f"🗂️ Predicted Task Category: **{task_category}**")
    st.success(f"⚡ Predicted Priority Level: **{predicted_priority}**")
