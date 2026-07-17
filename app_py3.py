import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection")

# Initialize session state
if "message" not in st.session_state:
    st.session_state.message = ""

# Text area
message = st.text_area(
    "Enter Email or SMS",
    key="message"
)

# Two buttons side by side
col1, col2 = st.columns(2)

with col1:
    predict = st.button("Predict")

with col2:
    clear = st.button("Clear All")

# Clear button
if clear:
    st.session_state.message = ""
    st.rerun()

# Predict button
if predict:
    if st.session_state.message.strip() == "":
        st.warning("Please enter a message.")
    else:
        vector = vectorizer.transform([st.session_state.message])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
