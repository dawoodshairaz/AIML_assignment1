import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection")

# Initialize session state
if "message" not in st.session_state:
    st.session_state.message = ""

# Text area
st.text_area(
    "Enter Email or SMS",
    key="message"
)

# Buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict"):
        if st.session_state.message.strip() == "":
            st.warning("Please enter a message.")
        else:
            vector = vectorizer.transform([st.session_state.message])
            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")

with col2:
    if st.button("Clear All"):
        st.session_state.message = ""
        st.rerun()
