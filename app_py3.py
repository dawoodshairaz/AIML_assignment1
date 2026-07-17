import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection")

# Store the text separately
if "text" not in st.session_state:
    st.session_state.text = ""

# Text area
message = st.text_area(
    "Enter Email or SMS",
    value=st.session_state.text
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message.")
        else:
            vector = vectorizer.transform([message])
            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")

with col2:
    if st.button("Clear All"):
        st.session_state.text = ""
        st.rerun()
