import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection")

# Counter used to reset the text area
if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0

message = st.text_area(
    "Enter Email or SMS",
    key=f"message_{st.session_state.clear_count}"
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
        st.session_state.clear_count += 1
        st.rerun()
