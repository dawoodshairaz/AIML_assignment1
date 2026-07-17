import streamlit as st
import joblib
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection")

message = st.text_area("Enter Email or SMS")

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

