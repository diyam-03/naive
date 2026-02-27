import streamlit as st
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------- Training function ----------
def train_naive_bayes(df, target, features, test_size):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    joblib.dump(model, "model.pkl")

    return metrics


# ---------- Streamlit UI ----------
st.title("🧠 Naive Bayes Trainer")

st.write("Upload a dataset, select target & features, and train a Gaussian Naive Bayes model.")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    target = st.selectbox("Select target column", columns)
    features = st.multiselect("Select feature columns", columns)

    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

    if st.button("Train Model"):
        if target and features:
            metrics = train_naive_bayes(df, target, features, test_size)

            st.success("Model trained successfully!")

            st.subheader("Accuracy")
            st.write(metrics["accuracy"])

            st.subheader("Classification Report")
            st.json(metrics["report"])

            st.subheader("Confusion Matrix")
            st.write(metrics["confusion_matrix"])

            with open("model.pkl", "rb") as f:
                st.download_button(
                    "Download trained model",
                    f,
                    file_name="model.pkl"
                )
        else:
            st.warning("Please select target and features.")