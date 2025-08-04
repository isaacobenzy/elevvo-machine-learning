import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# App title
st.title("📚 Student Final Score Predictor")

# Sidebar file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset")
    st.write(df.head())

    # Drop missing values
    df.dropna(inplace=True)

    # Correlation heatmap (numeric columns only)
    st.subheader("🔍 Correlation Heatmap (Numeric Only)")
    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(8, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # Feature and target selection
    st.subheader("⚙️ Feature Selection")
    numeric_cols = numeric_df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least one input feature and one target column.")
    else:
        target = st.selectbox("Select target variable:", numeric_cols, index=len(numeric_cols) - 1)
        input_features = st.multiselect(
            "Select input features:",
            [col for col in numeric_cols if col != target],
            default=[col for col in numeric_cols if col != target]
        )

        if input_features:
            X = df[input_features]
            y = df[target]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Choose model type
            st.subheader("🔁 Model Selection")
            use_poly = st.checkbox("Use Polynomial Regression (degree 2)")

            if use_poly:
                model = make_pipeline(PolynomialFeatures(2), LinearRegression())
                st.info("Using Polynomial Regression")
            else:
                model = LinearRegression()
                st.info("Using Linear Regression")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model evaluation
            st.subheader("📈 Model Evaluation")
            st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

            # Actual vs Predicted Plot
            st.write("📉 Predicted vs Actual Scores")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='teal', edgecolors='k')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Final Scores")
            st.pyplot(fig)

            # Custom prediction tool
            st.subheader("🧠 Predict Custom Student Score")
            custom_input = {}
            for feature in input_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].mean())
                if df[feature].dtype in ["int64", "float64"]:
                    custom_input[feature] = st.slider(f"{feature}", min_val, max_val, default_val)

            custom_df = pd.DataFrame([custom_input])
            custom_prediction = model.predict(custom_df)[0]
            st.success(f"🎯 Predicted Final Score: **{custom_prediction:.2f}**")

            # Download predictions
            st.subheader("📥 Export Predictions")
            predictions_df = X_test.copy()
            predictions_df["Actual"] = y_test.values
            predictions_df["Predicted"] = y_pred
            csv_data = predictions_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name="student_score_predictions.csv",
                mime="text/csv"
            )

        else:
            st.warning("Please select at least one input feature.")
else:
    st.info("👈 Upload your dataset to begin.")
