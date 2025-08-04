import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit setup
st.set_page_config(page_title="📚 Student Score Predictor", layout="wide")
st.title("📚 Student Final Score Predictor Dashboard")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("📥 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    # Basic dataset validation
    if df.shape[0] < 10:
        st.error("⚠️ Dataset is too small (fewer than 10 rows). Please upload a larger dataset.")
        st.stop()
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("⚠️ Dataset must have at least 2 numeric columns for analysis. Please upload a valid dataset.")
        st.stop()

    st.subheader("📊 Data Preview")
    st.write(df.head())
    st.write(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Expected columns for analysis
    required_columns = {
        "StudyHours": None,
        "SleepHours": None,
        "Participation": None,
        "PreviousScores": None,
        "FinalScore": None
    }

    df_cols = df.columns.tolist()

    # Column mapping form (popup-like)
    with st.form(key="column_mapping_form"):
        st.subheader("🗂️ Map Your Columns")
        st.write("Please map your dataset columns to the required fields for analysis.")
        
        for col in required_columns:
            # Default to None, let user choose
            required_columns[col] = st.selectbox(
                f"Select column for '{col}':",
                options=[None] + df_cols,
                key=f"map_{col}"
            )
        
        submit_mapping = st.form_submit_button("Confirm Column Mapping")

    if submit_mapping:
        # Validate all required columns are mapped
        if None in required_columns.values():
            st.error("⚠️ Please map all required columns before proceeding.")
            st.stop()

        # Check if mapped columns are valid (exist and are numeric where needed)
        for key, val in required_columns.items():
            if val not in df_cols:
                st.error(f"⚠️ Mapped column '{val}' for '{key}' does not exist in the dataset.")
                st.stop()
            if key != "FinalScore" and df[val].dtype not in [np.float64, np.int64]:
                st.warning(f"⚠️ Column '{val}' mapped to '{key}' is not numeric. This may affect analysis accuracy.")

        # Rename columns in the dataframe based on user mapping
        for key, val in required_columns.items():
            df[key] = df[val]
            if val != key:
                df.drop(columns=[val], inplace=True)

        # Correlation heatmap
        st.subheader("📌 Correlation Heatmap")
        plt.figure(figsize=(8, 4))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        # 📊 Pie + Bar Charts
        st.subheader("📊 Data Insights")

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Select a categorical column for Pie Chart", cat_cols)
            pie_data = df[cat_col].value_counts().reset_index()
            pie_data.columns = ['Category', 'Count']

            pie_fig = px.pie(
                pie_data,
                names='Category',
                values='Count',
                title=f"Distribution of {cat_col}",
                hole=0.3
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        if "Participation" in df.columns:
            try:
                df['ParticipationGroup'] = pd.cut(
                    df['Participation'],
                    bins=[0, 25, 50, 75, 100],
                    labels=['0–25%', '26–50%', '51–75%', '76–100%']
                )
                bar_data = df.groupby('ParticipationGroup')['FinalScore'].mean().reset_index()
                bar_fig = px.bar(
                    bar_data,
                    x='ParticipationGroup',
                    y='FinalScore',
                    title="🎯 Avg Final Score by Participation Level",
                    color='FinalScore',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not process Participation data: {str(e)}")

        # Feature selection
        st.subheader("⚙️ Feature Selection")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if "FinalScore" in numeric_cols:
            numeric_cols.remove("FinalScore")

        input_features = st.multiselect("Select input features (at least 2):", numeric_cols, default=numeric_cols[:2])

        if len(input_features) < 2:
            st.error("Please select at least 2 numeric features.")
            st.stop()

        X = df[input_features]
        y = df["FinalScore"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model config
        st.subheader("🔁 Model Configuration")
        use_poly = st.checkbox("Use Polynomial Regression (degree 2)")
        if use_poly:
            model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            st.info("Using Polynomial Regression")
        else:
            model = LinearRegression()
            st.info("Using Linear Regression")

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📈 Model Evaluation")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Download predictions
        pred_df = X_test.copy()
        pred_df["Actual"] = y_test.values
        pred_df["Predicted"] = y_pred
        st.download_button("📥 Download Predictions", pred_df.to_csv(index=False), file_name="student_predictions.csv")

        # 🎛️ Custom Prediction
        st.subheader("🎛️ Predict Custom Student Score")
        custom_input = {}
        for feature in input_features:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            custom_input[feature] = st.slider(f"{feature}", min_val, max_val, mean_val)

        custom_df = pd.DataFrame([custom_input])
        custom_prediction = model.predict(custom_df)[0]
        st.success(f"🎯 Predicted Final Score: **{custom_prediction:.2f}**")

else:
    st.info("👈 Upload a student dataset to begin.")