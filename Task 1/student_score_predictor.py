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

    st.subheader("📊 Data Preview")
    st.write(df.head())
    st.write(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Expected columns
    required_columns = {
        "StudyHours": None,
        "SleepHours": None,
        "Participation": None,
        "PreviousScores": None,
        "FinalScore": None
    }

    df_cols = df.columns.tolist()
    missing = [col for col in required_columns if col not in df_cols]

    if missing:
        st.warning(f"⚠️ Missing columns: {', '.join(missing)}. Please make sure they are named correctly.")
        for col in missing:
            selected = st.selectbox(f"Map your column for '{col}':", df_cols, key=col)
            required_columns[col] = selected
        for key, val in required_columns.items():
            if val:
                df[key] = df[val]
    else:
        for col in required_columns:
            required_columns[col] = col

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap")
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # 📊 Pie + Bar Charts
    st.subheader("📊 3D-style Data Insights")

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

    # Feature selection
    st.subheader("⚙️ Feature Selection")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "FinalScore" in numeric_cols:
        numeric_cols.remove("FinalScore")

    input_features = st.multiselect("Select input features (at least 2):", numeric_cols, default=numeric_cols)

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
