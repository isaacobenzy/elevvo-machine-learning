import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Streamlit setup
st.set_page_config(page_title="🛍️ Customer Segmentation", layout="centered")
st.title("🛍️ Mall Customer Segmentation App")

# Upload section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    # Basic dataset validation
    if df.shape[0] < 10:
        st.error("⚠️ Dataset is too small (fewer than 10 rows). Please upload a larger dataset.")
        st.stop()
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("⚠️ Dataset must have at least 2 numeric columns for clustering. Please upload a valid dataset.")
        st.stop()

    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    # Show basic stats
    st.subheader("📈 Data Overview")
    st.write(df.describe())

    # Expected columns for clustering
    required_columns = {
        "AnnualIncome": None,
        "SpendingScore": None
    }

    df_cols = df.columns.tolist()

    # Column mapping form (popup-like)
    with st.form(key="column_mapping_form"):
        st.subheader("🗂️ Map Your Columns")
        st.write("Please map your dataset columns to the required fields for clustering (e.g., income and spending score).")
        
        for col in required_columns:
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

        # Check if mapped columns are valid (exist and are numeric)
        for key, val in required_columns.items():
            if val not in df_cols:
                st.error(f"⚠️ Mapped column '{val}' for '{key}' does not exist in the dataset.")
                st.stop()
            if df[val].dtype not in [np.float64, np.int64]:
                st.error(f"⚠️ Column '{val}' mapped to '{key}' must be numeric for clustering.")
                st.stop()

        # Rename columns in the dataframe based on user mapping
        for key, val in required_columns.items():
            df[key] = df[val]
            if val != key:
                df.drop(columns=[val], inplace=True)

        # Feature selection
        selected_features = ["AnnualIncome", "SpendingScore"]
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method for K
        st.subheader("📍 Elbow Method - Optimal Clusters")
        distortions = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, distortions, 'bo-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
        ax.set_title('Elbow Method For Optimal k')
        st.pyplot(fig)

        # Cluster with user-defined K
        st.subheader("🔢 K-Means Clustering")
        k = st.slider("Select number of clusters (k)", 2, 10, 5)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters

        # Cluster Visualization
        st.subheader("🎨 Cluster Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=X[selected_features[0]],
            y=X[selected_features[1]],
            hue=clusters,
            palette='Set2',
            s=100,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"Customer Segmentation (k={k})")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        st.pyplot(fig)

        # Cluster Insights
        st.subheader("📋 Cluster Averages")
        cluster_summary = df.groupby('Cluster')[selected_features].mean().round(2)
        st.dataframe(cluster_summary)

        # Export option
        st.download_button(
            label="📥 Download Clustered Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='clustered_customers_predictions.csv',
            mime='text/csv'
        )

        # BONUS: DBSCAN
        st.subheader("🧪 Bonus: DBSCAN Clustering")
        eps = st.slider("DBSCAN - eps (neighborhood size)", 0.1, 3.0, 0.5, 0.1)
        min_samples = st.slider("DBSCAN - min_samples", 2, 10, 5)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = dbscan.fit_predict(X_scaled)
        df['DBSCAN_Cluster'] = db_labels

        fig, ax = plt.subplots()
        sns.scatterplot(
            x=X[selected_features[0]],
            y=X[selected_features[1]],
            hue=db_labels,
            palette='Set1',
            s=100,
            alpha=0.7,
            ax=ax
        )
        ax.set_title("DBSCAN Clustering")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        st.pyplot(fig)

else:
    st.info("👈 Upload your dataset to get started.")