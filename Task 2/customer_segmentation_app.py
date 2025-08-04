import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🛍️ Customer Segmentation", layout="centered")
st.title("🛍️ Mall Customer Segmentation App")

# Upload section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    # Show basic stats
    st.subheader("📈 Data Overview")
    st.write(df.describe())

    # Feature selection
    st.subheader("🔧 Select Features for Clustering")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    selected_features = st.multiselect(
        "Pick two numeric features (ideally: income & spending score)",
        numeric_cols,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(selected_features) != 2:
        st.warning("Please select exactly two features for 2D clustering.")
    else:
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
        st.pyplot(fig)

else:
    st.info("👈 Upload your dataset to get started.")
