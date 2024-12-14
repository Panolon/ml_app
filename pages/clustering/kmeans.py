from imports import *

def run():
    st.subheader("K Means Clustering")
    st.write("Upload your data to train a K Means algorithm.")
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())
        st.write(f"Dataset shape: {data.shape}")

        #handle na values

        features = data.copy()

        # select index if necessary
        index_col = st.sidebar.multiselect("Select Index Column", 
                                           options=features.columns,
                                           max_selections=1,
                                           default=None)
        if index_col:
            features.set_index(index_col[0], inplace=True)

        
        encode_option = st.sidebar.selectbox("Choose encoding method", ("None", "Label Encoding", "One Hot Encoding"), index=1)
        if encode_option == "Label Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            le = LabelEncoder()
            for col in categorical_cols:
                features[col] = le.fit_transform(features[col])
        elif encode_option == "One Hot Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            features = pd.get_dummies(features, columns=categorical_cols)

        # Sidebar for Feature Selection
        feature_selection = st.sidebar.radio("Select feature selection method:", ("All Features", "Random", "Specific"))
        
        if feature_selection == "Random":
            num_features = int(0.5 * features.shape[1])  # 60% of the total features
            selected_features = np.random.choice(features.columns, num_features, replace=False)
            features = features[selected_features]
            #st.write(f"Randomly selected {num_features} features:", data.head())
        
        elif feature_selection == "All Features":
            st.write("Using all features")
    
        elif feature_selection == "Specific":
            selected_columns = st.sidebar.multiselect("Select specific features", options=features.columns)
            if selected_columns:
                features = features[selected_columns]
                st.write(f"Using specific features:")

        # Standardization option
        standardize_data = st.sidebar.checkbox("Standardize the data", value=True)
        if standardize_data:
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)

        # Hyperparameter Tuning Widget
        hyper_tune = st.sidebar.checkbox("Hyper Tune Parameters")

        random_state = st.sidebar.slider("Random State", value=42, step=1)
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=30, value=8, step=1)
        n_init =  st.sidebar.slider("Number of runs with different centroid seeds", min_value=2, max_value=100, value=10, step=1)
        algorithm = st.sidebar.selectbox("Algorithm", options=["lloyd","elkan"],index=0)

        classifier = KMeans(n_clusters=n_clusters, 
                            n_init=n_init,
                            algorithm=algorithm,
                            random_state=random_state
                                          )
        y_kmeans = classifier.fit_predict(features)
        

         # Metrics Calculation
        silhouette_avg = silhouette_score(features, y_kmeans)
        silhouette_vals = silhouette_samples(features, y_kmeans)


        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Performance Metrics")
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")

        with col2:
            # Elbow Method
            distortions = []
            K = range(2, 11)
            for k in K:
                kmeans_model = KMeans(n_clusters=k, random_state=random_state)
                kmeans_model.fit(features)
                distortions.append(kmeans_model.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K, distortions, marker='o')
            ax.set_title("Elbow Method")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Distortion")
            ax.grid()
            st.pyplot(fig)
            

        #plots
        col1, col2 = st.columns(2)
        with col1:
            # Silhouette Analysis
            fig, ax = plt.subplots(figsize=(8,6))
            sns.histplot(silhouette_vals, kde=True, bins=20, ax=ax, color="green")
            ax.set_title("Silhouette Analysis")
            ax.set_xlabel("Silhouette Score")
            ax.set_ylabel("Frequency")
            ax.grid()
            st.pyplot(fig)

        with col2:
            st.pyplot(fig)

        with col1:
            # Cluster Centers Plot
            fig, ax = plt.subplots()
            sns.heatmap(pd.DataFrame(classifier.cluster_centers_, columns=features.columns), cmap="coolwarm", annot=True, ax=ax)
            ax.set_title("Cluster Centers Heatmap")
            st.pyplot(fig)

 