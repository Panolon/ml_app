from imports import *

def run():
    st.subheader("Agglomerative Clustering")
    st.write("Upload your data to train  algorithm.")
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())
        st.write(f"Dataset shape: {data.shape}")

        #handle na values

        features = data.copy()

        #handle na values
        if features.isnull().sum().sum() > 0:
            st.sidebar.header("Handle Missing Values")
            missing_option = st.sidebar.radio(
                "Choose how to handle missing values:",
                ("Remove Rows", "Remove Columns", "Impute Values"),
            )
            if missing_option == "Remove Rows":
                features = features.dropna()
                st.write("Rows with missing values removed:")
            elif missing_option == "Remove Columns":
                features = features.dropna(axis=1)
                st.write("Columns with missing values removed:")
            elif missing_option == "Impute Values":
                impute_option = st.sidebar.selectbox(
                    "Choose imputation method:", ("Mean", "Median", "Mode", "Custom Value")
                )
    
                if impute_option == "Mean":
                    features = features.fillna(features.mean())
                    st.write("Missing values imputed with Mean:")
    
                elif impute_option == "Median":
                    features = features.fillna(features.median())
                    st.write("Missing values imputed with Median:")

                elif impute_option == "Mode":
                    features = features.fillna(features.mode().iloc[0])
                    st.write("Missing values imputed with Mode:")
    
                elif impute_option == "Custom Value":
                    custom_value = st.sidebar.number_input("Enter custom value for imputation:")
                    features = features.fillna(custom_value)
                    st.write(f"Missing values imputed with custom value: {custom_value}")


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

        # if labels are known calculate adjusted rand score
        supervised = st.sidebar.checkbox("I know the labels!",value=False)
        if supervised:
            labels = st.sidebar.selectbox("Select target column",
                                             options=features.columns,
                                             index=len(features.columns)-1
                                            )

        n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=30, value=8)
        linkage_method = st.sidebar.selectbox("Linkage", options=["ward","complete","average","single"], index=0)
        metric = st.sidebar.selectbox("Metric", options=["euclidean","l1","l2","manhattan","cosine"],index=0)

        classifier = AgglomerativeClustering(n_clusters=n_clusters, 
                            metric=metric,
                            linkage=linkage_method
                                          )
        y_agglomerative = classifier.fit_predict(features)
        

         # Metrics Calculation
        silhouette_avg = silhouette_score(features, y_agglomerative)
        silhouette_vals = silhouette_samples(features, y_agglomerative)
        ch_index = calinski_harabasz_score(features, y_agglomerative)
        if supervised:
            ari = adjusted_rand_score(features[labels], y_agglomerative)
            nmi = normalized_mutual_info_score(features[labels], y_agglomerative)
            homogeneity = homogeneity_score(features[labels], y_agglomerative)
            completeness = completeness_score(features[labels], y_agglomerative)
            v_measure = v_measure_score(features[labels], y_agglomerative)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(features)

        Z = linkage(features, method=linkage_method)


        col1,col2 = st.columns(2)
        with col1:
        	# Performance metrics
            st.subheader("Performance Metrics")
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")
            st.write(f"Calinski-Harabasz Index: {ch_index:.3f}")
            if supervised:
                st.write(f"Adjusted Rand Index: {ari:.3f}")
                st.write(f"Normalized mutual info score: {nmi:.3f}")
                st.write(f"Homogeneity: {homogeneity:.3f}")
                st.write(f"Completeness: {completeness:.3f}")
                st.write(f"V-measure: {v_measure:.3f}")

        with col2:
            # Silhouette Analysis
            fig, ax = plt.subplots(figsize=(8,6))
            sns.histplot(silhouette_vals, kde=True, bins=20, ax=ax, color="green")
            ax.set_title("Silhouette Analysis")
            ax.set_xlabel("Silhouette Score")
            ax.set_ylabel("Frequency")
            ax.grid()
            st.pyplot(fig)

        #plots
        col1, col2 = st.columns(2)

        with col1:
            truncate_mode = st.slider("Truncate Mode", min_value=3, max_value=10, value=4)
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(Z, truncate_mode="level", p=truncate_mode, ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Sample Index or Cluster Size")
            ax.set_ylabel("Distance")
            st.pyplot(fig)

        with col2:
            labels = classifier.fit_predict(X_2d)
            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis', alpha=0.8)
            ax.set_title(f"Agglomerative clustering")
            ax.set_xlabel(f"Principal Component 1")
            ax.set_ylabel(f"Principal Component 2")
            ax.grid()
            st.pyplot(fig)
