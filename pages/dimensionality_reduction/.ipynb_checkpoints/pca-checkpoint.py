from imports import *

def run():
    st.subheader("Principal Component Analysis")
    st.write("Upload your data to implement Dimensionality Reduction.")
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())
        st.write(f"Dataset shape: {data.shape}")

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
        index_col = st.sidebar.multiselect("Select Index Column", options=features.columns, max_selections=1,default=None)
        if index_col:
            features.set_index(index_col[0], inplace=True)
        
       # Sidebar for Encoding Method
        encode_option = st.sidebar.selectbox("Choose encoding method", ("None", "Label Encoding", "One Hot Encoding"), index=1)

        if encode_option == "Label Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            le = LabelEncoder()
            for col in categorical_cols:
                features[col] = le.fit_transform(features[col])
            #st.write("Label Encoded Data:", data.head())
        
        elif encode_option == "One Hot Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            features = pd.get_dummies(features, columns=categorical_cols)
            #st.write("One Hot Encoded Data:", data.head())

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
                st.write(features.columns)

        # Standardization option
        standardize_data = st.sidebar.checkbox("Standardize the data", value=True)
        if standardize_data:
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)

        # Kernel PCA Parameters
        random_state = st.sidebar.slider("Random State", value=42, step=1)
        n_components = st.sidebar.slider("Number of Components", 
                                         min_value=2, 
                                         max_value=len(features.columns), 
                                         value=len(features.columns),
                                         step=1)
        n_components_explained_var = st.sidebar.slider("% Variance Explained of Components", 
                                                      min_value=0.1, 
                                                      max_value=.99, 
                                                      value=.95,
                                                      step=.1)
        kernel = st.sidebar.selectbox("Kernel", options=['poly','rbf', 'sigmoid', 'cosine'], index=1)
        gamma = st.sidebar.slider("gamma",min_value=0., max_value=100., step=0.1, value=1.)
        degree = st.sidebar.slider("degree", min_value=0, max_value=20, step=1, value=3)
        eigen_solver = st.sidebar.selectbox("Eigen Solver", options=['auto','dense','arpack','randomized'], index=0)
        remove_zero_eig = st.sidebar.checkbox("Remove Zero Eigenvalues", value=False)
        target = st.sidebar.multiselect("Select Feature to color",
                                        options = features.columns,
                                        max_selections = 1
                                        )
    
        #Kernel PCA
        pca_kernel = KernelPCA(n_components=n_components,
                        kernel=kernel, 
                        degree=degree, 
                        gamma=gamma,
                        eigen_solver=eigen_solver,
                        remove_zero_eig=remove_zero_eig,
                        random_state=random_state)

        #Linear PCA
        pca_linear = PCA(n_components = n_components_explained_var,
                        random_state=random_state)

        X_pca_linear = pca_linear.fit_transform(features)
        pca_linear_df = pd.DataFrame(X_pca_linear, columns = [f'PCA{i+1}' for i in range(X_pca_linear.shape[1])])
        loadings = pd.DataFrame(
            pca_linear.components_.T,  # transpose the matrix of loadings
            columns=pca_linear_df.columns,  # set the columns are the principal components
            index=features.columns,  # and the rows are the original features
            )
        explained_variance_ratio_linear = np.cumsum(pca_linear.explained_variance_ratio_)
        filtered_loadings = loadings.copy()
        filtered_loadings[(filtered_loadings <= 0.5) & (filtered_loadings >= -0.4)] = np.nan
        filtered_loadings_clean = filtered_loadings.dropna(axis=0, how='all').dropna(axis=1, how='all')


        #Kernel PCA metrics
        X_pca_kernel = pca_kernel.fit_transform(features)
        pca_kernel_df = pd.DataFrame(X_pca_kernel, columns = [f'PCA{i+1}' for i in range(n_components)])
        explained_variance_ratio_kernel = pca_kernel.eigenvalues_ / np.sum(pca_kernel.eigenvalues_)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio_kernel)*100

        # plots KERNEL PCA
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f'PCA with {kernel} kernel')
            fig, ax = plt.subplots(figsize=(8,6))
            if len(target) == 1:
                pca_kernel_df['Color'] = features[target[0]].values
                scatter = ax.scatter(pca_kernel_df['PCA1'], pca_kernel_df['PCA2'], 
                                     c=pca_kernel_df['Color'].values, 
                                     cmap='viridis', alpha=1, 
                                     s=15)
                legend = ax.legend(*scatter.legend_elements(), title='Class')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                ax.set_title(f'Kernel {kernel}')
                ax.add_artist(legend)
                st.pyplot(fig)
            elif len(target) == 0:
                scatter = ax.scatter(pca_kernel_df['PCA1'], pca_kernel_df['PCA2'], c=None, cmap='viridis', alpha=1, s=15)
                legend = ax.legend(*scatter.legend_elements(), title='Class')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                ax.set_title(f'Kernel {kernel}')
                ax.add_artist(legend)
                st.pyplot(fig)
                
        with col2:
            st.write("Explained Variance Ratio")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(cumulative_variance_ratio, marker='o', linestyle='--', color='purple')
            ax.set_xticks([])
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('Cumulative Explained Variance Ratio')
            ax.grid()
            st.pyplot(fig)

        with col3:
            st.write("3D plot ")
            fig, ax = plt.subplots(figsize=(8,6))
            if len(target) == 1:
                pca_kernel_df['Color'] = features[target[0]].values
                fig = px.scatter_3d(pca_kernel_df,
                               x='PCA1',
                               y='PCA2',
                               z='PCA3',
                               color='Color')
                plt.title(f'kernel : {kernel}')
                st.plotly_chart(fig, selection_mode='points')

            elif len(target) == 0:
                fig = px.scatter_3d(pca_kernel_df,
                               x='PCA1',
                               y='PCA2',
                               z='PCA3')
                plt.title(f'kernel : {kernel}')
                st.plotly_chart(fig, selection_mode='points')

        # PLOTS LINEAR PCA
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f'Simple PCA')
            fig, ax = plt.subplots(figsize=(8,6))
            if len(target) == 1:
                pca_linear_df['Color'] = features[target[0]].values
                scatter = ax.scatter(pca_linear_df['PCA1'], pca_linear_df['PCA2'], 
                                     c=pca_linear_df['Color'].values, 
                                     cmap='viridis', alpha=1, 
                                     s=15)
                legend = ax.legend(*scatter.legend_elements(), title='Class')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                #ax.set_title(f'Kernel {kernel}')
                ax.add_artist(legend)
                st.pyplot(fig)
            elif len(target) == 2:
                pca_linear_df['Color'] = features[target[0]].values + features[target[1]].values
                scatter = ax.scatter(pca_linear_df['PCA1'], pca_linear_df['PCA2'], c=None, cmap='viridis', alpha=1, s=15)
                legend = ax.legend(*scatter.legend_elements(), title='Class')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                #ax.set_title(f'Kernel {kernel}')
                ax.add_artist(legend)
                st.pyplot(fig)
            elif len(target) == 0:
                scatter = ax.scatter(pca_linear_df['PCA1'], pca_linear_df['PCA2'], c=None, cmap='viridis', alpha=1, s=15)
                legend = ax.legend(*scatter.legend_elements(), title='Class')
                ax.set_xlabel('PCA 1')
                ax.set_ylabel('PCA 2')
                #ax.set_title(f'Kernel {kernel}')
                ax.add_artist(legend)
                st.pyplot(fig)
                
        with col2:
            st.write("Explained Variance Ratio")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(explained_variance_ratio_linear, marker='o', linestyle='--', color='purple')
            ax.set_xticks([])
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('Cumulative Explained Variance Ratio')
            ax.grid()
            st.pyplot(fig)

        with col3:
            st.write(" 3D plot ")
            fig, ax = plt.subplots(figsize=(8,6))
            if len(target) == 1:
                pca_linear_df['Color'] = features[target[0]].values
                fig = px.scatter_3d(pca_linear_df,
                               x='PCA1',
                               y='PCA2',
                               z='PCA3',
                               color='Color')
                st.plotly_chart(fig, selection_mode='points')

            elif len(target) == 0:
                fig = px.scatter_3d(pca_linear_df,
                               x='PCA1',
                               y='PCA2',
                               z='PCA3')
                st.plotly_chart(fig, selection_mode='points')


        st.write("Loadings")
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.imshow(filtered_loadings_clean, norm='linear', cmap='viridis', aspect='auto')
        # Add annotations inside the cells
        for i in range(filtered_loadings_clean.shape[0]):  # Loop through rows
            for j in range(filtered_loadings_clean.shape[1]):  # Loop through columns
                ax.text(j, i, f'{filtered_loadings_clean.iloc[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

        cbar = ax.figure.colorbar(cax, ax=ax)
        ax.set_title('Loadings')
        ax.set_xlabel('PC components')
        ax.set_ylabel('genes')
        ax.set_xticks(ticks=[i for i in range(len(filtered_loadings_clean.columns))], labels = list(filtered_loadings_clean.columns), rotation=90)
        ax.set_yticks(ticks=[i for i in range(len(filtered_loadings_clean.index))], labels = list(filtered_loadings_clean.index), rotation=0)
        ax.grid()

        st.pyplot(fig)

