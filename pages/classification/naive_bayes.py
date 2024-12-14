from imports import *

def run():
    st.subheader("Naive Bayes Classifier")
    st.write("Upload your data to train a Naive Bayes Classifier.")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file, index_col=0)
        st.write("Dataset Preview:", data.head())
        st.write(f"Dataset shape: {data.shape}")

        #copy data
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

        # Choose label column
        target_column = st.sidebar.selectbox("Select target column",
                                             options=features.columns,
                                             index=len(features.columns)-1
                                            )
        target = features[target_column]
        features = features.drop(columns=[target_column])
        

        encode_option = st.sidebar.selectbox("Choose encoding method", ("None", "Label Encoding", "One Hot Encoding"), index=1)
        if encode_option == "Label Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            le = LabelEncoder()
            for col in categorical_cols:
                features[col] = le.fit_transform(features[col])
        elif encode_option == "One Hot Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            features = pd.get_dummies(features, columns=categorical_cols)

        # Encode target
        le = LabelEncoder()
        target = le.fit_transform(target)

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
                st.write(f"Using specific features")

        # Standardization option
        standardize_data = st.sidebar.checkbox("Standardize the data", value=True)
        if standardize_data:
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)


        test_size = st.sidebar.slider("Test Split Ratio %", min_value=10, max_value=90, step=1, value=25)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size / 100, random_state=42)

        classifier = GaussianNB()     
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

         # Metrics Calculation
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_proba[:,1])
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype("float") / cm.sum()
        precisions, recalls, thresholds1 = precision_recall_curve(y_test, y_proba[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        certainty = abs(y_proba[:, 1] - y_proba[:, 0])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance Metrics")
            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")
            st.write(f"F1 Score: {f1:.3f}")
            st.write(f"ROC AUC score: {roc_auc:.3f}")
            st.write(cm)
        with col2:
            st.write("Classification Report")
            st.dataframe(report)

        #plots
        col1, col2 = st.columns(2)
        with col1:
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm_normalized, annot=True, 
                        fmt=".2f", 
                        cmap="viridis",
                        cbar=False, 
                        annot_kws={"size":14},
                        linewidths=0.5,
                        linecolor="black",
                        ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with col2:
            # Scatter Plot: (Probabilities)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(y_proba[:, 0], y_proba[:, 1], alpha=0.6, c=y_test)
            ax.plot([0,1],[.5,.5], linestyle='dashed',color='black')
            ax.plot([0.5,0.5],[0,1], linestyle='dashed',color='black')
            ax.set_xlabel("Probability of Class 0")
            ax.set_ylabel("Probability of Class 1")
            ax.set_title("Scatter Plot of Class Probabilities")
            ax.grid()
            st.pyplot(fig)

        with col1:
            # Precision-Recall Plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(thresholds1, precisions[:-1], label="Precision", color="b")
            ax.plot(thresholds1, recalls[:-1], label="Recall", color="r")
            ax.set_xlabel("Threshold")
            ax.set_title("Precision-Recall Curve")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        with col2:
            # ROC-AUC Curve
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid()
            st.pyplot(fig)


        with col1:
            # Certainty Plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.hist(certainty, bins='auto', color="purple", alpha=0.7)
            ax.set_title("Certainty Histogram")
            ax.set_xlabel("Certainty (Class 1 Prob - Class 0 Prob)")
            ax.set_ylabel("Frequency")
            ax.grid()
            st.pyplot(fig)
