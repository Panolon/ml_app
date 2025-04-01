from imports import *

def run():
    st.subheader("Random Forest Classifier")
    st.write("Upload your data to train a Random Forest Classifier.")

    # Upload Dataset
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file,  delimiter=";")
        st.write("Dataset Preview:", data.head())
        st.write(f"Dataset shape: {data.shape}")

        # copy of the data
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
                st.write(f"New Dataset shape: {features.shape}")
            elif missing_option == "Remove Columns":
                features = features.dropna(axis=1)
                st.write("Columns with missing values removed:")
                st.write(f"New Dataset shape: {features.shape}")
            elif missing_option == "Impute Values":
                impute_option = st.sidebar.selectbox(
                    "Choose imputation method:", ("Mean", "Median", "Mode", "Custom Value")
                )
    
                if impute_option == "Mean":
                    features = features.fillna(features.mean())
                    st.write("Missing values imputed with Mean:")
                    st.write(f"New Dataset shape: {features.shape}")
    
                elif impute_option == "Median":
                    features = features.fillna(features.median())
                    st.write("Missing values imputed with Median:")
                    st.write(f"New Dataset shape: {features.shape}")

                elif impute_option == "Mode":
                    features = features.fillna(features.mode().iloc[0])
                    st.write("Missing values imputed with Mode:")
                    st.write(f"New Dataset shape: {features.shape}")
    
                elif impute_option == "Custom Value":
                    custom_value = st.sidebar.number_input("Enter custom value for imputation:")
                    features = features.fillna(custom_value)
                    st.write(f"Missing values imputed with custom value: {custom_value}")
                    st.write(f"New Dataset shape: {features.shape}")

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

        # Sidebar for Encoding Method
        encode_option = st.sidebar.selectbox("Choose encoding method", ("None", "Label Encoding", "One Hot Encoding"), index=1)

        if encode_option == "Label Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            le = LabelEncoder()
            for col in categorical_cols:
                features[col] = le.fit_transform(features[col])
        
        elif encode_option == "One Hot Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            features = pd.get_dummies(features, columns=categorical_cols)

        # Encoding for Target
        le = LabelEncoder()
        target = le.fit_transform(target)
        st.write(f"Label Encoding Mapping: 0 → {le.classes_[0]},   1 → {le.classes_[1]}")

        # Sidebar for Feature Selection
        feature_selection = st.sidebar.radio("Select feature selection method:", ("All Features", "Random", "Specific"))
        
        if feature_selection == "Random":
            num_features = int(0.5 * features.shape[1])  # 60% of the total features
            selected_features = np.random.choice(features.columns, num_features, replace=False)
            features = features[selected_features]
        
        elif feature_selection == "All Features":
            st.write("Using all features")
    
        elif feature_selection == "Specific":
            selected_columns = st.sidebar.multiselect("Select specific features", options=features.columns)
            if selected_columns:
                features = features[selected_columns]
                st.write(f"Using specific features:")

        # extract specific features

        # Standardization option
        standardize_data = st.sidebar.checkbox("Standardize the data", value=True)
        if standardize_data:
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)

        # Validation set

        # Train-Test Split
        test_size = st.sidebar.slider("Test Split Ratio %", min_value=10, max_value=90, step=1, value=25)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size/100, random_state=42)

        # Random Forest Classifier Parameters
        random_state = st.sidebar.slider("Random State", value=42, step=1)
        n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=100,step=1)
        max_features = st.sidebar.selectbox("Max Features", options=["sqrt", "log2",None])
        bootstrap = st.sidebar.checkbox("Bootstrap", value=True)
        oob_score = st.sidebar.checkbox("Use OOB Score", value=False)
        class_weight = st.sidebar.selectbox("Class Weight", [None, "balanced", "balanced_subsample"])
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])

        # Hyperparameter Tuning Widget
        hyper_tune = st.sidebar.checkbox("Hyper Tune Parameters")

        # Hyperparameter Tuning Logic

        
        # Initialize Classifier
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            class_weight=class_weight,
            criterion=criterion,
        )
        #train classifier
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

        # Metrics Calculation
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_proba[:,1])
        report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).T
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype("float") / cm.sum()

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Performance Metrics")
            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")
            st.write(f"F1 Score: {f1:.3f}")
            st.write(f"ROC-AUC Score: {roc_auc:.3f}")
            st.write(confusion_matrix(y_test,y_pred))
        with col2:
            st.write("Classification report")
            st.dataframe(report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support":"{:.2f}"}))
            
            
        precisions, recalls, thresholds1 = precision_recall_curve(y_test, y_proba[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        feature_importance = classifier.feature_importances_
        importance_df = pd.DataFrame({"Feature": features.columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
        certainty = abs(y_proba[:, 1] - y_proba[:, 0])

        #plots
        col1, col2 = st.columns(2)
        with col1:
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm_normalized, annot=True, 
                        fmt=".2f", 
                        cmap="YlGn",
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
            ax.scatter(y_proba[:, 0], y_proba[:, 1], alpha=0.6, c=y_test, cmap='RdYlGn')
            ax.plot([0,1],[.5,.5], linestyle='dashed',color='black')
            ax.plot([0.5,0.5],[0,1], linestyle='dashed',color='black')
            ax.set_xlabel("Probability of Class 0")
            ax.set_ylabel("Probability of Class 1")
            ax.set_title("Scatter Plot of Class Probabilities\ncoloured by true values")
            ax.grid()
            st.pyplot(fig)

        with col1:
            # Precision-Recall Plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(thresholds1, precisions[:-1], label="Precision", color="g")
            ax.plot(thresholds1, recalls[:-1], label="Recall", color="y")
            ax.set_xlabel("Threshold")
            ax.set_title("Precision-Recall Curve")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        with col2:
            # ROC-AUC Curve
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color='green')
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        with col1:
            # Feature Importance
            size = st.slider("Top Features Selected", min_value=1, max_value=30, value=10,step=1)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df.iloc[0:size,], ax=ax, color='g')
            ax.set_title("Feature Importance")
            ax.grid()
            st.pyplot(fig)

        with col2:
            # Certainty Plot
            fig, ax = plt.subplots(figsize=(8,6))
            ax.hist(certainty, bins='auto', color="green", alpha=0.8)
            ax.set_title("Certainty Histogram")
            ax.set_xlabel("Certainty (Class 1 Prob - Class 0 Prob)")
            ax.set_ylabel("Frequency")
            ax.grid()
            st.pyplot(fig)


