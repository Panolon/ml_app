import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve,
                                classification_report, accuracy_score)
SHOW_BORDERS = False
sns.set_style("whitegrid")

def initiate_scores():
    if 'precision' not in st.session_state:
        st.session_state.precision = 0.01
    if 'recall' not in st.session_state:
        st.session_state.recall = 0.01
    if 'f1score' not in st.session_state:
        st.session_state.f1score = 0.01
    if 'rocaucscore' not in st.session_state:
        st.session_state.rocaucscore = 0.01
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = 0.01
    if 'true_positives' not in st.session_state:
        st.session_state.true_positives = 0.01
    if 'true_negatives' not in st.session_state:
        st.session_state.true_negatives = 0.01
    if 'false_positives' not in st.session_state:
        st.session_state.false_positives = 0.01
    if 'false_negatives' not in st.session_state:
        st.session_state.false_negatives = 0.01

initiate_scores()

def run():
    if 'uploaded_file' in st.session_state:
        data = pd.read_csv(st.session_state.uploaded_file,  delimiter=",",encoding='utf8', decimal=',')
        # copy of the data
        features = data.copy()
        st.write("**Preprocessing**")
        with st.container():
            col1,col2 = st.columns([0.8,0.2], gap='small', vertical_alignment='top',border=True)
            with col1:
                st.write("Dataset Preview:", data.head())
            with col2:
                #handle na values--------------------------------------------------------------------------------------------------------------------
                if features.isnull().sum().sum() > 0:
                    st.sidebar.header("Handle Missing Values")
                    missing_option = st.sidebar.radio(
                        "Choose how to handle missing values:",
                        ("Remove Rows", "Remove Columns", "Impute Values"),
                    )
                    if missing_option == "Remove Rows":
                        features = features.dropna()
                        st.write("Rows with missing values removed")
                        st.write(f"New Dataset shape: {features.shape}")
                    elif missing_option == "Remove Columns":
                        features = features.dropna(axis=1)
                        st.write("Columns with missing values removed")
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

        # select index if necessary--------------------------------------------------------------------------------------------------------------
        index_col = st.sidebar.multiselect("Select Index Column", 
                                           options=features.columns,
                                           max_selections=1,
                                           default=None)
        if index_col:
            features.set_index(index_col[0], inplace=True)

        # Choose label column----------------------------------------------------------------------------------------------------------------------
        target_column = st.sidebar.selectbox("Select target column",
                                             options=features.columns,
                                             index=len(features.columns)-1
                                            )
        target = features[target_column]
        features = features.drop(columns=[target_column])

        # Sidebar for Encoding Method------------------------------------------------------------------------------------------------------------
        encode_option = st.sidebar.selectbox("Choose encoding method", ("None", "Label Encoding", "One Hot Encoding"), index=1)

        if encode_option == "Label Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            le = LabelEncoder()
            for col in categorical_cols:
                features[col] = le.fit_transform(features[col])
        
        elif encode_option == "One Hot Encoding":
            categorical_cols = features.select_dtypes(include=["object"]).columns
            features = pd.get_dummies(features, columns=categorical_cols)

        # Encoding for Target-----------------------------------------------------------------------------------------------------------------------------
        le = LabelEncoder()
        target = le.fit_transform(target)
        with col2:
            st.write(f"Shape : {data.shape}")
            st.write(f"0 → {le.classes_[0]},   1 → {le.classes_[1]}")
            st.write(data[target_column].value_counts(normalize=True))

        # Sidebar for Feature Selection------------------------------------------------------------------------------------------------------------------
        feature_selection = st.sidebar.radio("Select feature selection method:", ("All Features", "Random", "Specific"))
        
        if feature_selection == "Random":
            num_features = int(0.5 * features.shape[1])  # 50% of the total features
            selected_features = np.random.choice(features.columns, num_features, replace=False)
            features = features[selected_features]
    
        elif feature_selection == "Specific":
            selected_columns = st.sidebar.multiselect("Select specific features", options=features.columns)
            if selected_columns:
                features = features[selected_columns]
                with col2:
                    st.write(f"Using specific features:")

        # extract specific features

        # Standardization option--------------------------------------------------------------------------------------------------------------------------
        standardize_data = st.sidebar.checkbox("Standardize the data", value=True)
        if standardize_data:
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)

        # Validation set

        # Train-Test Split----------------------------------------------------------------------------------------------------------------------------------
        test_size = st.sidebar.slider("Test Split Ratio %", min_value=10, max_value=90, step=1, value=25)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size/100, random_state=42)

        # Random Forest Classifier Parameters----------------------------------------------------------------------------------------------------------------------------------
        random_state = st.sidebar.slider("Random State", value=42, step=1)
        n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=100,step=1)
        max_features = st.sidebar.selectbox("Max Features", options=["sqrt", "log2",None])
        bootstrap = st.sidebar.checkbox("Bootstrap", value=True)
        oob_score = st.sidebar.checkbox("Use OOB Score", value=False)
        class_weight = st.sidebar.selectbox("Class Weight", [None, "balanced", "balanced_subsample"])
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])

        # Hyperparameter Tuning Widget
        hyper_tune = st.sidebar.checkbox("Hyper Tune Parameters")
        
        # Initialize Classifier----------------------------------------------------------------------------------------------------------------------------------
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            class_weight=class_weight,
            criterion=criterion,
        )
        #train classifier----------------------------------------------------------------------------------------------------------------------------------
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

        # Metrics Calculation----------------------------------------------------------------------------------------------------------------------------------
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_proba[:,1])
        report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).T
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype("float") / cm.sum()
        accuracy = accuracy_score(y_test, y_pred)

        st.write("**Performance Metrics**")
        with st.container():
            col1,col2,col3,col4,col5 = st.columns([.2, .2, .2, .2, .2], gap='small', vertical_alignment='top',border=SHOW_BORDERS)
            with col1:
                st.metric(label="Precision", value=f"{precision:.3f}",delta=f"{100*(precision - st.session_state.precision)/st.session_state.precision:.2f}%", border=SHOW_BORDERS)
                st.session_state.precision = precision
            with col2:
                st.metric(label="Recall", value=f"{recall:.3f}", delta=f"{100*(recall - st.session_state.recall) / st.session_state.recall:.2f}%" ,border=SHOW_BORDERS)
                st.session_state.recall = recall
            with col3:
                st.metric(label="F1 score", value=f"{f1:.3f}",delta=f"{100*(f1 - st.session_state.f1score) / st.session_state.f1score:.2f}%", border=SHOW_BORDERS)
                st.session_state.f1score = f1
            with col4:
                st.metric(label="ROC-AUC score", value=f"{roc_auc:.3f}",delta=f"{100*(roc_auc - st.session_state.rocaucscore) / st.session_state.rocaucscore:.2f}%", border=SHOW_BORDERS)
                st.session_state.rocaucscore = roc_auc
            with col5:
                st.metric(label="Accuracy", value=f"{accuracy:.2f}" ,delta=f"{100*(accuracy - st.session_state.accuracy) / st.session_state.accuracy :.2f}%", border=SHOW_BORDERS)
                st.session_state.accuracy = accuracy

        st.write("**Classification report**")
        with st.container():
            col1,col2 = st.columns([0.5, 0.5], gap='small', vertical_alignment='top',border=SHOW_BORDERS)
            with col1:
                st.dataframe(report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support":"{:.2f}"}))
            with col2:
                corr_matrix = features.corr(numeric_only=True)
                
        precisions, recalls, thresholds1 = precision_recall_curve(y_test, y_proba[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        feature_importance = classifier.feature_importances_
        importance_df = pd.DataFrame({"Feature": features.columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
        certainty = abs(y_proba[:, 1] - y_proba[:, 0])
        #plots
        with st.container():
                col1, col2 = st.columns([0.5, 0.5], gap='small', 
                                        **({"vertical_alignment": "top", "border": SHOW_BORDERS} if hasattr(st, "columns") else {}))
                with col1:
                    # Precision-Recall Plot
                    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
                    ax1.plot(thresholds1, precisions[:-1], label="Precision", color="mediumseagreen")
                    ax1.plot(thresholds1, recalls[:-1], label="Recall", color="goldenrod")
                    ax1.set_xlabel("Threshold", fontsize=11)
                    ax1.set_ylabel("Score", fontsize=11)
                    ax1.set_title("Precision & Recall vs Threshold", fontsize=13, fontweight='bold')
                    ax1.legend(loc='lower left')
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    ax1.set_facecolor("#f9f9f9")
                    fig1.patch.set_facecolor('#f0f2f6')
                    fig1.tight_layout()
                    st.pyplot(fig1)
                with col2:
                    # ROC Curve Plot
                    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                    ax2.fill_between(fpr, tpr, alpha=0.3, color='lightgreen', label=f"ROC-AUC = {roc_auc:.3f}")
                    ax2.plot(fpr, tpr, color='forestgreen', lw=2)
                    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
                    ax2.set_xlabel("False Positive Rate", fontsize=11)
                    ax2.set_ylabel("True Positive Rate", fontsize=11)
                    ax2.set_title("ROC Curve", fontsize=13, fontweight='bold')
                    ax2.legend(loc="lower right")
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    ax2.set_facecolor("#f9f9f9")
                    fig2.patch.set_facecolor('#f0f2f6')
                    fig2.tight_layout()
                    st.pyplot(fig2)
        with st.container():
            col1, col2 = st.columns([0.5, 0.5], gap='small', **({"vertical_alignment": "top", "border": SHOW_BORDERS} if hasattr(st, "columns") else {}))
            with col1:
            # Confusion Matrix
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(cm_normalized, 
                            annot=True, 
                            fmt=".2f", 
                            cmap="YlGn",
                            cbar=False,
                            annot_kws={"size": 12},
                            linewidths=0.5,
                            linecolor="black",
                            ax=ax_cm)
                ax_cm.set_title("Normalized Confusion Matrix", fontsize=13, fontweight='bold')
                fig_cm.patch.set_facecolor('#f0f2f6')
                fig_cm.tight_layout()
                st.pyplot(fig_cm)
            with col2:
                # Scatter Plot of Class Probabilities
                fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4.5))
                ax_scatter.scatter(
                    y_proba[:, 0], y_proba[:, 1], 
                    alpha=0.7, 
                    c=y_test, 
                    cmap='RdYlGn', 
                    edgecolors='k', 
                    linewidths=0.2
                )
                ax_scatter.plot([0, 1], [0.5, 0.5], linestyle='--', color='black', lw=1)
                ax_scatter.plot([0.5, 0.5], [0, 1], linestyle='--', color='black', lw=1)
                ax_scatter.set_xlabel("Probability of Class 0", fontsize=11)
                ax_scatter.set_ylabel("Probability of Class 1", fontsize=11)
                ax_scatter.set_title("Class Probability Scatter\n(colored by true labels)", fontsize=13, fontweight='bold')
                ax_scatter.grid(True, linestyle='--', alpha=0.6)
                ax_scatter.set_facecolor("#f9f9f9")
                fig_scatter.patch.set_facecolor('#f0f2f6')
                fig_scatter.tight_layout()
                st.pyplot(fig_scatter)
        with st.container():
            col1, col2 = st.columns([0.5, 0.5], gap='small',
                                **({"vertical_alignment": "top", "border": SHOW_BORDERS} if hasattr(st, "columns") else {}))
            with col1:
                # Feature Importance
                size = st.sidebar.slider("Top Features Selected", min_value=1, max_value=min(30, len(importance_df)), value=10, step=1)
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4.5))
                sns.barplot(
                    x="Importance", 
                    y="Feature", 
                    data=importance_df.iloc[:size], 
                    ax=ax_imp, 
                    color='seagreen'
                )
                ax_imp.set_title("Top Feature Importances", fontsize=13, fontweight='bold')
                ax_imp.set_xlabel("Importance", fontsize=11)
                ax_imp.set_ylabel("Feature", fontsize=11)
                ax_imp.grid(True, linestyle='--', alpha=0.6)
                ax_imp.set_facecolor("#f9f9f9")
                fig_imp.patch.set_facecolor('#f0f2f6')
                fig_imp.tight_layout()
                st.pyplot(fig_imp)
            with col2:
                # Certainty Histogram
                fig_cert, ax_cert = plt.subplots(figsize=(6, 4.5))
                ax_cert.hist(certainty, bins='auto', color="mediumseagreen", alpha=0.8, edgecolor='black')
                ax_cert.set_title("Certainty Histogram", fontsize=13, fontweight='bold')
                ax_cert.set_xlabel("Certainty (P(Class 1) - P(Class 0))", fontsize=11)
                ax_cert.set_ylabel("Frequency", fontsize=11)
                ax_cert.grid(True, linestyle='--', alpha=0.6)
                ax_cert.set_facecolor("#f9f9f9")
                fig_cert.patch.set_facecolor('#f0f2f6')
                fig_cert.tight_layout()
                st.pyplot(fig_cert)
        with st.container():
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".1f",
                cmap="viridis",
                linewidths=0.3,
                square=True,
                cbar_kws={"shrink": 0.6},
                ax=ax
            )

            ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)

        selected_columns = st.multiselect("Select specific features", options=features.columns)
        #plot_woe_line(pd.concat([features, pd.Series(target, name='Creditability')], axis=1), feature=selected_columns, target='Creditability', default_bins=5)
    else:
        st.info("Please upload a file to start data exploration!")
#run()