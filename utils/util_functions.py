import streamlit as st

def under_construction():
    st.title("🚧 Page Under Construction 🚧")
    st.markdown("""
    ### Oops! This page is still being built... 🛠️  
    We apologize for the inconvenience 😅  
    If you have any brilliant ideas, suggestions, or want to send us a meme,  
    👉 please visit: [github/Panolon](https://github.com/Panolon)

    Meanwhile, enjoy this construction cat 😺:
    """)
    st.image("https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif", caption="Work in progress... sort of.")

def intro():
    st.title("🤖 Welcome to the Machine Learning Playground!")
    
    st.markdown("""
    ### 🎉 Hello curious minds, data wranglers, and future ML rockstars!

    You've just stepped into a magical place where data meets models,  
    sliders tweak algorithms, and CSVs live their best lives.

    This interactive Streamlit app lets you:
    - 🧠 **Classify** like a pro (Random Forests, SVMs, KNN — oh my!)
    - 🧩 **Cluster** your way through chaos (K-Means, DBSCAN, and more)
    - 📈 **Regress** responsibly (because predicting house prices is the new cool)
    
    Whether you're:
    - 👶 A beginner exploring machine learning for the first time,
    - 🎓 An educator looking for a teaching aid,
    - 💼 A professional needing a quick sandbox...

    ...you’re in the right place.

    ⚠️ **Note:** This app is still under construction — like a robot learning to walk,  
    it's improving every day. Feel free to contribute or suggest ideas via [github/Panolon](https://github.com/Panolon)!

    Let's get clicking, tweaking, and learning! 🚀
    """)

def hyper_tune_logic():
####################################################################################################
        # Hyperparameter Tuning
        # START
        ####################################################################################################
        if hyper_tune:
            # Initialize Classifier
            model = LogisticRegression(max_iter=10000, solver='saga', class_weight='balanced')


            st.sidebar.subheader("Hyperparameter Tuning")
            st.sidebar.write("Using Grid Search for hyperparameter tuning.")

            # Scoring Metric Selection
            scoring_metric = st.sidebar.selectbox("Select scoring metric",
                options=["accuracy", "f1", "roc_auc", "precision", "recall"]
            )

            # Cross Validation Folds
            cv = st.sidebar.slider("Cross Validation Folds", 
                                   min_value=2, 
                                   max_value=min(20,int(len(X_train)*0.5)), 
                                   value=5, 
                                   step=1)
            st.sidebar.write(f"Using {cv} folds for cross-validation.")

            if scoring_metric == "f1":
                scoring = "f1"
            elif scoring_metric == "accuracy":
                scoring = "accuracy"
            elif scoring_metric == "roc_auc":
                scoring = "roc_auc"
            elif scoring_metric == "precision":
                scoring = "precision"
            elif scoring_metric == "recall":
                scoring = "recall"
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2', 'elasticnet']
                }
            
            if 'elasticnet' in param_grid['penalty']:
                param_grid['l1_ratio'] = [0.0, 0.25, 0.5, 0.75, 1.0]  # Only used for elasticnet penalty

            # Grid search with 5-fold CV
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_
            st.write(f"\nBest Parameters: {grid_search.best_params_}")
            st.write(f"Best CV score: {grid_search.best_score_:.4f}")

            st.write(pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score').head(10))

            # Model evaluation
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)

            # Metrics Calculation
            precision = precision_score(y_test, y_pred, average="binary")
            recall = recall_score(y_test, y_pred, average="binary")
            f1 = f1_score(y_test, y_pred, average="binary")
            roc_auc = None
            if len(np.unique(target)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:,1])
                fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
            report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).T
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype("float") / cm.sum()
            precisions, recalls, thresholds1 = precision_recall_curve(y_test, y_proba[:, 1])
            certainty = abs(y_proba[:, 1] - y_proba[:, 0])


        ##############################################################################################################
        # End of hyperparameter tuning logic
        ##############################################################################################################