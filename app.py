import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Machine Learning with scikit-learn", 
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="🎲"
                   )

# Sidebar navigation
st.sidebar.title("Navigation")
selected_task = st.sidebar.radio("Select a Task", 
                                 ["Home", "Classification", "Clustering", "Regression","Dimensionality Reduction"]
                                 )

 # Upload Dataset globally
uploaded_file = st.file_uploader(
    label="Upload your CSV file",
    key="main_file_uploader",
    help="Upload a CSV file containing your dataset. The last column will be treated as the target variable.",
    accept_multiple_files=False,
    label_visibility="collapsed",
    type="csv"
)
if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

if selected_task == "Home":
    #st.title("Machine Learning with scikit-learn")
    from utils.util_functions import intro

    intro()

#########################
# Classification task  ##
#########################
elif selected_task == "Classification":
    from pages.classification import random_forest, xgboost, svm, knn, naive_bayes, logistic_regression
    from utils.util_functions import under_construction
    #st.subheader("Choose a Classification task to proceed: ")

    # Select Classifier
    logistic_regression_tab, rmforest_tab, xgboost_tab, svm_tab, knn_tab, naive_bayes_tab = st.tabs([
        "Logistic Regression", "Random Forest", "XGBoost", "SVM", "KNN", "Naive Bayes"
        ])   
    with logistic_regression_tab    :
        logistic_regression.run()
    with rmforest_tab:
        random_forest.run()
    with xgboost_tab:
        under_construction()
        #xgboost.run()
    with knn_tab:
        under_construction()
        #knn.run()
    with naive_bayes_tab:
        under_construction()
        #naive_bayes.run()
    with svm_tab:
        under_construction()
        #svm.run()
#####################
# Clustering task  ##
#####################
elif selected_task == "Clustering":
    from pages.clustering import kmeans,agglomerative
    st.subheader("Choose a Clustering task to proceed: ")
    
    # Select Clustering method
    selected_model = st.radio("Select a Model", ["K Means", "DBSCAN", "Agglomerative Clustering", "Gaussian Mixtures"])
    if selected_model == "K Means":
        kmeans.run()

    if selected_model == "Agglomerative Clustering":
    	agglomerative.run()


#####################
# Regression task  ##
#####################
elif selected_task == "Regression":
    from pages import regression
    regression.run()

elif selected_task == "Dimensionality Reduction":
    from pages.dimensionality_reduction import pca
    st.subheader("Choose a Dimensionality Reduction task to proceed: ")

    # Select task
    selected_model = st.radio("Select a Model", ["Principal Component Analysis"])
    if selected_model == "Principal Component Analysis":
        pca.run()












