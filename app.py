import streamlit as st
from imports import *

# Page configuration
st.set_page_config(page_title="ML App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
selected_task = st.sidebar.radio("Select a Task", ["Home", "Classification", "Clustering", "Regression","Dimensionality Reduction"])

if selected_task == "Home":
    st.title("Machine Learning with scikit-learn")
    st.write("Select a task from the sidebar.")

#########################
# Classification task  ##
#########################
elif selected_task == "Classification":
    from pages.classification import random_forest, svm, knn, naive_bayes, logistic_regression
    st.subheader("Choose a Classification task to proceed: ")

    # Select Classifier
    selected_model = st.radio("Select a Model", ["Random Forest", "SVM", "KNN", "Naive Bayes", "Logistic Regression"])
    if selected_model == "Random Forest":
        random_forest.run()

    if selected_model == "KNN":
        knn.run()

    if selected_model == "Naive Bayes":
        naive_bayes.run()

    if selected_model == "SVM":
        svm.run()

    if selected_model == "Logistic Regression":
        logistic_regression.run()

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












