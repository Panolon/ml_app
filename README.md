# Machine Learning Streamlit Application

This repository contains a **Streamlit-based Machine Learning Application** designed to provide an intuitive and interactive interface for performing essential machine learning tasks, including **Classification**, **Clustering**, and **Regression**. It is ideal for beginners, educators, and professionals looking for a quick way to experiment with various machine learning models.

---

## Features

### 1. Homepage  
The homepage serves as the entry point to the app, providing a brief overview of the available tasks and guidance on how to use the application.

### 2. Classification  
Perform classification using popular machine learning models:
- **Random Forest Classifier**  
- **Support Vector Machine (SVM)**  
- **K-Nearest Neighbors (KNN)**  
- **Naive Bayes Classifier**
- **Logistic Regression Classifier**  


Each model has its own dedicated page with an easy-to-follow interface to:
- Upload datasets in CSV format.
- Select features and target variables.
- Configure model-specific parameters.
- View detailed classification reports with metrics like accuracy, precision, recall, and F1-score.

### 3. Clustering  
Explore unsupervised learning models to group data points into clusters. Users can upload datasets and visualize clustering results dynamically.

- **K Means**
- **DBSCAN**
- **Agglomerative Clustering**
- **Gaussian Mixtures**


### 4. Regression  
Experiment with regression models to predict continuous target variables. The app allows you to train and evaluate models interactively.

### 5. Interactive Widgets  
The app uses Streamlit’s widgets (sliders, dropdowns, and file uploaders) for a seamless user experience.

### 6. Modular Design  
Each task and model is implemented in a modular structure, making the app easy to extend or customize. 

---

## Technologies Used

- **Python**  
  The backbone of the application.
  
- **Streamlit**  
  A powerful framework for building interactive web applications.

- **Scikit-learn**  
  For implementing machine learning models.

- **Pandas**  
  For data preprocessing and manipulation.

- **Matplotlib & Seaborn**  
  For data visualization (if needed).

---

## How to Run the Application

### 1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/ml-streamlit-app.git
   cd ml-streamlit-app

# Install dependencies
   pip install -r requirements.txt

# Run the application
    streamlit run app.py
