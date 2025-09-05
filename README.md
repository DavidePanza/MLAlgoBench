# ML-Models Benchmarker

**[Try the App](https://davidepanza-mlalgobench-srcmain-cp6iq9.streamlit.app/)**  


**ML-Models Benchmarker** is a toy project that allows users to upload their own datasets and benchmark different machine learning algorithms. The frontend is built with Streamlit, while the models are implemented using Scikit-learn, LightGBM, and CatBoost.

Currently, the following classification algorithms are supported:

- Naive Bayes (GaussianNB – scikit-learn)

- Logistic Regression (scikit-learn)

- K-Nearest Neighbors (KNN) (scikit-learn)

- Linear Discriminant Analysis (LDA) (scikit-learn)

- Stochastic Gradient Descent Classifier (SVM – hinge loss) (scikit-learn)

- Random Forest Classifier (scikit-learn)

- LightGBM Classifier (LightGBM)

- CatBoost Classifier (CatBoost)

This setup enables users to quickly compare different algorithms on their custom datasets without writing additional code. In addition, users can perform basic preprocessing operations such as outlier detection, handling missing values, and feature selection, as well as inspect variable distributions. Depending on the target variable, they can carry out either classification or regression analysis. All plots and visualizations are generated using Plotly.
   
    
## Steps

- **🗂️ Data Loading**: Upload your own dataset (CSV or Excel) or explore preloaded toy datasets.  
- **📊 Data Exploration**: Visualize distributions, correlations, and inspect data quality through interactive charts.  
- **🎯 Feature Selection**: Choose your target variable and select features manually.  
- **🛠️ Preprocessing Tools**: Handle missing data, remove outliers, assess normality, and manage high-cardinality features.  
- **🧠 Model Training**: Select from popular classification or regression algorithms and configure train/test splits.  
- **📈 Performance Benchmarking**: Evaluate and compare models using metrics like accuracy, RMSE, F1-score, and more.  
- **📋 Logging & Transparency**: Track decisions and changes with real-time logging for better reproducibility.



  
