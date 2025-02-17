import streamlit as st

def get_categorical_models():
    """
    Get a dictionary of categorical models
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    categorical_models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Stochastic Vector Machine': SGDClassifier(loss='hinge'),
        'Random Forest': RandomForestClassifier(),
        'LightGBM': LGBMClassifier(),
        'CatBoost Classification': CatBoostClassifier(),
    }

    return categorical_models

def get_regression_models():
    """
    Get a dictionary of regression models
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Machine': SVR(),
        'Random Forest Regression': RandomForestRegressor(),
        'LightGBM': LGBMRegressor(),
        'CatBoost': CatBoostRegressor(),
    }

    return regression_models

def model_selection(models):
    """
    Checkbox selection for models in a single column using list comprehension.
    """
    selected_models = st.multiselect("Select models", options=models, default=models)
    return selected_models