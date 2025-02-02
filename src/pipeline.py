from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce


def create_preprocessing_pipeline(numeric_normal_features = [], numeric_not_normal_features = [], categorical_onehot = [], categorical_binary = []):
    transformers = []

    # Pipeline for normally distributed numerical features
    if numeric_normal_features:
        transformers.append(('num_norm', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler())  
        ]), numeric_normal_features))

    # Pipeline for non-normally distributed numerical features
    if numeric_not_normal_features:
        transformers.append(('num_not_norm', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  
            ('scaler', RobustScaler())  
        ]), numeric_not_normal_features))

    # Pipeline for categorical features using One-Hot Encoding
    if categorical_onehot:
        transformers.append(('cat_onehot', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  
        ]), categorical_onehot))

    # Pipeline for categorical features using Base-N Encoding (avoid error with empty lists)
    if categorical_binary:
        transformers.append(('cat_baseN', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', ce.BinaryEncoder(cols=categorical_binary)) 
        ]), categorical_binary))

    # Combine only the valid transformers
    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor

def create_model_pipelines(models_dict, numeric_normal, numeric_not_normal, cat_onehot, cat_binary):
    """
    Create a dictionary of pipelines where each model is associated with a preprocessing pipeline.
    """
    pipelines = {}
    
    for model_name, model in models_dict.items():
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline(numeric_normal, numeric_not_normal, cat_onehot, cat_binary)),
            ('model', model)
        ])
        pipelines[model_name] = pipeline
    
    return pipelines
