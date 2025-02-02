from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, r2_adj

def prepare_data(df, target, test_threshold):
    
    # Features & Target
    y = df.loc[:,target]
    X = df.drop(columns=[target])

    # Split data
    test_threshold = test_threshold / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_threshold, random_state=42)

    return X_train, X_test, y_train, y_test

def train_models(pipelines, X_train, X_test, y_train, y_test, target_type):
    """
    Train each model sequentially and store results in a dictionary.
    """
    results = {}
    
    for model_name, pipeline in pipelines.items():
        print(f"Training model: {model_name}")
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        if target_type == 'object':
            # Evaluate (Accuracy as an example, use other metrics as needed)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy

        else:
            # Evaluate (R2 as an example, use other metrics as needed)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = r2
    
    return results