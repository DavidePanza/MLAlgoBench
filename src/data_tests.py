from scipy.stats import shapiro

def run_shapiro_test(df, numeric_features):
    """
    Run Shapiro-Wilk test for normality.
    """
    normal, not_normal = [], [] 
    for col in numeric_features:
        _, p = shapiro(df[col])
        print(f'{col}: {p}')
        if p < 0.05:
            not_normal.append(col)
        else:
            normal.append(col)
    
    return normal, not_normal

def check_cardinality(df, categorical_features, treshold=10):
    """
    Check cardinality of categorical features.
    """
    high_cardinality = [] 
    low_cardinality = []
    for col in categorical_features:
        if df[col].nunique() > treshold:
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
    
    return high_cardinality, low_cardinality