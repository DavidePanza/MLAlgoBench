from scipy.stats import shapiro

def run_shapiro_test(df, numeric_features):
    normal, not_normal = [], [] 
    for col in numeric_features:
        stat, p = shapiro(df[col])
        print(f'{col}: {p}')
        if p < 0.05:
            print(f'{col} is not normally distributed')
            not_normal.append(col)
        else:
            print(f'{col} is normally distributed')
            normal.append(col)
    
    return normal, not_normal

def check_cardinality(df, categorical_features, treshold=10):
    high_cardinality = [] 
    low_cardinality = []
    for col in categorical_features:
        if df[col].nunique() > treshold:
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
    
    return high_cardinality, low_cardinality