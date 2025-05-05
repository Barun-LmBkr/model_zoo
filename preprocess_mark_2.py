import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_data(X, y):
    # Step 1: Automatically detect date columns
    date_columns = X.select_dtypes(include=['object']).columns[
        X.select_dtypes(include=['object']).apply(pd.to_datetime, errors='coerce').notna().any()].tolist()
    
    # Step 2: Handle date columns (Convert to numeric features like year, month, etc.)
    for col in date_columns:
        X[col] = pd.to_datetime(X[col], errors='coerce')  # Convert to datetime
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek
        X.drop(col, axis=1, inplace=True)  # Drop the original date column

    # Step 3: Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Step 4: Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handling missing values in numeric columns
        ('scaler', StandardScaler())  # Standardizing numeric columns
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handling missing categorical values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoding for categorical columns
    ])

    # Step 5: Combine transformations into a single column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Step 6: Apply the transformations
    X_transformed = preprocessor.fit_transform(X)

    # Step 7: Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
