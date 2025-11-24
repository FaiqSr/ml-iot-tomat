import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib


def find_target_column(df):
    candidates = ['target', 'label', 'class', 'y']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column
    return df.columns[-1]


def load_data(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(base, '..', 'dataset_tomat.csv'))
    df = pd.read_csv(path)
    return df


def build_and_train(df, target_col=None, random_state=42):
    if target_col is None:
        target_col = find_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # Decide whether to do classification or regression
    is_regression = False
    try:
        # if float dtype or many unique values, use regression
        if pd.api.types.is_float_dtype(y) or y.nunique() > 20:
            is_regression = True
    except Exception:
        is_regression = False

    if is_regression:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    results = {}
    if is_regression:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results['rmse'] = float(rmse)
        results['r2'] = float(r2)
    else:
        acc = accuracy_score(y_test, y_pred)
        results['accuracy'] = float(acc)
        results['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

    return pipeline, results


def save_model(pipeline, out_path=None):
    if out_path is None:
        base = os.path.dirname(__file__)
        model_dir = os.path.normpath(os.path.join(base, '..', 'models'))
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, 'rf_model.joblib')
    joblib.dump(pipeline, out_path)
    return out_path


def main(csv_path=None):
    print('Loading data...')
    df = load_data(csv_path)
    print(f'Data shape: {df.shape}')
    pipeline, results = build_and_train(df)
    model_path = save_model(pipeline)
    print('Model saved to:', model_path)
    print('Results:')
    for k, v in results.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Random Forest on dataset_tomat.csv')
    parser.add_argument('--csv', help='Path to CSV file (default: ../dataset_tomat.csv relative to script)')
    args = parser.parse_args()
    main(args.csv)
