import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def main(dataset_path, model_path, n_estimators, random_state, test_size):
    df = pd.read_csv(dataset_path)
    # Expecting target column named 'Tinggi_Tanaman'
    if 'Tinggi_Tanaman' not in df.columns:
        raise SystemExit("Target column 'Tinggi_Tanaman' not found in dataset")

    X = df.drop(columns=['Tinggi_Tanaman'])
    y = df['Tinggi_Tanaman']

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train RandomForest
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    # ensure predictions are 1d numpy array
    preds = np.asarray(preds).ravel()
    # some sklearn versions don't accept the `squared` kwarg in wrapped signature;
    # compute RMSE as sqrt of MSE to avoid compatibility issues
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # Ensure model dir exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'features': list(X.columns)}, model_path)
    print(f"Saved model to: {model_path}")

    # Print a few sample predictions
    sample_X = X_test.head(5)
    sample_y = y_test.head(5)
    sample_pred = model.predict(sample_X)
    print("Sample predictions (first 5):")
    for i, (inp, gt, p) in enumerate(zip(sample_X.values, sample_y.values, sample_pred)):
        print(f"{i+1}: GT={gt:.3f} | PRED={p:.3f} | INPUT={inp}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RandomForest on tomato dataset')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.getcwd(), 'dataset_tomat.csv'),
                        help='Path to CSV dataset (default: dataset_tomat.csv in repo root)')
    parser.add_argument('--model', type=str, default=os.path.join(os.getcwd(), 'models', 'rf_model.pkl'),
                        help='Output path for saved model')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    args = parser.parse_args()

    main(args.dataset, args.model, args.n_estimators, args.random_state, args.test_size)
