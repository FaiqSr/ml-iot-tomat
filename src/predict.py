import os
import argparse
import pandas as pd
import joblib


def main(model_path, input_csv=None, row_index=None):
    data_path = input_csv or os.path.join(os.getcwd(), 'dataset_tomat.csv')
    df = pd.read_csv(data_path)

    m = joblib.load(model_path)
    model = m['model']
    features = m.get('features')

    if features is None:
        raise SystemExit('Saved model does not contain feature list')

    X = df[features]

    if row_index is not None:
        sample = X.iloc[[row_index]]
    else:
        sample = X.tail(1)

    pred = model.predict(sample)
    print(f"Input (features):\n{sample}")
    print(f"Predicted Tinggi_Tanaman: {pred[0]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load RF model and predict')
    parser.add_argument('--model', type=str, default=os.path.join(os.getcwd(), 'models', 'rf_model.pkl'))
    parser.add_argument('--csv', type=str, default=None, help='Optional CSV file to predict on')
    parser.add_argument('--row', type=int, default=None, help='Row index to predict (0-based). Defaults to last row')
    args = parser.parse_args()

    main(args.model, args.csv, args.row)
