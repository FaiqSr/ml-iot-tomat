# Random Forest Tomato Model

This repository contains a simple Random Forest regression pipeline that predicts `Tinggi_Tanaman` (plant height) from environmental features in `dataset_tomat.csv`.

Quick start:

- Train the model:

```powershell
& .\env\Scripts\python.exe src\train_random_forest.py
```

- Run prediction (uses last row by default):

```powershell
& .\env\Scripts\python.exe src\predict.py
```

Model is saved to `models/rf_model.pkl` by default.
"# ml-iot-tomat" 
