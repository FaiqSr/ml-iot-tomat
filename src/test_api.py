import os
import importlib.util
import pandas as pd

# load src/app.py as module by path to avoid package import issues
app_path = os.path.join(os.getcwd(), 'src', 'app.py')
spec = importlib.util.spec_from_file_location('app_module', app_path)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

app = app_module.create_app()
features = app.config['features']

# load dataset and use first row as sample
df = pd.read_csv(os.path.join(os.getcwd(), 'dataset_tomat.csv'))
sample = df[features].iloc[0].to_dict()

with app.test_client() as c:
    resp = c.post('/predict', json={'features': sample})
    print('Status code:', resp.status_code)
    print('Response JSON:', resp.get_data(as_text=True))
