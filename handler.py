import pickle
# import os

import pandas as pd

from flask import Flask, request

from wine_quality.wine_quality import WineQuality

model = pickle.load(open('./model/model_wine_quality.pkl', 'rb'))

app = Flask(__name__)


@app.post('/predict')
def predict():
    test_json = request.get_json()

    if isinstance(test_json, dict):
        df_raw = pd.DataFrame(test_json, index=[0])
    else:
        df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    pipeline = WineQuality()

    df1 = pipeline.data_preparation(df_raw)

    pred = model.predict(df1)

    df1['predict'] = pred

    return df1.to_json(orient='records')


# if __name__ == '__main__':
#     PORT = os.environ('PORT', 5000)
#     app.run(host='0.0.0.0', port=PORT)
