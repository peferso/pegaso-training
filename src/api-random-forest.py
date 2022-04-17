from flask import Flask
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

rf = joblib.load("models/rf_2022-02-12_22-41-03312819.joblib")

@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/predict_rf/<brand>/<int:kilometers>/<int:power>/<int:doors>/<int:age>')
def predict_rf(brand, kilometers, power, doors, age):

   params = {
      "brand": brand,
      "kilometers": kilometers,
      "power": power,
      "doors": doors,
      "age": age
   }

   price = random_forest(params)

   return str(price[0])


def random_forest(params):
   brd = params["brand"]
   kms = params["kilometers"]
   pwr = params["power"]
   dor = params["doors"]
   yyr = params["age"]

   features_labels = ['kilometers', 'power', 'doors', 'age', 'brand_AUDI', 'brand_BMW', 'brand_CITROEN', 'brand_FIAT',
                      'brand_FORD',
                      'brand_HYUNDAI', 'brand_JAGUAR', 'brand_KIA', 'brand_LANDROVER', 'brand_MAZDA', 'brand_MERCEDES',
                      'brand_MINI',
                      'brand_NISSAN', 'brand_OPEL', 'brand_PEUGEOT', 'brand_PORSCHE', 'brand_RENAULT', 'brand_SEAT',
                      'brand_SKODA',
                      'brand_SMART', 'brand_TOYOTA', 'brand_VOLKSWAGEN', 'brand_VOLVO']

   values = []

   for col in features_labels:
      if col.split('_')[0] == 'brand' and col.split('_')[1] == params['brand']:
         values.append(1)
      elif col.split('_')[0] == 'brand' and col.split('_')[1] != params['brand']:
         values.append(0)
      else:
         values.append(params[col])

   X = np.array(values)
   X = X.reshape(1, -1)

   price = rf.predict(X)

   return price

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8081, debug=True)