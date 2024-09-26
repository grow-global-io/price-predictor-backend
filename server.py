from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

sheet_id = '1VUOiDUrvtUge8StvOoPecLe0aV4vU_3gYTQH8OAdphw'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'
df = pd.read_csv(url)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

@app.route("/predictprice/<locality>/<noOfYears>")
def predictPrice(locality:str, noOfYears:int):
  action_area_1 = df[df["Location"] == locality].iloc[0, 2:].astype(float)
  action_area_1 = action_area_1.interpolate()
  train_data = action_area_1.dropna()
  train_data.index = train_data.index.astype(int)
  lastYear = int(train_data.index[-1])
  X_train = train_data.index.values.reshape(-1, 1)  # Years
  y_train = train_data.values  # Prices
  model = LinearRegression()
  model.fit(X_train, y_train)
  X_predict = np.array([(lastYear + int(i)) for i in range(1, int(noOfYears) + 1)]).reshape(-1, 1)
  predictions = model.predict(X_predict)
  # Create a new Series for the predictions
  rangerstr = [str(lastYear + int(i)) for i in range(1, int(noOfYears) + 1)] 
  predictions_series = pd.Series(predictions.flatten(), index=rangerstr)
  # Combine the original data with predictions
  full_data = pd.concat([action_area_1, predictions_series])
  full_data_dict = full_data.to_dict()  # Convert to dictionary for JSON serialization
  return jsonify(full_data_dict), 200

@app.route("/")
def home():
    return "This API is Written in Python and is working just fine. "

@app.route("/get-user/<user_id>")
def get_user(user_id):
    user_data = {
        "user_id": user_id,
        "name": "John Doe",
        "email": "job2@gmail.com"
    }
    extra = request.args.get("extra")
    if extra:
        user_data["extra"]= extra

    return jsonify(user_data),200

@app.route("/create-user", methods=["POST"])
def create_user():
    # if request.method == 'POST'
    data = request.get_json()
    return jsonify(data), 201

if __name__ == "__main__":
    app.run(debug=True)