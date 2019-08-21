from flask import Flask, jsonify, request, session, json  
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle        

app = Flask(__name__)             # create an app instance

# predict function
def predict_function(data, model):
    model = f'models/{model}.pkl'
    loaded_model = pickle.load(open(model, 'rb'))
    predicted = loaded_model.predict(data)
    predicted = predicted.tolist()
    return predicted

# handle predict
@app.route('/model/api/v1.0/predict', methods=['POST'])                  # at the end point /
def model_predict():   
    if request.json:
        json_rec = request.json['data']
        model = json_rec['model']
        data = json_rec['data']
        result = predict_function(data, model)
        req_json = request.json
        request.json['data']['prediction'] = result
        return jsonify(req_json)
    else:
        return "Must post json"                  
            
if __name__ == "__main__":        
    app.run(debug=True)                    