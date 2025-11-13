from flask import Flask, jsonify,request
import joblib
import numpy as np

app=Flask(__name__)
model,acc=joblib.load('linear_model.pkl')

@app.route('/predict',methods=['POST'])
def predict():
  data=request.get_json(force=True)
  features=np.array([data['Glucose'],data['BloodPressure'],data['Insulin'],data['BMI']data['DiabetesPedigreeFunction'],data['Age'],]).reshape(1,-1)
  prediction=model.predict(features)[0]
  return jsonify({'Predicted species': prediction,'Model Accuracy':round(acc,2)})
if __name__=='__main__':
  app.run(host='0.0.0.0',port=5000)