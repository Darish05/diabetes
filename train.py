import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import joblib

data=pd.read_csv('diabetes.csv')
x=data[['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=data['outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print("Accuracy:",acc)

joblib.dump((model,acc),'linear_model.pkl')
print("model and accuracy saved in linear_model.pkl")
