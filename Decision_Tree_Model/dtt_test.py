import pandas as pandas
import numpy as np 
import pickle 

file_path1 = 'Model/decision_model.pickle'
file_path2 = 'Model/test.pickle'

model = pickle.load(open(file_path1,'rb'))
test_model = pickle.load(open(file_path2,'rb'))

class decision_tree():
    def predict_diabeties(self,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        z = np.zeros(len(test_model.columns))
        z[0] = Pregnancies
        z[1] = Glucose
        z[2] = BloodPressure
        z[3] = SkinThickness
        z[4] = Insulin
        z[5] = BMI
        z[6] = DiabetesPedigreeFunction
        z[7] = Age
        return model.predict([z])[0]

if __name__ == "__main__":
    dt = decision_tree()
    diabeties = dt.predict_diabeties(4,62,83,22,0,32.4,1.242,47)
    print('The Prediction Is :',diabeties)