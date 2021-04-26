from flask import Flask, request, jsonify,render_template
import dtt_test
import diabetes_db
app = Flask(__name__)


@app.route('/get_predict_diabetes', methods=['GET','POST'])
def get_predict_diabetes():
    if request.method == 'POST':
        Preg = int(request.form['Pregnancies'])
        gluco = int(request.form['Glucose'])
        BP = int(request.form['BloodPressure'])
        ST = int(request.form['SkinThickness'])
        Ins = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetespred = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        prediction = dtt_test.decision_tree().predict_diabeties(Preg,gluco,BP,ST,Ins,bmi,diabetespred,age)
        diabetes_db.save_diabetes_details(Preg,gluco,BP,ST,Ins,bmi,diabetespred,age,prediction)
        return "The Predicted diabetes is {} ".format(prediction) 


if __name__ == "__main__":
    print("Starting Python Flask Server For Diabetes Prediction...")
    app.run(debug = False)