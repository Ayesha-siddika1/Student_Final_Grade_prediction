from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Define the feature order
FEATURE_ORDER = [
    'Attendance_Rate',
    'Assignment_Score_Avg',
    'Quiz_Score_Avg',
    'Midterm_Score',
    'Participation_Level',
    'Study_Hours_Per_Week',
    'Sleep_Hours_Per_Night',
    'Internet_Access',
    'Extracurricular_Activities',
    'Parental_Education_Level_Bachelors',
    'Parental_Education_Level_Masters',
    'Parental_Education_Level_PhD'
]

# Load the trained model
try:
    with open('final_grade_predictor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully:", type(model))  # Debug print
except Exception as e:
    print("Error loading model:", str(e))
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            # Get values from the form
            features = {
                'Attendance_Rate': float(request.form['attendance_rate']),
                'Assignment_Score_Avg': float(request.form['assignment_score']),
                'Quiz_Score_Avg': float(request.form['quiz_score']),
                'Midterm_Score': float(request.form['midterm_score']),
                'Participation_Level': float(request.form['participation_level']),
                'Study_Hours_Per_Week': float(request.form['study_hours']),
                'Sleep_Hours_Per_Night': float(request.form['sleep_hours']),
                'Internet_Access': 1 if request.form.get('internet_access') == 'Yes' else 0,
                'Extracurricular_Activities': 1 if request.form.get('extracurricular') == 'Yes' else 0,
                'Parental_Education_Level_Bachelors': 0,
                'Parental_Education_Level_Masters': 0,
                'Parental_Education_Level_PhD': 0
            }
            
            # Set the appropriate education level to 1 (skip HighSchool as it's the reference)
            education_level = request.form['parental_education']
            if education_level in ['Bachelors', 'Masters', 'PhD']:
                features[f'Parental_Education_Level_{education_level}'] = 1
            
            # Create input array with correct feature order
            input_array = np.array([[features[feature] for feature in FEATURE_ORDER]])
            
            # Make prediction
            if isinstance(model, np.ndarray):
                # If model is coefficients array, manually calculate prediction
                prediction = np.dot(input_array, model)
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item()  # Convert single-element array to scalar
            else:
                # If model is sklearn model object
                prediction = model.predict(input_array)[0]
            
            prediction = round(float(prediction), 2)
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            print(error)
    
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
