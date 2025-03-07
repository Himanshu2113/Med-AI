from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# database
symptoms = pd.read_csv('datasets/symtoms_df.csv')
precautions = pd.read_csv('datasets/precautions_df.csv')
workout = pd.read_csv('datasets/workout_df.csv')
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv('datasets/diets.csv')
description = pd.read_csv('datasets/description.csv')

# model
with open("models\content\saved_models.pkl", "rb") as file:
    loaded_models = pickle.load(file)
rfc = loaded_models["RandomForestClassifier"]


input_cols = ["Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills",
                      "Joint Pain", "Stomach Pain", "Acidity", "Ulcers on Tongue", "Muscle Wasting", "Vomiting",
                      "Burning Micturition", "Fatigue",
                      "Weight Gain", "Anxiety", "Cold Hands and Feets", "Mood Swings", "Weight Loss", "Restlessness",
                      "Lethargy", "Patches in Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes",
                      "Breathlessness",
                      "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea",
                      "Back Pain", "Constipation", "Abdominal Pain", "Diarrhoea", "Mild Fever", "Yellow Urine",
                      "Yellowing of Eyes",
                      "Chest Pain", "Dizziness", "Muscle Pain", "Red Spots Over Body", "Belly Pain",
                      "Abnormal Menstruation"]

all_symptoms = ["Itching", "Skin Rash", "Nodal Skin Eruptions", "Continuous Sneezing", "Shivering", "Chills","Joint Pain", "Stomach Pain", "Acidity", "Ulcers on Tongue", "Muscle Wasting", "Vomiting",
                    "Burning Micturition", "Fatigue",
                    "Weight Gain", "Anxiety", "Cold Hands and Feets", "Mood Swings", "Weight Loss", "Restlessness",
                    "Lethargy", "Patches in Throat", "Irregular Sugar Level", "Cough", "High Fever", "Sunken Eyes",
                    "Breathlessness",
                    "Sweating", "Dehydration", "Indigestion", "Headache", "Yellowish Skin", "Dark Urine", "Nausea",
                    "Back Pain", "Constipation", "Abdominal Pain", "Diarrhoea", "Mild Fever", "Yellow Urine",
                    "Yellowing of Eyes",
                    "Chest Pain", "Dizziness", "Muscle Pain", "Red Spots Over Body", "Belly Pain",
                    "Abnormal Menstruation"]


label_mapping = {'(vertigo) Paroymsal  Positional Vertigo': 15,
                         'AIDS': 18,
                         'Acne': 21,
                         'Alcoholic hepatitis': 33,
                         'Allergy': 9,
                         'Arthritis': 23,
                         'Bronchial Asthma': 27,
                         'Cervical spondylosis': 4,
                         'Chicken pox': 26,
                         'Chronic cholestasis': 39,
                         'Common Cold': 29,
                         'Dengue': 16,
                         'Diabetes ': 14,
                         'Dimorphic hemmorhoids(piles)': 8,
                         'Drug Reaction': 36,
                         'Fungal infection': 28,
                         'GERD': 4,
                         'Gastroenteritis': 25,
                         'Heart attack': 32,
                         'Hepatitis B': 28,
                         'Hepatitis C': 23,
                         'Hepatitis D': 27,
                         'Hepatitis E': 19,
                         'Hypertension ': 4,
                         'Hyperthyroidism': 26,
                         'Hypoglycemia': 13,
                         'Hypothyroidism': 20,
                         'Impetigo': 26,
                         'Jaundice': 34,
                         'Malaria': 19,
                         'Migraine': 40,
                         'Osteoarthristis': 7,
                         'Paralysis (brain hemorrhage)': 20,
                         'Peptic ulcer diseae': 10,
                         'Pneumonia': 35,
                         'Psoriasis': 15,
                         'Tuberculosis': 29,
                         'Typhoid': 32,
                         'Urinary tract infection': 17,
                         'Varicose veins': 22,
                         'hepatitis A': 18}

app = Flask(__name__)


def get(disease):
    desc = description[description['Disease'] == disease]['Description']

    precautions_list = precautions[precautions['Disease'] == disease][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

    workout_list = workout[workout['disease'] == disease]['workout']

    medications_list = medications[medications['Disease'] == disease]['Medication']

    diets_list = diets[diets['Disease'] == disease]['Diet']

    return desc, precautions_list, workout_list, medications_list, diets_list


# create routes
@app.route('/', methods=['POST', 'GET'])
def index():
    selected_symptoms = request.form.get("symptomss", "").split(",") if request.method == "POST" else []

    predicted_disease = None


    return render_template("index.html", all_symptoms=all_symptoms, selected_symptoms=selected_symptoms, predicted_disease=predicted_disease)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptomss')
        input_arr = np.zeros((1, 132))

        # Assign 1 to corresponding symptoms
        for i, col in enumerate(input_cols):
            if col in symptoms:
                input_arr[0, i] = 1

        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        # Predict disease
        encoded_value = rfc.predict(input_arr)[0]  # Extracting single value
        disease = reverse_label_mapping.get(encoded_value, "Unknown Disease")

        # Handle case when disease is unknown
        if disease == "Unknown Disease":
            desc, precautions, workout, medications, diets = "N/A", [], [], [], []
        else:
            desc, precautions, workout, medications, diets = get(disease)

        if desc.empty:
            desc = "No description available"
        else:
            desc = str(desc.iloc[0]) if isinstance(desc, pd.Series) else str(desc)

        precautions = ', '.join(precautions.iloc[0].dropna()) if isinstance(precautions, pd.DataFrame) else str(
            precautions)

        if isinstance(workout, pd.Series) and not workout.empty:
            workout = ', '.join(workout.dropna().tolist())  # Remove NaN values and join
        elif isinstance(workout, list):
            workout = ', '.join(workout)  # Convert list to a string
        else:
            workout = "No workout recommendations available"  # Handle empty case

        if isinstance(medications, pd.Series) and not medications.empty:
            medications = medications.iloc[0]  # Extract first value if it's a Series
        elif isinstance(medications, list):
            medications = ', '.join(medications)  # Join list elements into a string
        else:
            medications = "No medication available"  # Handle empty case

        if isinstance(diets, pd.Series) and not diets.empty:
            diets = diets.iloc[0]  # Extract first value if it's a Series
        elif isinstance(diets, list):
            diets = ', '.join(diets)  # Convert list to a readable string
        else:
            diets = "No diet recommendations available"  # Handle empty case

        return render_template('index.html', predicted_disease=disease, desc=desc,
                               precautions=precautions, workout=workout,
                               medications=medications, diets=diets)


if __name__ == "__main__":
    app.run(debug=True)
