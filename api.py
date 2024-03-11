import csv
from flask import Flask, request, jsonify
from sklearn.impute import SimpleImputer
import pandas as pd
from scipy.stats import mode
import joblib

app = Flask(__name__)

#reading dataset
df= pd.read_csv("dataset.csv")

X=df.iloc[:,:-1].values

imputer = SimpleImputer(strategy='mean')

X_imputed = imputer.fit_transform(X)


disease_map = {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
               'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
               'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11,
               'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14,
               'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
               'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
               'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27,
               'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29, 'Varicose veins': 30,
               'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34,
               'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
               'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}


df.replace({'prognosis': disease_map}, inplace=True)

#gives description of predicted disease
def disease_desc(query):
    results = []
    with open('disease_description.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if query.lower() in row['Disease'].lower():
                results.append(row)
    return results

#gives what precaution to take for predicted disease
def disease_prec(query):
    results = []
    with open('disease_precaution.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if query.lower() in row['Disease'].lower():
                results.append(row)
    return results

# gives which doctor to consult for predicted disease
def disease_doc(query):
    results = []
    with open('Doctor_Versus_Disease.csv', 'r', encoding='ANSI') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if query.lower() in row['Drug Reaction'].lower():
                results.append(row)
    return results


#all symptoms
symptoms = ['itching',
    'skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
    'vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain',
    'diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes',
    'malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain',
    'weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness',
    'stiff_neck','swelling_joints', 'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell',
    'bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression',
    'irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting',
    'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

#making flaskk api
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    #load all models
    dt_classifier = joblib.load('DT_model.pkl')
    nb_classifier = joblib.load('NB_model.pkl')
    rf_classifier = joblib.load('RF_model.pkl')
    knn_classifier = joblib.load('KNN_model.pkl')
    sfm = joblib.load('feature_selection_model.pkl')
    
    # Get symptoms from request
    symptoms = request.json.get('symptoms', [])
    if len(symptoms) < 4 or len(symptoms) > 6:
        return jsonify({'error': 'Please provide between 4 to 6 symptoms.'}), 400

    # Prepare user input for prediction
    user_input = {symptom: 1 for symptom in symptoms}
    user_df = pd.DataFrame([user_input], columns=df.columns[:-1])
    user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=df.columns[:-1])
    user_selected = sfm.transform(user_df_imputed)

    # Predict disease using the trained model
    dt_prediction = dt_classifier.predict(user_selected)
    nb_prediction = nb_classifier.predict(user_selected)
    rf_prediction = rf_classifier.predict(user_selected)
    knn_prediction = knn_classifier.predict(user_selected)
    ensemble_prediction = mode([dt_prediction, nb_prediction, rf_prediction, knn_prediction])[0][0]
    predicted_disease = [k for k, v in disease_map.items() if v == ensemble_prediction][0]

    # Get additional information about the predicted disease
    description = disease_desc(predicted_disease)
    precautions = disease_prec(predicted_disease)
    doctor_recommendations = disease_doc(predicted_disease)
 
    #return the desired results back top application
    return jsonify({
        'disease': predicted_disease,
        'description': description,
        'precaution': precautions,
        'doctor': doctor_recommendations
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    