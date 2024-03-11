import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import joblib



#all the diseases
disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']


#reading dataset
df= pd.read_csv("dataset.csv")
#df1=pd.read_csv("testing.csv")
#df.drop(columns=df.columns[-1], inplace=True)
imputer = SimpleImputer(strategy='mean')


#giving values to diseases
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

#df1.replace({'prognosis': disease_map}, inplace=True)

print(df.head())
#print(df1.head())

#plots
def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = math.ceil((nCol + nGraphPerRow - 1) / nGraphPerRow)
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    

def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df1 = df1.dropna()
    df1 = df1[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
    
#plotPerColumnDistribution(df, 10, 5)

#plotScatterMatrix(df, 20, 10)

#dividing into features and results
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y=np.ravel(Y)

X_imputed = imputer.fit_transform(X)

"""X_test= df1.iloc[:,:-1].values
Y_test=df1.iloc[:,-1].values
Y1=np.ravel(Y_test)"""



# all teh used classification models
dt_classifier= DecisionTreeClassifier(random_state=0,criterion='entropy')
nb_classifier = GaussianNB()
rf_classifier = RandomForestClassifier(criterion='entropy',random_state=0,n_estimators=50)
knn_classifier = KNeighborsClassifier()

#feature selection
rf_classifier.fit(X, Y)
sfm = SelectFromModel(rf_classifier, prefit=True)
sfm.fit(X, df.columns[:-1])
X_selected = sfm.transform(X)
print("Selected features shape:", X_selected.shape)
joblib.dump(sfm, 'feature_selection_model.pkl')

#dividing into training and test sets
X_train,X_test,Y_train,Y_test= train_test_split(X_selected,Y,test_size=0.25,random_state=0)

#fitting all clasiifiers
dt_classifier.fit(X_train,Y_train)
nb_classifier.fit(X_train, Y_train)
rf_classifier.fit(X_train, Y_train)
knn_classifier.fit(X_train, Y_train)

#saving all ml models
joblib.dump(dt_classifier, 'DT_model.pkl')
joblib.dump(nb_classifier, 'NB_model.pkl')
joblib.dump(rf_classifier, 'RF_model.pkl')
joblib.dump(knn_classifier, 'KNN_model.pkl')

#predicting using training set
Y_pred_dt = dt_classifier.predict(X_test)
Y_pred_nb = nb_classifier.predict(X_test)
Y_pred_rf = rf_classifier.predict(X_test)
Y_pred_knn = knn_classifier.predict(X_test)

#taking mode of all models to get best results
Y_pred_mode = mode([Y_pred_dt, Y_pred_nb, Y_pred_rf, Y_pred_knn])[0]

#printing accuracy an other metrics for each model
print("accuracy score DT")
print(accuracy_score(Y_test,Y_pred_dt))
print(accuracy_score(Y_test, Y_pred_dt,normalize=False))
print("Confusion matrix DT")
print(confusion_matrix(Y_test,Y_pred_dt))
print(classification_report(Y_test, Y_pred_dt, target_names=disease))

print("accuracy score NB")
print(accuracy_score(Y_test,Y_pred_nb))
print(accuracy_score(Y_test, Y_pred_nb,normalize=False))
print("Confusion matrix NB")
print(confusion_matrix(Y_test,Y_pred_nb))
print(classification_report(Y_test, Y_pred_nb, target_names=disease))

print("accuracy score RF")
print(accuracy_score(Y_test,Y_pred_rf))
print(accuracy_score(Y_test, Y_pred_rf,normalize=False))
print("Confusion matrix RF")
print(confusion_matrix(Y_test,Y_pred_rf))
print(classification_report(Y_test, Y_pred_rf, target_names=disease))

print("accuracy score KNN")
print(accuracy_score(Y_test,Y_pred_knn))
print(accuracy_score(Y_test, Y_pred_knn,normalize=False))
print("Confusion matrix KNN")
print(confusion_matrix(Y_test,Y_pred_knn))
print(classification_report(Y_test, Y_pred_knn, target_names=disease))

#using k fold cross validation to avoid overfitting and get best results
accuracies=cross_val_score(estimator=dt_classifier,X=X_train,y=Y_train,cv=10)
print("Accuracy DT: {:.2f}%".format(accuracies.mean()*100))
print("Standard deviation: {:.2f}%".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=nb_classifier,X=X_train,y=Y_train,cv=10)
print("Accuracy NB: {:.2f}%".format(accuracies.mean()*100))
print("Standard deviation: {:.2f}%".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=rf_classifier,X=X_train,y=Y_train,cv=10)
print("Accuracy RF: {:.2f}%".format(accuracies.mean()*100))
print("Standard deviation: {:.2f}%".format(accuracies.std()*100))

accuracies=cross_val_score(estimator=knn_classifier,X=X_train,y=Y_train,cv=10)
print("Accuracy KNN: {:.2f}%".format(accuracies.mean()*100))
print("Standard deviation: {:.2f}%".format(accuracies.std()*100))

#printing mode accuracy or overall accuracy
print("accuracy of mode")
accuracy = accuracy_score(Y_test, Y_pred_mode)
print("Accuracy:", accuracy)








    
    
"""user_input = {}

# Get user input for each symptom
print("Enter presence of 6 symptoms:")
for i in range(6):
    while True:  # Loop until valid input is provided
        symptom = input(f"Symptom {i+1}: ").strip().lower()  # Normalize the input
        if symptom in symptoms:  # Check if the symptom is in the symptoms array
            user_input[symptom] = 1
            break  # Exit the loop if valid input is provided
        else:
            print("Invalid symptom! Please enter a valid symptom.")
# Convert user input to DataFrame
user_df = pd.DataFrame([user_input], columns=df.columns[:-1])
user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=df.columns[:-1])
# Perform feature selection on user input using SelectFromModel
user_selected = sfm.transform(user_df_imputed)

# Predict disease using all classifiers
dt_prediction = dt_classifier.predict(user_selected)
nb_prediction = nb_classifier.predict(user_selected)
rf_prediction = rf_classifier.predict(user_selected)
knn_prediction = knn_classifier.predict(user_selected)

# Compute the mode of predictions
ensemble_prediction = mode([dt_prediction, nb_prediction, rf_prediction, knn_prediction])[0][0]

# Map disease index back to disease name
predicted_disease = [k for k, v in disease_map.items() if v == ensemble_prediction][0]

# Display the predicted disease
print("Predicted Disease:", predicted_disease)"""