# Check Dependencies
from lib_installer import installer
installer()

# Main general libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
# Model libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier

def data_setup(train_path, test_path):
    # Load the data
    data_train = pd.read_csv(cwd + train_path, sep=";")
    data_test = pd.read_csv(cwd + test_path,sep=";")
    used_values = ["edad", "trabajo", "saldo", "duracion", "tiempo_transcurrido", "resultado_campanas_anteriores"]

    # Convert data into Boolean
    # Train set
    unique_jobs_train = data_train['trabajo'].unique()
    unique_jobs_train.sort()
    jobs_counter_train = np.zeros(data_train['trabajo'].size)
    for i, job in enumerate(unique_jobs_train):
        jobs_counter_train += i*(data_train['trabajo'] == job)
    data_train['trabajo'] = jobs_counter_train
    data_train["target"] = data_train["target"] == 'si'
    data_train["resultado_campanas_anteriores"] = data_train["resultado_campanas_anteriores"] == 'exito'

    # Test set
    unique_jobs_test = data_test['trabajo'].unique()
    unique_jobs_test.sort()
    jobs_counter_test = np.zeros(data_test['trabajo'].size)
    for i, job in enumerate(unique_jobs_test):
        jobs_counter_test += i*(data_test['trabajo'] == job)
    data_test['trabajo'] = jobs_counter_test
    data_test["target"] = data_test["target"] == 'si'
    data_test["resultado_campanas_anteriores"] = data_test["resultado_campanas_anteriores"] == 'exito'

    Y_train = data_train["target"].values
    X_train = data_train[used_values].values

    Y_test = data_test["target"].values
    X_test = data_test[used_values].values

    return(X_train, Y_train, X_test, Y_test, used_values)

def train_model_rf(X_train_rf, Y_train, X_test_rf, Y_test):
    # Training a random forest model with the best parameters found before and only the most relevant features
    max_features, n_estimators = 3, 30
    rf_final_model = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
    rf_final_model.fit(X_train_rf, Y_train)

    random_forest_predictions = rf_final_model.predict(X_test_rf)

    random_forest_scores = accuracy_score(Y_test, random_forest_predictions), precision_score(Y_test, random_forest_predictions), recall_score(Y_test, random_forest_predictions), f1_score(Y_test, random_forest_predictions)

    scores_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig0, ax0 = plt.subplots(num=0, dpi=150)
    ax0.axis('off')
    ax0.axis('tight')
    ax0.table(cellText=[[scores_titles[i], f'{100 * score:.2f} %'] for i, score in enumerate(random_forest_scores)], loc='center', cellLoc='center', edges='horizontal')
    plt.show()

    return (rf_final_model, random_forest_scores)

def make_prediction(model, data):
    rf_prediction = model.predict([data])
    return (rf_prediction[0])

def results(pred, used_values, user_features):
    fig1, ax1 = plt.subplots(num=1, dpi=150)
    results_summary = [[v, user_features[i]] for i, v in enumerate(used_values)]
    results_summary = np.vstack((results_summary, ['Prediction', pred]))
    ax1.set_title('User Prediction')
    ax1.axis('off')
    ax1.axis('tight')
    ax1.table(cellText=results_summary, loc='center', cellLoc='center', edges='horizontal')
    plt.show()

if __name__ == "__main__":
    train_path, test_path = "/dataset/train.csv", "/dataset/test.csv"
    X_train_rf, Y_train, X_test_rf, Y_test, used_values = data_setup(train_path, test_path)

    rf_final_model, rf_scores = train_model_rf(X_train_rf, Y_train, X_test_rf, Y_test)

    # Feel free to test any information with the final model below. Consult the table at the begining for info codification
    user_features = np.array([0, # Last Call Duration
                        0, # Account Balance
                        21, # Age
                        False, # Past Campaigns' Results
                        0, # Job (consult the table at the begining for reference)
                        -1]) # Days from Last Call
    
    pred = make_prediction(rf_final_model, user_features)
    results(pred, used_values, user_features)
