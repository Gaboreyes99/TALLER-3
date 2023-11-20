from joblib import load
import numpy as np

variables = ["customerID",
             "gender",
             "SeniorCitizen",
             "Partner",
             "Dependents",
             "tenure",
             "PhoneService",
             "MultipleLines",
             "InternetService",
             "OnlineSecurity",
             "OnlineBackup",
             "DeviceProtection",
             "TechSupport",
             "StreamingTV",
             "StreamingMovies",
             "Contract",
             "PaperlessBilling",
             "PaymentMethod",
             "MonthlyCharges",
             "TotalCharges"]

class PredictionModeloBaseline:

    def __init__(self):
        self.model = load("pipeline_punto2.joblib")

    def make_predictions(self, data):
        result_label = self.model.predict(data)
        array_label = np.array(result_label)
        result_proba= self.model.predict_proba(data)
        proba_array = np.array(result_proba)
        proba_final = []
        for i in range(0, len(proba_array)):
            if proba_array[i][0] >= proba_array[i][1]:
                proba_final.append(proba_array[i][0])
            else:
                proba_final.append(proba_array[i][1])
        result = []
        for i in range(0, len(array_label)):
            result.append([array_label[i], proba_final[i]])
        return result
    
    def get_explanation(self):
        modelo = self.model.steps[-1][1]
        pesos = modelo.feature_importances_
        features = variables
        mapa_pesos = dict(zip(features, pesos))
        result = sorted(mapa_pesos.items(), key=lambda x: x[1], reverse=True)[:3]
        return result
    
    def make_predictions_ab(self, data):
        result_label = self.model.predict(data)
        array_label = np.array(result_label)
        result = []
        result = ([array_label[0], 'Baseline Model'])
        return result
    
class PredictionModeloFinal:

    def __init__(self):
        self.model = load("pipeline_punto3.joblib")

    def make_predictions(self, data):
        result_label = self.model.predict(data)
        array_label = np.array(result_label)
        result_proba= self.model.predict_proba(data)
        proba_array = np.array(result_proba)
        proba_final = []
        for i in range(0, len(proba_array)):
            if proba_array[i][0] >= proba_array[i][1]:
                proba_final.append(proba_array[i][0])
            else:
                proba_final.append(proba_array[i][1])
        result = []
        for i in range(0, len(array_label)):
            result.append([array_label[i], proba_final[i]])
        return result
    
    def get_explanation(self):
        modelo = self.model.steps[-1][1]
        pesos = modelo.feature_importances_
        features = variables
        mapa_pesos = dict(zip(features, pesos))
        result = sorted(mapa_pesos.items(), key=lambda x: x[1], reverse=True)[:3]
        return result
    
    def make_predictions_ab(self, data):
        result_label = self.model.predict(data)
        array_label = np.array(result_label)
        result = []
        result = ([array_label[0], 'Final Model'])
        return result