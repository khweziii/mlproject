import os 
import sys 
import numpy as np 
import pandas as pd 
import dill 
from src.exception import CustomException
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj: # write byte mode
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
       
        # X_train, X_test, y_train, y_test = train_test_split(
        #         X, y, test_size=0.2, random_state=42
        # )

        report = {}

        for i in range(len(list(models))):
            '''
            For each model, you will train it, evaluate it and store performance in 
            report dictionary
            '''
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gridSearch = GridSearchCV(model, para, cv=3) # performing hyperparameter tuning using grid search
            gridSearch.fit(X_train, y_train)

            model.set_params(**gridSearch.best_params_)
            model.fit(X_train, y_train) # Train model

            y_train_pred = model.predict(X_train) # evaluate model on train data

            y_test_pred = model.predict(X_test) # evaluate model on test data

            train_model_score = r2_score(y_train, y_train_pred) # calculate r2 on train data

            test_model_score = r2_score(y_test, y_test_pred) # calculate r2 on test data

            report[list(models.keys())[i]] = test_model_score # save model performance to report dictionary

            return report 
        

        
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)