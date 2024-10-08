import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "KNN Classifier": KNeighborsClassifier(),
                "XGB Classifier": XGBClassifier(),
                "Catboost Classifier": CatBoostClassifier(),
                "SVM": svm.SVC()
            }
            params={
                "Random Forest": {
                        #'n_estimators': [100, 200, 300],
                        #'max_depth': [10, 20, 30, None],
                        #'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'bootstrap': [True, False]
                },
                "KNN Classifier":{
                    #'n_neighbors': [3, 5, 7, 9],
                    #'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "XGB Classifier":{
                    #'learning_rate': [0.01, 0.1, 0.2],
                    #'n_estimators': [100, 200, 300],
                    #'max_depth': [3, 5, 7],
                    'colsample_bytree': [0.3, 0.7]
                },
                "Catboost Classifier":{
                    #'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    #'iterations': [100, 200, 500]
                },
                "SVM":{
                    #'C': [0.1, 1, 10, 100],
                    #'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ### This will give you best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            ### This will give you best model score from the dictionary

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both train and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy


        except Exception as e:
            raise CustomException(e,sys)
