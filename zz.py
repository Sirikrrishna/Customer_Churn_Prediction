import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tansformation_config=DataTransformationConfig()

    def get_data_tansformer_object(self):
        '''
        This function is responsible for data transformation.
        
        '''
        try:
            numerical_coloumns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_coloumns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

            num_pipeline= Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy='median')),
                   ("scaler", StandardScaler())

               ] 
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=True)),
                ("scaler",StandardScaler(with_mean=False))
                ]
            
            
            )

            logging.info(f"Numerical columns: {numerical_coloumns} standard scaling completed")

            logging.info(f"Categorical columns: {categorical_coloumns} encoding completed")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_coloumns),
                ("cat_pipeline",cat_pipeline,categorical_coloumns)
                
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_tansformer_object()

            target_coloumn = "Churn"
            numerical_coloumns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

        # Convert non-numeric values to NaN for numerical columns
            train_df[numerical_coloumns] = train_df[numerical_coloumns].apply(pd.to_numeric, errors='coerce')
            test_df[numerical_coloumns] = test_df[numerical_coloumns].apply(pd.to_numeric, errors='coerce')
            
            input_feature_train_df =train_df.drop(columns=[target_coloumn], axis =1)
            target_feature_train_df=train_df[target_coloumn]

            input_feature_test_df=test_df.drop(columns=[target_coloumn], axis=1)
            target_feature_test_df=test_df[target_coloumn]

            logging.info(
                f"Applying pre processing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_tansformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                test_arr,
                self.data_tansformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        

        import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import accuracy_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
            