import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from imblearn.combine import SMOTEENN  # Import SMOTEENN

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                                   'PaperlessBilling', 'PaymentMethod']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("power_transformer", PowerTransformer()),  # Add PowerTransformer
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse=False, handle_unknown= 'ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns} standard scaling completed.")
            logging.info(f"Categorical columns: {categorical_columns} encoding completed.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Churn"
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

            # Convert non-numeric values to NaN for numerical columns
            train_df[numerical_columns] = train_df[numerical_columns].apply(pd.to_numeric, errors='coerce')
            test_df[numerical_columns] = test_df[numerical_columns].apply(pd.to_numeric, errors='coerce')

            # Encode target column if necessary
            train_df[target_column] = train_df[target_column].replace({'Yes': 1, 'No': 0})
            test_df[target_column] = test_df[target_column].replace({'Yes': 1, 'No': 0})

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Apply SMOTEENN to the training data
            logging.info("Applying SMOTEENN on Training dataset")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info(f"Train final features shape after SMOTEENN: {input_feature_train_final.shape}")

            # Do NOT apply SMOTEENN on the testing data to avoid data leakage
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df
            logging.info(f"Test features shape remains unchanged: {input_feature_test_final.shape}")

            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]
            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

