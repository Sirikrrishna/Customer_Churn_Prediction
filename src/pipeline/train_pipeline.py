# src/pipeline/training_pipeline.py
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer_and_evaluation import ModelTrainer
from src.components.model_trainer_and_evaluation import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def __init__(self):
        logging.info("Initializing the Training Pipeline...")
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        """
        Orchestrates the data ingestion, transformation, model training, and evaluation.
        Returns the model accuracy after training and evaluation.
        """
        try:
            # Step 1: Data Ingestion
            logging.info("Starting the data ingestion process...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Starting the data transformation process...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Training and Evaluation
            logging.info("Starting the model training and evaluation process...")
            model_accuracy = self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed successfully with model accuracy: {model_accuracy}")
            return model_accuracy

        except Exception as e:
            logging.error(f"Error occurred in training pipeline: {str(e)}")
            raise CustomException(e, sys)

# Main entry point for the training pipeline
if __name__ == "__main__":
    logging.info("Running the training pipeline...")
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
