# main.py
from src.pipeline.train_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    accuracy = pipeline.run_pipeline()
    print(f"Model accuracy: {accuracy}")

