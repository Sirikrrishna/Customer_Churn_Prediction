import sys
import pandas as pd
from pymongo import MongoClient
from src.exception import CustomException
from src.logger import logging

class MongoDBConnection:
    def __init__(self, mongo_uri, database_name, collection_name):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name

    def connect_to_mongo(self):
        """Connect to MongoDB and return the client."""
        try:
            client = MongoClient(self.mongo_uri)
            logging.info(f"Connected to MongoDB at {self.mongo_uri}")
            return client
        except Exception as e:
            raise CustomException(f"Failed to connect to MongoDB: {e}", sys)

    def get_data_from_mongo(self):
        """Retrieve the data from MongoDB and return as a pandas DataFrame."""
        try:
            client = self.connect_to_mongo()
            db = client[self.database_name]
            collection = db[self.collection_name]

            # Retrieve all documents from the collection and convert to DataFrame
            data = pd.DataFrame(list(collection.find()))
            logging.info(f"Retrieved {len(data)} records from MongoDB.")
            return data
        except Exception as e:
            raise CustomException(f"Error in retrieving data from MongoDB: {e}", sys)
