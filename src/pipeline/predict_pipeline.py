import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

class CustomData:
    def __init__(self,
        customerID:int,      
        gender: str,   
        SeniorCitizen: int,   
        Partner: str,           
        Dependents: str,          
        tenure: int,              
        PhoneService        
        MultipleLines       
        InternetService     
        OnlineSecurity     
        OnlineBackup        
        DeviceProtection    
        TechSupport         
        StreamingTV         
        StreamingMovies     
        Contract            
        PaperlessBilling    
        PaymentMethod       
        MonthlyCharges      
        TotalCharges        
Churn)
        pass