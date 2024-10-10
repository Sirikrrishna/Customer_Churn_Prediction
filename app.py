from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for data prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Retrieve and process form data
        data = CustomData(
            gender=request.form.get('gender'),
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=request.form.get('tenure'),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=request.form.get('MonthlyCharges'),
            TotalCharges=request.form.get('TotalCharges'),
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)


        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results="Yes" if results[0] == 1 else "No")

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
