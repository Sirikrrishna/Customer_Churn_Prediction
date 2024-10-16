<h1 align="center">Customer Churn Prediction</h1>

An **end-to-end machine learning pipeline** to predict customer churn, including training, prediction, and deployment to AWS EC2 with Docker and GitHub Actions.
### Table of Contents
## 📚 Table of Contents
- [About The Project](#about-the-project)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Training Pipeline](#training-pipeline)
- [Prediction Pipeline](#prediction-pipeline)
- [Deployment](#deployment)
- [Web Application](#web-application)
- [AWS CI/CD Deployment](#aws-cicd-deployment)
- [Export Environment Variables](#export-environment-variables)
- [Setup Github Secrets](#setup-github-secrets)

## 🔍 About The Project

This project is designed to **accurately predict customer churn**. The pipeline involves **data ingestion**, **transformation**, **model training**, **evaluation**, and **deployment**.

### Project Structure
├── .github
│   └── workflows/
├── artifacts/
├── mlproject_env 
├── notebook       
├── src/
│   ├── components/
│   ├── configuration/
│   ├── exception.py
│   ├── logger.py
│   ├── pipeline/
│   └── utils.py
├── templates/
├── .gitignore
├── DOCKERFILE
├── app.py
├── requirements.txt
├── setup.py

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Database**: MongoDB
- **Containerization**: Docker
- **Cloud Services**:
  - **AWS EC2**: Hosting the application
  - **AWS ECR**: Docker image repository
- **CI/CD**: GitHub Actions
- **Web Application**: HTML

## 🖥️ How to Run

Instructions to set up your local environment for running the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sirikrrishna/Customer_Churn_Prediction.git
   cd Customer_Churn_Prediction
2. Set up a virtual environment:
   ```bash
   conda create -n mlproject_env python=3.8 -y
   conda activate mlproject_env
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   #run
   python src/pipeline/main.py
   python app.py

## ⚙️ Training Pipeline

### Data Ingestion:
- Ingest data from MongoDB.
- Split the data into train and test sets.
- Export the data from MongoDB to CSV files for further processing.

### Data Transformation:
- Transform raw data into a suitable format for model building.
- Apply transformations like **One-Hot Encoding**, **Ordinal Encoding**, and **Scaling**.
- Handle imbalanced data using techniques like **SMOTEENN**.

### Model Trainer and Evaluation:
- Train multiple machine learning models such as **K-Neighbors**, **Random Forest**, **SVM**, **XGBoost**, and **CatBoost**.
- Perform hyperparameter tuning using **GridSearchCV** to find the best model with optimal parameters.
- Evaluate the best model from the training pipeline.
- Use the best-performing model for predictions on test data.

### Train Pipeline:
- Execute the entire training pipeline to process data, train, evaluate, and deploy the model.

## 🔮 Prediction Pipeline

### Ingest New Data:
- Ingest new or unseen data from users or MongoDB.

### Data Transformation:
- Transform the new data using the preprocessing steps from the training pipeline.

### Make Predictions:
- Use the best-trained model to make predictions on the transformed data.

## 🚀 Deployment

### Containerize the Application:
- Use **Docker** to containerize the application for easy deployment and scalability.
- Store the Docker image in the **AWS ECR** repository.

### Set Up AWS EC2 Instance:
- Host the deployed application on an **AWS EC2** instance.
- Pull the Docker image from **AWS ECR** and run the application on EC2.

### Automate Deployment with GitHub Actions:
- Use **GitHub Actions** to automate the deployment workflow.
- On each code push:
  - Retrain the model.
  - Build the Docker image.
  - Push it to **AWS ECR**.
  - Pull the image to **EC2**.
  - Run the application.


## 🌐 Web Application
- Build a basic web application using **FLASK** and **HTML** to expose the model's prediction functionality.
- The web app allows users to input customer data and receive predictions on churn status.
- Ensure that the front-end is user-friendly and responsive to enhance user experience.

## ⚙️ AWS CI/CD Deployment with GitHub Actions
1. **Login to AWS Console.**
2. **Create IAM User for Deployment** with specific access:
   - **EC2 access:** It is a virtual machine.
   - **ECR:** Elastic Container Registry to save your Docker image in AWS.

### Description of the Deployment Steps:
- Build Docker image of the source code.
- Push your Docker image to **ECR**.
- Launch your **EC2** instance.
- Pull your image from **ECR** in **EC2**.
- Launch your Docker image in **EC2**.

### Policy:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

### Create ECR Repo to Store/Save Docker Image:
- Save the URI: `235494811035.dkr.ecr.us-east-1.amazonaws.com/customer_churn`

### Create EC2 Machine (Ubuntu):
- Open EC2 and Install Docker in the EC2 Machine:

#### Optional:

   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade

### Required:

    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker

### Configure EC2 as Self-Hosted Runner:
Go to **Settings > Actions > Runners > New Self-Hosted Runner**.
Choose your **OS** and run the provided commands one by one.

### Export Environment Variables

Before running your application, make sure to export the following environment variables in your terminal:

   ```bash
   export MONGODB_URL="mongodb+srv://<username>:<password>...."
   export AWS_ACCESS_KEY_ID="<Your AWS Access Key ID>"
   export AWS_SECRET_ACCESS_KEY="<Your AWS Secret Access Key>"


### Setup GitHub Secrets

To configure your GitHub repository secrets, add the following key-value pairs:

- **AWS_ACCESS_KEY_ID**: `<Your AWS Access Key ID>`
- **AWS_SECRET_ACCESS_KEY**: `<Your AWS Secret Access Key>`
- **AWS_REGION**: `us-east-1`
- **AWS_ECR_LOGIN_URI**: `235494811035.dkr.ecr.us-east-1.amazonaws.com`
- **ECR_REPOSITORY_NAME**: `customer_churn`
