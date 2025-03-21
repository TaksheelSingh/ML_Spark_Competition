import os

print("Starting ML Spark Competition Pipeline...")

# Step 1: Install Dependencies
os.system("pip install -r requirements.txt")

# Step 2: Run Model Training & Prediction
os.system("python src/train_model.py")

print("Pipeline Execution Completed! Check the output folder for results.")
