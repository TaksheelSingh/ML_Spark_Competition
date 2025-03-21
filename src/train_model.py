import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv('data/train.csv')

def preprocess_data(df):
    # Ensure the target column name matches exactly
    if 'Target' not in df.columns:
        raise KeyError("Target column not found in dataset!")
    
    # Separate features and target
    X = df.drop(columns=['id', 'Target'])  # Drop 'id' if not needed
    y = df['Target']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Apply Label Encoding to categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # Store encoders for future use
    
    return X, y, label_encoders

# Preprocess data
X, y, label_encoders = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Model Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/trained_model.pkl')
print("Model saved successfully!")

# Ensure the 'output' directory exists
os.makedirs('output', exist_ok=True)

# Predict probabilities for submission
test_probabilities = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'ID': df.index[:len(test_probabilities)], 'TARGET': test_probabilities})
submission.to_csv('output/submission.csv', index=False)
print("Submission file saved as output/submission.csv")

print("Pipeline Execution Completed! Check the output folder for results.")
