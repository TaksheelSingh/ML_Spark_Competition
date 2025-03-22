# ML Spark Competition

## Project Overview
This project focuses on developing a predictive framework to optimize customer engagement in bank marketing campaigns using machine learning.

## Folder Structure
```
ML_Spark_Competition/
│── data/               # Store dataset here
│   ├── train.csv       # Training data (to be added by user)
│   ├── test.csv        # Test data (to be added by user)
│
│── src/                # Python scripts for model building
│   ├── preprocess.py   # Data cleaning & feature engineering
│   ├── train_model.py  # Model training script
│   ├── predict.py      # Generate predictions
│
│── output/             # Save results & submissions
│   ├── submission.csv  # Final Kaggle submission file
│
│── requirements.txt    # Dependencies
│── README.md           # Instructions to run the project
```

## How to Run the Project
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Place `train.csv` and `test.csv` inside the `data/` folder.
3. Run the model training script:
   ```sh
   python src/train_model.py
   ```
4. The submission file will be saved in the `output/` folder as `submission.csv`.

## Model Used
- XGBoost Classifier
- Feature Engineering: One-Hot Encoding & Standard Scaling
- Evaluation Metric: F1 Score

## Submission Format
The final submission file format:
```
ID,TARGET
2,0.85
5,0.32
6,0.78
```
