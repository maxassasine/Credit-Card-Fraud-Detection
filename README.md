Credit Card Fraud Detection with XGBoost

📌 Overview

This project builds a credit card fraud detection model using XGBoost without SMOTE. The model is optimized using RandomizedSearchCV to fine-tune hyperparameters. The final threshold for classification is adjusted based on the Precision-Recall curve to balance fraud detection performance.

📂 Dataset

Dataset Used: creditcard.csv

Target Variable: Class (0 = Non-Fraud, 1 = Fraud)

Features: 30 numerical features (excluding Time, which is dropped)

🚀 Steps in the Pipeline

Load and Preprocess Data

Remove duplicate records.

Drop the Time column as it does not contribute to fraud detection.

Train-Test Split

80% training, 20% testing.

Stratified split to maintain fraud ratio.

Handle Class Imbalance

Compute fraud-to-non-fraud ratio.

Use scale_pos_weight in XGBoost to balance the dataset without oversampling.

Train XGBoost Model with Hyperparameter Tuning

Use RandomizedSearchCV to optimize:

n_estimators (number of trees)

max_depth (tree depth)

learning_rate (adjust learning speed)

subsample, colsample_bytree

scale_pos_weight to balance fraud detection

Fine-Tune Decision Threshold

Use Precision-Recall Curve to find the best threshold.

Slightly lower the threshold to increase recall (catch more fraud cases).

Evaluate Performance

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Accuracy Score

🔹 Key Improvements

✅ No SMOTE used – avoids overfitting while balancing fraud detection.✅ Threshold adjustment – prevents high precision at the cost of low recall.✅ Fine-tuned hyperparameters – optimized for fraud classification.✅ Stratified train-test split – preserves fraud ratio in both sets.

📜 Usage

Install dependencies:

pip install numpy pandas scikit-learn xgboost

Run the script:

python fraud_detection.py

Check the output for:

Optimal Decision Threshold

Fine-Tuned Model Performance

📊 Example Results

Optimal Threshold (Adjusted): 0.4350561201572418
📌 Fine-Tuned XGBoost Performance (Without SMOTE):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.94      0.79      0.86        95

    accuracy                           1.00     56746
   macro avg       0.97      0.89      0.93     56746
weighted avg       1.00      1.00      1.00     56746

Confusion Matrix:
[[56646     5]
 [   20    75]]
Accuracy: 0.9995594403129736

📌 Next Steps

Further optimize the threshold dynamically based on business needs.

Test with real-world fraud data.

Deploy the model as an API for real-time fraud detection.

✅ Developed with XGBoost for efficient fraud detection.
