# Importing the  necessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Loading the dataset required 
data_path = 'C://Users//prava//OneDrive//Desktop//profcess//proj data.csv'
data = pd.read_csv(data_path)

# Step 2: Data Cleaning process 
data.columns = data.iloc[0]  # Set the first row as header
data = data.drop(0)  # Drop the header row from the data
data = data.rename(columns=lambda x: x.strip())  # Remove any leading/trailing spaces from column names

# Convert all columns to numeric, coercing any non-numeric values to NaN, and drop NaNs
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()  # Drop rows with missing values

# Step 3: Feature and Target Separation process
X = data.drop(columns=['class'])
y = data['class']

# Step 4: Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5a: Train and Evaluate Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_report = classification_report(y_test, y_pred_log_reg)

# Step 5b: Train and Evaluate XGBoost Model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb)

# Printing the  Results
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression Report:\n", log_reg_report)
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Report:\n", xgb_report)
