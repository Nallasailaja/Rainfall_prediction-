import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Load the dataset
data = pd.read_csv('usa_rain_prediction_dataset_2024_2025.csv')

# Encode target variable
le = LabelEncoder()
data['Rain Tomorrow'] = le.fit_transform(data['Rain Tomorrow'])

# Features and target
X = data.drop(['Rain Tomorrow', 'Date'], axis=1)
y = data['Rain Tomorrow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding for train and test
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluation
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Model Accuracy: {accuracy:.2f}")

# Save model, scaler, and expected features
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/rainfall_prediction_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Save the expected features
expected_features = X_train.columns.tolist()
joblib.dump(expected_features, 'model/expected_features.pkl')

print("✅ Model, Scaler, and expected features saved successfully in the 'model/' directory.")

# Function to prepare input data
def prepare_input(input_data, train_columns):
    input_encoded = pd.get_dummies(pd.DataFrame([input_data]))
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)
    return input_encoded
