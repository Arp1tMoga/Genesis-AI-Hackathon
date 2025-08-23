# -*- coding: utf-8 -*-
"""
End-to-End Cybersecurity Detection Pipeline (XGBoost Baseline without SMOTE).
"""

# 1. SETUP: IMPORTING LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# Configure settings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format


# 2. FEATURE SELECTION
# =============================================================================
selected_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'ACK Flag Count',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow IAT Mean',
    'Flow IAT Min', 'Flow IAT Max', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Packet Length Std', 'Packet Length Variance', 'Idle Mean', 'Idle Max',
    'Idle Min', 'Active Mean', 'Active Max', 'Active Min'
]


# 3. DATA LOADING AND PREPROCESSING
# =============================================================================
FILE_PATH = 'balanced_data.csv'
try:
    df = pd.read_csv(FILE_PATH)
    # randomly take 100000 samples
    df = df.sample(n=1000000, random_state=42)
    print(f"Dataset '{FILE_PATH}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: '{FILE_PATH}' not found. Using a dummy dataframe.")
    columns = selected_features + ['Label']
    dummy_data = {col: np.random.rand(100) for col in columns}
    dummy_data['Label'] = np.random.choice(['BENIGN', 'DDoS', 'PortScan'], 100)
    df = pd.DataFrame(dummy_data)

print(f"Original shape: {df.shape}")

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape after cleaning infinities/NaNs: {df.shape}")

# Feature selection
available_features = [feature for feature in selected_features if feature in df.columns]
df = df[available_features + ['Label']]
print(f"Shape after feature selection: {df.shape}")
print(f"\nUsing {len(available_features)} features for the model.")

#only use the first 100000 samples

# Define Features (X) and Target (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. MODEL TRAINING (XGBoost without SMOTE)
# =============================================================================
print("\n--- Training the Detection Model using XGBoost (Baseline without SMOTE) ---")

model = XGBClassifier(
    n_estimators=2,       # Number of trees
    learning_rate=0.1,      # Step size shrinkage
    max_depth=3,            # Tree depth
    subsample=0.8,          # Row sampling
    colsample_bytree=0.8,   # Feature sampling
    random_state=47,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='mlogloss'  # Avoids warning for classification
)

model.fit(X_train_scaled, y_train)
print("Model training complete.")


# 5. MODEL EVALUATION
# =============================================================================
print("\n--- Evaluating Model Performance on the Test Set ---")
y_pred = model.predict(X_test_scaled)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix', fontsize=20)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n--- Feature Importance Analysis (XGBoost) ---")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 most important features:")
print(feature_importance_df.head(10))

# 7. SAMPLE PREDICTIONS (First 5 Test Cases)
# =============================================================================
print("\n--- Sample Predictions (First 5 Test Cases) ---")

# Select first 5 samples from X_test
sample_X = X_test_scaled[:5]
sample_original_X = X_test.iloc[:5]
sample_y = y_test[:5]

# Predictions & probabilities
predictions = model.predict(sample_X)
probabilities = model.predict_proba(sample_X)

# Decode labels
actual_labels = le.inverse_transform(sample_y)
predicted_labels = le.inverse_transform(predictions)

# Display results
for i in range(5):
    print(f"\nTest Case {i+1}:")
    print(f"Features: {sample_original_X.iloc[i].to_dict()}")
    print(f"Actual Label: {actual_labels[i]}")
    print(f"Predicted Label: {predicted_labels[i]}")

    # Top 2 probabilities
    prob_dict = dict(zip(le.classes_, probabilities[i]))
    top2 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"Top 2 Predicted Classes with Probabilities: {top2}")
    print("-" * 60)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Feature Importances from XGBoost Model', fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()