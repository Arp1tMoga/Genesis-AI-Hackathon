import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data (smaller sample for debugging)
df = pd.read_csv('balanced_data.csv')
df = df.sample(n=50000, random_state=42)  # Smaller sample for debugging

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Feature selection (using same features as original)
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

available_features = [feature for feature in selected_features if feature in df.columns]
df = df[available_features + ['Label']]

# Split features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Label distribution in training set:")
print(pd.Series(y_train).value_counts())
print("\nLabel names:", le.classes_)

# Test different n_estimators values
estimator_values = [1, 2, 5, 10, 50, 100]

for n_est in estimator_values:
    print(f"\n=== Testing with {n_est} estimators ===")
    
    model = XGBClassifier(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=47,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {n_est} estimators: {accuracy:.4f}")
    
    # Check if model is just predicting the most common class
    unique_predictions = np.unique(y_pred)
    print(f"Number of unique predictions: {len(unique_predictions)} out of {len(le.classes_)} classes")
    
    # Show prediction distribution
    pred_counts = pd.Series(y_pred).value_counts()
    print("Prediction distribution:")
    for i, count in pred_counts.items():
        print(f"  {le.classes_[i]}: {count} ({count/len(y_pred)*100:.1f}%)")
