import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('balanced_data.csv')
df = df.sample(n=10000, random_state=42)  # Even smaller sample for detailed analysis

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Feature selection
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

print("Data shape:", df.shape)
print("\nLabel distribution:")
print(df['Label'].value_counts())
print("\nLabel percentages:")
print(df['Label'].value_counts(normalize=True) * 100)

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

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# 1. Baseline: Dummy classifier (most frequent class)
print("\n=== BASELINE: Dummy Classifier (Most Frequent) ===")
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_scaled, y_train)
dummy_pred = dummy.predict(X_test_scaled)
dummy_accuracy = accuracy_score(y_test, dummy_pred)
print(f"Dummy classifier accuracy: {dummy_accuracy:.4f}")

# 2. XGBoost with 2 estimators (your original setting)
print("\n=== XGBoost with 2 estimators ===")
model = XGBClassifier(
    n_estimators=2,
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

# Training accuracy
train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)

# Test accuracy
test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Difference (potential overfitting): {train_accuracy - test_accuracy:.4f}")

# 3. Detailed analysis of predictions
print("\n=== Detailed Analysis ===")
print("Actual test distribution:")
test_labels = [le.classes_[i] for i in y_test]
actual_counts = pd.Series(test_labels).value_counts()
print(actual_counts)

print("\nPredicted test distribution:")
pred_labels = [le.classes_[i] for i in test_pred]
pred_counts = pd.Series(pred_labels).value_counts()
print(pred_counts)

# Check if predictions match the original distribution too closely
print("\n=== Suspicious Pattern Check ===")
for class_name in le.classes_:
    if class_name in actual_counts.index and class_name in pred_counts.index:
        actual_pct = actual_counts[class_name] / len(y_test) * 100
        pred_pct = pred_counts[class_name] / len(test_pred) * 100
        print(f"{class_name}: Actual {actual_pct:.1f}%, Predicted {pred_pct:.1f}%")

# 4. Check feature importance
print("\n=== Feature Importance ===")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top 10 features:")
print(feature_importance_df.head(10))

# 5. Check if any feature has unusual values
print("\n=== Feature Statistics ===")
print("Features with high importance and their statistics:")
for i in range(min(5, len(feature_importance_df))):
    feat = feature_importance_df.iloc[i]['Feature']
    values = X[feat]
    print(f"\n{feat}:")
    print(f"  Min: {values.min():.2f}, Max: {values.max():.2f}")
    print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")
    print(f"  Unique values: {len(values.unique())}")
    
    # Check for potential data leakage indicators
    unique_vals = len(values.unique())
    if unique_vals < 10:
        print(f"  WARNING: Only {unique_vals} unique values - possible categorical encoding issue")
    if values.std() == 0:
        print(f"  WARNING: Zero variance - this feature is constant!")
