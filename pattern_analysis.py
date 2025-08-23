import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('balanced_data.csv')
df = df.sample(n=10000, random_state=42)

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("=== ANALYZING FEATURES FOR EASY CLASSIFICATION PATTERNS ===")

# Check features that might make classification too easy
print("\n1. Checking Destination Port patterns by label:")
port_by_label = df.groupby('Label')['Destination Port'].agg(['mean', 'std', 'count'])
print(port_by_label)

print("\n2. Checking specific ports for each label:")
for label in df['Label'].unique():
    label_data = df[df['Label'] == label]
    top_ports = label_data['Destination Port'].value_counts().head(5)
    print(f"\n{label} - Top 5 ports:")
    print(top_ports)

print("\n3. Checking if certain features are label-specific:")
features_to_check = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count']

for feature in features_to_check:
    print(f"\n{feature} by Label:")
    feature_stats = df.groupby('Label')[feature].agg(['mean', 'std', 'min', 'max'])
    print(feature_stats)
    
    # Check if any label has very different ranges
    means = feature_stats['mean']
    if means.max() / (means.min() + 1e-10) > 100:  # Avoid division by zero
        print(f"  ⚠️ WARNING: {feature} has very different ranges across labels!")

print("\n4. Checking for perfect separators:")
# Check if any single feature can perfectly separate classes
selected_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'ACK Flag Count'
]

for feature in selected_features:
    if feature in df.columns:
        # Check if any values are exclusive to certain labels
        feature_label_combos = df.groupby([feature, 'Label']).size().unstack(fill_value=0)
        
        # Find values that appear in only one label
        exclusive_values = []
        for value in feature_label_combos.index:
            non_zero_labels = (feature_label_combos.loc[value] > 0).sum()
            if non_zero_labels == 1:
                label = feature_label_combos.loc[value].idxmax()
                count = feature_label_combos.loc[value, label]
                exclusive_values.append((value, label, count))
        
        if len(exclusive_values) > 0:
            print(f"\n{feature} has values exclusive to certain labels:")
            for value, label, count in exclusive_values[:10]:  # Show first 10
                print(f"  Value {value} -> Only in {label} ({count} times)")

print("\n5. Testing model with shuffled labels (should get ~random accuracy):")
# This will tell us if the model is actually learning patterns or just memorizing
df_shuffled = df.copy()
df_shuffled['Label'] = df_shuffled['Label'].sample(frac=1, random_state=123).reset_index(drop=True)

available_features = [f for f in selected_features if f in df_shuffled.columns]
X_shuffled = df_shuffled[available_features]
y_shuffled = df_shuffled['Label']

le_shuffled = LabelEncoder()
y_shuffled_encoded = le_shuffled.fit_transform(y_shuffled)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_shuffled, y_shuffled_encoded, test_size=0.2, random_state=42, stratify=y_shuffled_encoded
)

scaler_s = StandardScaler()
X_train_s_scaled = scaler_s.fit_transform(X_train_s)
X_test_s_scaled = scaler_s.transform(X_test_s)

model_shuffled = XGBClassifier(
    n_estimators=2,
    learning_rate=0.1,
    max_depth=3,
    random_state=47,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model_shuffled.fit(X_train_s_scaled, y_train_s)
y_pred_shuffled = model_shuffled.predict(X_test_s_scaled)
shuffled_accuracy = accuracy_score(y_test_s, y_pred_shuffled)

print(f"Accuracy with shuffled labels: {shuffled_accuracy:.4f}")
print("Expected random accuracy (1/num_classes):", 1/len(df['Label'].unique()))

if shuffled_accuracy > 0.3:  # If still high with shuffled labels
    print("⚠️ WARNING: High accuracy even with shuffled labels suggests data leakage or overfitting!")
