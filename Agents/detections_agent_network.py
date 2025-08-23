def preprocess_data(df, fit_transformers=True, scaler=None, label_encoder=None, feature_columns=None):
    """
    Preprocesses the network data for training or prediction.
    
    Args:
        df: Input DataFrame
        fit_transformers: Whether to fit new transformers (True for training, False for prediction)
        scaler: Pre-fitted StandardScaler (for prediction)
        label_encoder: Pre-fitted LabelEncoder (for prediction)
        feature_columns: List of feature columns to use (for prediction)
    
    Returns:
        Dictionary containing preprocessed data and fitted transformers
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from xgboost import XGBClassifier

    # Selected features for the model
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
    
    print(f"Input data shape: {df.shape}")
    
    # Clean data - handle infinities and NaNs
    df_clean = df.copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    print(f"Shape after cleaning infinities/NaNs: {df_clean.shape}")
    
    # Feature selection
    if fit_transformers:
        # Training mode - use original selected features
        available_features = [feature for feature in selected_features if feature in df_clean.columns]
        if 'Label' in df_clean.columns:
            df_processed = df_clean[available_features + ['Label']]
        else:
            df_processed = df_clean[available_features]
    else:
        # Prediction mode - use pre-determined feature columns
        if feature_columns is not None:
            df_processed = df_clean[feature_columns]
        else:
            available_features = [feature for feature in selected_features if feature in df_clean.columns]
            df_processed = df_clean[available_features]
    
    print(f"Shape after feature selection: {df_processed.shape}")
    
    # Prepare features and target
    if 'Label' in df_processed.columns:
        X = df_processed.drop('Label', axis=1)
        y = df_processed['Label']
    else:
        X = df_processed
        y = None
    
    # Initialize transformers if fitting
    if fit_transformers:
        scaler = StandardScaler()
        if y is not None:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            label_encoder = None
            y_encoded = None
        
        # Fit and transform features
        X_scaled = scaler.fit_transform(X)
        feature_columns = X.columns.tolist()
        
    else:
        # Transform using pre-fitted transformers
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_transformers=False")
        X_scaled = scaler.transform(X)
        y_encoded = None
        
    print(f"Using {X.shape[1]} features for the model.")
    
    result = {
        'X_scaled': X_scaled,
        'y_encoded': y_encoded,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': X.columns.tolist() if fit_transformers else feature_columns,
        'X_raw': X,
        'y_raw': y
    }
    
    return result


def train_model(file_path):
    """
    Trains the XGBoost detection model and saves it along with preprocessing artifacts.
    
    Args:
        file_path: Path to the CSV file containing training data
    """

    print(f"Loading dataset from '{file_path}'...")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from xgboost import XGBClassifier
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        df = df.head(1100000)  # Limit dataset size
        print(f"Dataset '{file_path}' loaded successfully!")
    except FileNotFoundError:
        print("File not found!")
        return
    
    # Preprocess data
    preprocessing_result = preprocess_data(df, fit_transformers=True)
    
    X_scaled = preprocessing_result['X_scaled']
    y_encoded = preprocessing_result['y_encoded']
    scaler = preprocessing_result['scaler']
    label_encoder = preprocessing_result['label_encoder']
    feature_columns = preprocessing_result['feature_columns']
    X_raw = preprocessing_result['X_raw']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train XGBoost model
    print("\n--- Training the Detection Model using XGBoost ---")
    
    model = XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # Evaluate model
    print("\n--- Evaluating Model Performance on Test Set ---")
    y_pred = model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance analysis
    print("\n--- Feature Importance Analysis ---")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance_df.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Importances from XGBoost Model', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    # Save model and preprocessing artifacts
    os.makedirs('Models', exist_ok=True)
    
    # Save model
    with open('Models/Network/detection_model_network.pkl', 'wb+') as f:
        pickle.dump(model, f)
    
    # Save preprocessing artifacts
    with open('Models/Network/scaler_network.pkl', 'wb+') as f:
        pickle.dump(scaler, f)
    
    with open('Models/Network/label_encoder_network.pkl', 'wb+') as f:
        pickle.dump(label_encoder, f)
    
    with open('Models/Network/feature_columns_network.pkl', 'wb+') as f:
        pickle.dump(feature_columns, f)
    
    print("\nModel and preprocessing artifacts saved successfully!")
    print("- Model: Models/Network/detection_model_network.pkl")
    print("- Scaler: Models/Network/scaler_network.pkl")
    print("- Label Encoder: Models/Network/label_encoder_network.pkl")
    print("- Feature Columns: Models/Network/feature_columns_network.pkl")


def predict_data(df):
    import pandas as pd
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from xgboost import XGBClassifier
    """
    Makes predictions on new data using the trained model.
    
    Args:
        df: DataFrame containing the features to predict
    
    Returns:
        Dictionary containing predictions, probabilities, and decoded labels
    """
    # Load model and preprocessing artifacts
    try:
        with open('Models/Network/detection_model_network.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('Models/Network/scaler_network.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('Models/Network/label_encoder_network.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('Models/Network/feature_columns_network.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        print("Please ensure the model has been trained and saved first.")
        return None
    
    print("Model and preprocessing artifacts loaded successfully.")
    
    # Preprocess the input data
    preprocessing_result = preprocess_data(
        df, 
        fit_transformers=False, 
        scaler=scaler, 
        label_encoder=label_encoder,
        feature_columns=feature_columns
    )
    
    X_scaled = preprocessing_result['X_scaled']
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Decode predictions to original labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    print(type(probabilities[0][0]))
    
    results = {
        'predicted_labels': predicted_labels[0],
        'probabilities': probabilities[0][0].item(),
    }
    
    print(f"Predictions generated for {len(predictions)} samples.")
    
    return results