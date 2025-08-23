"""
Example usage of the refactored Detection Agent Network functions.

This example demonstrates how to use the three independent functions:
1. preprocess_data() - Data preprocessing
2. train_model() - Model training  
3. predict_threats() - Threat prediction
"""

from Agents.detections_agent_network import preprocess_data, train_model, predict_threats
import pandas as pd
import numpy as np

def example_complete_pipeline(dataset_path):
    """
    Complete example showing the full pipeline from data preprocessing to prediction.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the cybersecurity dataset CSV file
    """
    print("=" * 100)
    print("COMPLETE DETECTION AGENT PIPELINE EXAMPLE")
    print("=" * 100)
    
    try:
        # Step 1: Preprocess the data
        print("\n🔄 STEP 1: Preprocessing Data")
        print("-" * 50)
        preprocessed_data = preprocess_data(dataset_path)
        
        # Step 2: Train the model
        print("\n🤖 STEP 2: Training Model")
        print("-" * 50)
        training_results = train_model(preprocessed_data)
        
        print(f"\n✅ Training completed with accuracy: {training_results['accuracy']:.4f}")
        
        # Step 3: Example predictions
        print("\n🔍 STEP 3: Making Sample Predictions")
        print("-" * 50)
        
        # Example 1: Single prediction with dictionary
        sample_data_single = {
            'Destination Port': 80,
            'Flow Duration': 120000,
            'Total Fwd Packets': 10,
            'Total Backward Packets': 8,
            'Total Length of Fwd Packets': 1500,
            'Total Length of Bwd Packets': 1200,
            'FIN Flag Count': 1,
            'SYN Flag Count': 1,
            'RST Flag Count': 0,
            'ACK Flag Count': 15,
            'Fwd Packet Length Mean': 150.0,
            'Bwd Packet Length Mean': 150.0,
            'Flow IAT Mean': 12000.0,
            'Flow IAT Min': 1000.0,
            'Flow IAT Max': 50000.0,
            'Init_Win_bytes_forward': 8192,
            'Init_Win_bytes_backward': 8192,
            'Packet Length Std': 25.5,
            'Packet Length Variance': 650.25,
            'Idle Mean': 5000.0,
            'Idle Max': 10000.0,
            'Idle Min': 1000.0,
            'Active Mean': 8000.0,
            'Active Max': 12000.0,
            'Active Min': 4000.0
        }
        
        print("\nSingle Prediction Example:")
        single_result = predict_threats(sample_data_single)
        
        # Example 2: Multiple predictions with list of dictionaries
        sample_data_multiple = [
            sample_data_single,
            {**sample_data_single, 'Destination Port': 22, 'Flow Duration': 500000},  # SSH-like
            {**sample_data_single, 'Destination Port': 443, 'SYN Flag Count': 50},     # HTTPS with many SYNs
        ]
        
        print("\nMultiple Predictions Example:")
        multiple_results = predict_threats(sample_data_multiple)
        
        return {
            'preprocessed_data': preprocessed_data,
            'training_results': training_results,
            'single_prediction': single_result,
            'multiple_predictions': multiple_results
        }
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        return None


def example_separate_functions():
    """
    Example showing how to use functions separately (useful for production scenarios).
    """
    print("\n" + "=" * 100)
    print("SEPARATE FUNCTIONS USAGE EXAMPLE")
    print("=" * 100)
    
    # This example assumes you already have a trained model
    # In production, you might preprocess and train once, then only use predict_threats()
    
    # Example: Making predictions with an already trained model
    sample_network_flow = {
        'Destination Port': 80,
        'Flow Duration': 75000,
        'Total Fwd Packets': 5,
        'Total Backward Packets': 3,
        'Total Length of Fwd Packets': 750,
        'Total Length of Bwd Packets': 450,
        'FIN Flag Count': 1,
        'SYN Flag Count': 1,
        'RST Flag Count': 0,
        'ACK Flag Count': 7,
        'Fwd Packet Length Mean': 150.0,
        'Bwd Packet Length Mean': 150.0,
        'Flow IAT Mean': 18750.0,
        'Flow IAT Min': 5000.0,
        'Flow IAT Max': 30000.0,
        'Init_Win_bytes_forward': 65535,
        'Init_Win_bytes_backward': 65535,
        'Packet Length Std': 15.2,
        'Packet Length Variance': 231.04,
        'Idle Mean': 7500.0,
        'Idle Max': 15000.0,
        'Idle Min': 2500.0,
        'Active Mean': 6000.0,
        'Active Max': 9000.0,
        'Active Min': 3000.0
    }
    
    try:
        print("🔍 Making prediction with pre-trained model...")
        result = predict_threats(sample_network_flow)
        
        print(f"✅ Prediction completed!")
        print(f"   Predicted class: {result['predictions'][0]}")
        print(f"   Confidence: {result['confidence_scores'][0]:.4f}")
        print(f"   Is threat: {result['is_threat'][0]}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Note: Make sure to train a model first using the complete pipeline example.")


def create_dummy_dataset(filename="dummy_dataset.csv", num_samples=1000):
    """
    Create a dummy dataset for testing purposes.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file to create
    num_samples : int
        Number of samples to generate
    """
    print(f"📊 Creating dummy dataset with {num_samples} samples...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate random data for each feature
    data = {}
    
    # Network features with realistic ranges
    data['Destination Port'] = np.random.choice([22, 53, 80, 443, 993, 995, 8080], num_samples)
    data['Flow Duration'] = np.random.exponential(50000, num_samples)
    data['Total Fwd Packets'] = np.random.poisson(10, num_samples)
    data['Total Backward Packets'] = np.random.poisson(8, num_samples)
    data['Total Length of Fwd Packets'] = np.random.normal(1500, 500, num_samples)
    data['Total Length of Bwd Packets'] = np.random.normal(1200, 400, num_samples)
    data['FIN Flag Count'] = np.random.poisson(1, num_samples)
    data['SYN Flag Count'] = np.random.poisson(1, num_samples)
    data['RST Flag Count'] = np.random.poisson(0.1, num_samples)
    data['ACK Flag Count'] = np.random.poisson(15, num_samples)
    data['Fwd Packet Length Mean'] = np.random.normal(150, 50, num_samples)
    data['Bwd Packet Length Mean'] = np.random.normal(150, 50, num_samples)
    data['Flow IAT Mean'] = np.random.exponential(10000, num_samples)
    data['Flow IAT Min'] = np.random.exponential(1000, num_samples)
    data['Flow IAT Max'] = np.random.exponential(50000, num_samples)
    data['Init_Win_bytes_forward'] = np.random.choice([8192, 16384, 32768, 65536], num_samples)
    data['Init_Win_bytes_backward'] = np.random.choice([8192, 16384, 32768, 65536], num_samples)
    data['Packet Length Std'] = np.random.exponential(25, num_samples)
    data['Packet Length Variance'] = data['Packet Length Std'] ** 2
    data['Idle Mean'] = np.random.exponential(5000, num_samples)
    data['Idle Max'] = np.random.exponential(10000, num_samples)
    data['Idle Min'] = np.random.exponential(1000, num_samples)
    data['Active Mean'] = np.random.exponential(8000, num_samples)
    data['Active Max'] = np.random.exponential(12000, num_samples)
    data['Active Min'] = np.random.exponential(4000, num_samples)
    
    # Generate labels (80% BENIGN, 10% DDoS, 10% PortScan)
    labels = np.random.choice(['BENIGN', 'DDoS', 'PortScan'], 
                             num_samples, 
                             p=[0.8, 0.1, 0.1])
    data['Label'] = labels
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure no negative values for certain features
    df[df < 0] = 0
    
    df.to_csv(filename, index=False)
    print(f"✅ Dummy dataset saved as '{filename}'")
    print(f"   Shape: {df.shape}")
    print(f"   Label distribution: {df['Label'].value_counts().to_dict()}")
    
    return filename


if __name__ == "__main__":
    print("🚀 Detection Agent Network - Usage Examples")
    print("=" * 100)
    
    # Option 1: Create dummy dataset for testing
    print("\n📊 Creating dummy dataset for demonstration...")
    dummy_file = create_dummy_dataset("demo_dataset.csv", 5000)
    
    # Option 2: Run complete pipeline with dummy data
    print(f"\n🔄 Running complete pipeline with dummy dataset...")
    pipeline_results = example_complete_pipeline(dummy_file)
    
    # Option 3: Show separate function usage
    if pipeline_results:
        example_separate_functions()
    
    print("\n✅ Example completed! Check the Models/ directory for saved model files.")
    print("\n💡 Usage Tips:")
    print("   - Use preprocess_data() once to prepare your dataset")
    print("   - Use train_model() to train and save your model") 
    print("   - Use predict_threats() for real-time predictions")
    print("   - Model files are saved in the Models/ directory")