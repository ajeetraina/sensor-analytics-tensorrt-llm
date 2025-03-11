import os
import numpy as np
import torch
import tensorrt as trt
import json
from datetime import datetime

def load_training_data(log_file):
    """
    Load training data from log file
    
    Args:
        log_file: Path to log file with raw and filtered sensor data
        
    Returns:
        X_train, y_train arrays for model training
    """
    X = []
    y = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                raw_data = entry['data']['raw']
                filtered_data = entry['data']['filtered']
                
                # Feature vector: normalized sensor values
                features = [
                    raw_data['temperature'] / 100.0,
                    raw_data['humidity'] / 100.0,
                    raw_data['pressure'] / 1100.0,
                    np.log10(max(raw_data['gas'], 1)) / 5.0
                ]
                
                # Target: validity flag and normalized filtered values
                # If raw == filtered, data is valid (1.0), otherwise invalid (< 1.0)
                is_valid = 1.0 if entry['status'] == 'normal' else 0.5
                
                targets = [
                    is_valid,
                    filtered_data['temperature'] / 100.0,
                    filtered_data['humidity'] / 100.0,
                    filtered_data['pressure'] / 1100.0,
                    np.log10(max(filtered_data['gas'], 1)) / 5.0
                ]
                
                X.append(features)
                y.append(targets)
            except Exception as e:
                print(f"Error processing log entry: {e}")
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_onnx_model(input_shape, output_shape, hidden_layers=[16, 8]):
    """
    Create a PyTorch model and export to ONNX
    
    Args:
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        hidden_layers: List of hidden layer sizes
        
    Returns:
        Path to exported ONNX model
    """
    class SensorModel(torch.nn.Module):
        def __init__(self, input_size, output_size, hidden_layers):
            super(SensorModel, self).__init__()
            
            # Build network layers
            layers = []
            prev_size = input_size
            
            for size in hidden_layers:
                layers.append(torch.nn.Linear(prev_size, size))
                layers.append(torch.nn.ReLU())
                prev_size = size
            
            layers.append(torch.nn.Linear(prev_size, output_size))
            
            # Add sigmoid activation for the first output (validity score)
            self.network = torch.nn.Sequential(*layers)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            outputs = self.network(x)
            # Apply sigmoid only to the first output (validity score)
            outputs[:, 0] = self.sigmoid(outputs[:, 0])
            return outputs
    
    # Create model
    input_size = input_shape[1]
    output_size = output_shape[1]
    model = SensorModel(input_size, output_size, hidden_layers)
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    onnx_path = "models/sensor_model.onnx"
    os.makedirs("models", exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    print(f"ONNX model exported to {onnx_path}")
    return onnx_path

def build_tensorrt_engine(onnx_path):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Path to TensorRT engine
    """
    # Initialize TRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set max workspace size (1GB)
    config.max_workspace_size = 1 << 30
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Build engine
    engine_path = onnx_path.replace('.onnx', '.engine')
    engine = builder.build_engine(network, config)
    
    # Serialize engine to file
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine built and saved to {engine_path}")
    return engine_path

def train_model(X_train, y_train, epochs=500, lr=0.001):
    """
    Train PyTorch model on sensor data
    
    Args:
        X_train: Training features
        y_train: Training targets
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained PyTorch model
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    class SensorModel(torch.nn.Module):
        def __init__(self, input_size, output_size, hidden_layers=[16, 8]):
            super(SensorModel, self).__init__()
            
            # Build network layers
            layers = []
            prev_size = input_size
            
            for size in hidden_layers:
                layers.append(torch.nn.Linear(prev_size, size))
                layers.append(torch.nn.ReLU())
                prev_size = size
            
            layers.append(torch.nn.Linear(prev_size, output_size))
            
            self.network = torch.nn.Sequential(*layers)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            outputs = self.network(x)
            # Apply sigmoid only to the first output (validity score)
            outputs[:, 0] = self.sigmoid(outputs[:, 0])
            return outputs
    
    model = SensorModel(input_size, output_size)
    
    # Define loss function and optimizer
    # Custom loss: MSE for filtered values, BCE for validity flag
    def custom_loss(pred, target):
        # MSE for the filtered values (outputs 1-4)
        mse_loss = torch.nn.functional.mse_loss(pred[:, 1:], target[:, 1:])
        
        # BCE for the validity flag (output 0)
        bce_loss = torch.nn.functional.binary_cross_entropy(pred[:, 0], target[:, 0])
        
        # Combine losses
        return mse_loss + bce_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = custom_loss(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}')
    
    print("Training completed")
    return model

def generate_sample_data(num_samples=1000):
    """
    Generate sample training data if real logs are not available
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        X_train, y_train arrays
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate normal readings
        temp = np.random.uniform(15, 35)  # 15-35°C
        humidity = np.random.uniform(30, 80)  # 30-80%
        pressure = np.random.uniform(980, 1030)  # 980-1030 hPa
        gas = np.random.uniform(1000, 20000)  # 1000-20000 ohms
        
        # Features
        features = [
            temp / 100.0,
            humidity / 100.0,
            pressure / 1100.0,
            np.log10(gas) / 5.0
        ]
        
        # Randomly make some samples anomalous
        is_anomaly = np.random.random() < 0.1  # 10% anomalies
        
        if is_anomaly:
            # Add significant deviation to one random reading
            anomaly_idx = np.random.randint(0, 4)
            if anomaly_idx == 0:
                # Temperature anomaly
                temp = np.random.choice([
                    np.random.uniform(-10, 5),  # Too cold
                    np.random.uniform(45, 60)   # Too hot
                ])
                features[0] = temp / 100.0
            elif anomaly_idx == 1:
                # Humidity anomaly
                humidity = np.random.choice([
                    np.random.uniform(0, 20),    # Too dry
                    np.random.uniform(90, 110)   # Too humid
                ])
                features[1] = humidity / 100.0
            elif anomaly_idx == 2:
                # Pressure anomaly
                pressure = np.random.choice([
                    np.random.uniform(900, 950),    # Too low
                    np.random.uniform(1050, 1100)   # Too high
                ])
                features[2] = pressure / 1100.0
            else:
                # Gas anomaly
                gas = np.random.choice([
                    np.random.uniform(100, 500),       # Too low
                    np.random.uniform(30000, 50000)    # Too high
                ])
                features[3] = np.log10(gas) / 5.0
            
            # Targets for anomalies - validity score is lower
            is_valid = 0.2  # Low validity score
            
            # Corrected values (what we'd expect)
            if anomaly_idx == 0:
                corrected_temp = np.clip(temp, 15, 35)
                targets = [
                    is_valid,
                    corrected_temp / 100.0,
                    humidity / 100.0,
                    pressure / 1100.0,
                    np.log10(gas) / 5.0
                ]
            elif anomaly_idx == 1:
                corrected_humidity = np.clip(humidity, 30, 80)
                targets = [
                    is_valid,
                    temp / 100.0,
                    corrected_humidity / 100.0,
                    pressure / 1100.0,
                    np.log10(gas) / 5.0
                ]
            elif anomaly_idx == 2:
                corrected_pressure = np.clip(pressure, 980, 1030)
                targets = [
                    is_valid,
                    temp / 100.0,
                    humidity / 100.0,
                    corrected_pressure / 1100.0,
                    np.log10(gas) / 5.0
                ]
            else:
                corrected_gas = np.clip(gas, 1000, 20000)
                targets = [
                    is_valid,
                    temp / 100.0,
                    humidity / 100.0,
                    pressure / 1100.0,
                    np.log10(corrected_gas) / 5.0
                ]
        else:
            # Normal data - all readings are valid
            targets = [
                1.0,  # Valid
                temp / 100.0,
                humidity / 100.0,
                pressure / 1100.0,
                np.log10(gas) / 5.0
            ]
        
        X.append(features)
        y.append(targets)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    """Main function to build TensorRT LLM model for sensor data filtering"""
    print("Building TensorRT LLM model for sensor data filtering...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if we have training data
    if os.path.exists("sensor_logs.json"):
        print("Loading training data from logs...")
        X_train, y_train = load_training_data("sensor_logs.json")
    else:
        print("No log data found. Generating synthetic training data...")
        X_train, y_train = generate_sample_data(num_samples=5000)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Train model
    model = train_model(X_train, y_train, epochs=500)
    
    # Export to ONNX
    input_shape = (1, X_train.shape[1])  # Batch size of 1, 4 features
    output_shape = (1, y_train.shape[1])  # Batch size of 1, 5 outputs
    onnx_path = create_onnx_model(input_shape, output_shape)
    
    # Build TensorRT engine
    engine_path = build_tensorrt_engine(onnx_path)
    
    print(f"TensorRT model built successfully: {engine_path}")
    print("You can now use this model with the sensor_filter.py module")

if __name__ == "__main__":
    main()