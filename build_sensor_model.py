#!/usr/bin/env python3
# build_sensor_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import tensorrt as trt
import pandas as pd
from datetime import datetime, timedelta
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

class SensorModel(nn.Module):
    """Neural network model for sensor data processing."""
    def __init__(self, input_size=4, hidden_size=64, output_size=5):
        super(SensorModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # For classification/filtering tasks
        )
    
    def forward(self, x):
        return self.layers(x)

def load_sensor_data(data_path=None):
    """
    Load sensor data from CSV or generate synthetic data if not available.
    
    Returns:
        X_data: Normalized sensor readings (temperature, humidity, pressure, gas_resistance)
        y_data: Labels for data filtering (1 for valid reading, 0 for anomaly)
    """
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Assuming columns: timestamp, temperature, humidity, pressure, gas_resistance, is_valid
            X_data = df[['temperature', 'humidity', 'pressure', 'gas_resistance']].values
            y_data = df[['is_valid', 'is_temperature_valid', 'is_humidity_valid', 
                         'is_pressure_valid', 'is_gas_valid']].values
            
            # Normalize data
            X_mean = X_data.mean(axis=0)
            X_std = X_data.std(axis=0)
            X_data = (X_data - X_mean) / X_std
            
            return X_data, y_data
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
    
    # Generate synthetic data if no data file found
    logger.info("No log data found. Generating synthetic training data...")
    return generate_synthetic_data()

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic sensor data for training."""
    # Define normal ranges for sensors
    temp_range = (15, 35)  # °C
    humidity_range = (30, 70)  # %
    pressure_range = (990, 1030)  # hPa
    gas_range = (500, 1500)  # kOhm
    
    # Generate normal readings
    temperature = np.random.uniform(temp_range[0], temp_range[1], n_samples)
    humidity = np.random.uniform(humidity_range[0], humidity_range[1], n_samples)
    pressure = np.random.uniform(pressure_range[0], pressure_range[1], n_samples)
    gas_resistance = np.random.uniform(gas_range[0], gas_range[1], n_samples)
    
    # Introduce some anomalies
    anomaly_rate = 0.15
    anomaly_indices = np.random.choice(n_samples, int(n_samples * anomaly_rate), replace=False)
    
    # Create larger anomalies for some points
    temperature[anomaly_indices] += np.random.choice([-20, 20], size=len(anomaly_indices))
    humidity[anomaly_indices] += np.random.choice([-40, 40], size=len(anomaly_indices))
    pressure[anomaly_indices] += np.random.choice([-50, 50], size=len(anomaly_indices))
    gas_resistance[anomaly_indices] += np.random.choice([-1000, 1000], size=len(anomaly_indices))
    
    # Stack features
    X_data = np.column_stack((temperature, humidity, pressure, gas_resistance))
    
    # Create labels (0 for anomaly, 1 for normal)
    y_data = np.ones((n_samples, 5))  # All valid by default
    
    # Mark specific anomalies
    temp_anomalies = (temperature < temp_range[0]) | (temperature > temp_range[1])
    humidity_anomalies = (humidity < humidity_range[0]) | (humidity > humidity_range[1])
    pressure_anomalies = (pressure < pressure_range[0]) | (pressure > pressure_range[1])
    gas_anomalies = (gas_resistance < gas_range[0]) | (gas_resistance > gas_range[1])
    
    # Set label for overall validity and individual sensor validity
    any_anomaly = temp_anomalies | humidity_anomalies | pressure_anomalies | gas_anomalies
    y_data[any_anomaly, 0] = 0  # Overall validity
    y_data[temp_anomalies, 1] = 0  # Temperature validity
    y_data[humidity_anomalies, 2] = 0  # Humidity validity  
    y_data[pressure_anomalies, 3] = 0  # Pressure validity
    y_data[gas_anomalies, 4] = 0  # Gas validity
    
    # Normalize data
    X_mean = X_data.mean(axis=0)
    X_std = X_data.std(axis=0)
    X_data = (X_data - X_mean) / X_std
    
    return X_data, y_data

def train_model(X_data, y_data, epochs=500, batch_size=64, learning_rate=0.001):
    """Train the sensor data filtering model."""
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_data)
    y_tensor = torch.FloatTensor(y_data)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SensorModel(input_size=X_data.shape[1], output_size=y_data.shape[1])
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training data shape: X={X_data.shape}, y={y_data.shape}")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
    
    print("Training completed")
    return model

def export_to_onnx(model, input_shape, onnx_path):
    """Export PyTorch model to ONNX format."""
    # Create dummy input
    dummy_input = torch.randn(1, input_shape, requires_grad=True)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX model exported to {onnx_path}")
    return onnx_path

def build_tensorrt_engine(onnx_path):
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
    
    config = builder.create_builder_config()
    # Updated API for TensorRT 8.x+
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # New TensorRT API (8.x+)
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Save engine to file
    engine_path = onnx_path.replace('.onnx', '.engine')
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine exported to {engine_path}")
    return engine_path

def generate_sample_neo4j_data():
    """Generate sample sensor readings for Neo4j import."""
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    readings = []
    current_date = start_date
    
    # Define normal ranges
    temp_range = (15, 35)
    humidity_range = (30, 70)
    pressure_range = (990, 1030)
    gas_range = (500, 1500)
    
    # Generate hourly readings
    while current_date <= end_date:
        # Normal values with some random variation
        temp = random.uniform(temp_range[0], temp_range[1])
        humidity = random.uniform(humidity_range[0], humidity_range[1])
        pressure = random.uniform(pressure_range[0], pressure_range[1])
        gas = random.uniform(gas_range[0], gas_range[1])
        
        # Add some anomalies (about 5% of readings)
        if random.random() < 0.05:
            # Choose one parameter to make anomalous
            anomaly_param = random.choice(['temp', 'humidity', 'pressure', 'gas'])
            if anomaly_param == 'temp':
                temp += random.choice([-20, 20])
            elif anomaly_param == 'humidity':
                humidity += random.choice([-40, 40])
            elif anomaly_param == 'pressure':
                pressure += random.choice([-50, 50])
            else:
                gas += random.choice([-1000, 1000])
        
        timestamp = int(current_date.timestamp() * 1000)  # milliseconds for Neo4j
        
        readings.append({
            'timestamp': timestamp,
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'pressure': round(pressure, 2),
            'gas': round(gas, 2)
        })
        
        # Next hour
        current_date += timedelta(hours=1)
    
    # Save to CSV for easy import to Neo4j
    df = pd.DataFrame(readings)
    csv_path = 'data/sensor_readings.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Sample data generated and saved to {csv_path}")
    
    # Generate Neo4j Cypher import script
    cypher_path = 'data/import_to_neo4j.cypher'
    with open(cypher_path, 'w') as f:
        f.write("""
// Create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (s:SensorReading) REQUIRE s.timestamp IS UNIQUE;

// Load CSV data
LOAD CSV WITH HEADERS FROM 'file:///sensor_readings.csv' AS row
CREATE (s:SensorReading {
  timestamp: toInteger(row.timestamp),
  temperature: toFloat(row.temperature),
  humidity: toFloat(row.humidity),
  pressure: toFloat(row.pressure),
  gas: toFloat(row.gas)
});

// Index for time range queries
CREATE INDEX ON :SensorReading(timestamp);
        """)
    
    print(f"Neo4j import script generated at {cypher_path}")
    
    # Generate example Grafana queries
    grafana_path = 'data/grafana_example_queries.txt'
    with open(grafana_path, 'w') as f:
        f.write("""
# Basic time series query
MATCH (sr:SensorReading)
WHERE sr.timestamp >= $timeFrom AND sr.timestamp <= $timeTo
RETURN sr.timestamp as time, sr.temperature as temp, sr.humidity as hum, sr.pressure as press, sr.gas as gas_res
ORDER BY sr.timestamp ASC

# Hourly aggregation query
MATCH (sr:SensorReading)
WHERE sr.timestamp >= $timeFrom AND sr.timestamp <= $timeTo
WITH datetime({epochMillis: sr.timestamp}) as hour, sr
WITH hour.hour as hourOfDay, avg(sr.temperature) as avgTemp, avg(sr.humidity) as avgHum, avg(sr.pressure) as avgPress
RETURN hourOfDay, avgTemp, avgHum, avgPress
ORDER BY hourOfDay

# Anomaly detection query
MATCH (sr:SensorReading)
WHERE sr.timestamp >= $timeFrom AND sr.timestamp <= $timeTo
WITH avg(sr.temperature) as avgTemp, stDev(sr.temperature) as stdTemp, collect(sr) as readings
UNWIND readings as sr
WITH sr, avgTemp, stdTemp
WHERE abs(sr.temperature - avgTemp) > 2 * stdTemp
RETURN sr.timestamp as time, sr.temperature as temp, avgTemp, stdTemp
ORDER BY time
        """)
    
    print(f"Grafana example queries saved to {grafana_path}")

def main():
    """Main function to build and export the model."""
    print("Building TensorRT LLM model for sensor data filtering...")
    
    # Load or generate training data
    X_data, y_data = load_sensor_data()
    
    # Train the model
    model = train_model(X_data, y_data)
    
    # Export to ONNX
    onnx_path = 'models/sensor_model.onnx'
    export_to_onnx(model, X_data.shape[1], onnx_path)
    
    # Build TensorRT engine
    engine_path = build_tensorrt_engine(onnx_path)
    
    # Generate sample Neo4j data and import scripts
    generate_sample_neo4j_data()
    
    print(f"Model processing complete! Engine saved at {engine_path}")
    print("You can now use the generated Neo4j data and Grafana queries for visualization.")

if __name__ == "__main__":
    main()