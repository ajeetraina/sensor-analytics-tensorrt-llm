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