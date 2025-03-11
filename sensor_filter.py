import numpy as np
import tensorrt as trt
import torch
import json
import os
import time
from datetime import datetime
from collections import deque

class SensorFilter:
    def __init__(self, model_path=None, window_size=10, threshold=0.8):
        """
        Initialize the sensor filter with TensorRT LLM model
        
        Args:
            model_path: Path to TensorRT engine file
            window_size: Number of readings to maintain for outlier detection
            threshold: Threshold for anomaly detection
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Buffers for rolling window of sensor values
        self.temp_buffer = deque(maxlen=window_size)
        self.humidity_buffer = deque(maxlen=window_size)
        self.pressure_buffer = deque(maxlen=window_size)
        self.gas_buffer = deque(maxlen=window_size)
        
        # Metadata for TensorRT LLM model
        self.model = None
        self.model_path = model_path
        
        # Initialize TensorRT engine if path is provided
        if model_path and os.path.exists(model_path):
            self._init_tensorrt_engine()
    
    def _init_tensorrt_engine(self):
        """Initialize TensorRT engine for inference"""
        try:
            # Logger for TensorRT
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create runtime and load engine
            runtime = trt.Runtime(logger)
            with open(self.model_path, 'rb') as f:
                engine_bytes = f.read()
            
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()
            
            # Get input and output binding shapes
            self.input_shape = self.engine.get_binding_shape(0)
            self.output_shape = self.engine.get_binding_shape(1)
            
            # Allocate device memory
            self.d_input = torch.zeros(tuple(self.input_shape), dtype=torch.float32, device='cuda')
            self.d_output = torch.zeros(tuple(self.output_shape), dtype=torch.float32, device='cuda')
            
            print(f"TensorRT LLM engine loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"Failed to initialize TensorRT engine: {e}")
            # Fallback to statistical filtering if TensorRT initialization fails
            self.engine = None
    
    def _statistical_filter(self, new_value, buffer):
        """
        Filter based on statistical methods (used when TensorRT model is not available)
        
        Args:
            new_value: New sensor reading
            buffer: Buffer of historical values
            
        Returns:
            Tuple of (is_valid, filtered_value)
        """
        if len(buffer) < 3:  # Not enough data for filtering
            return True, new_value
        
        # Calculate z-score
        mean = np.mean(list(buffer))
        std = max(np.std(list(buffer)), 0.0001)  # Avoid division by zero
        z_score = abs((new_value - mean) / std)
        
        # Check if value is an outlier (z-score > 3)
        if z_score > 3:
            # Replace with moving average
            return False, mean
        
        return True, new_value
    
    def _tensorrt_filter(self, sensor_data):
        """
        Use TensorRT LLM model to filter sensor data
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Tuple of (is_valid, filtered_data)
        """
        try:
            # Prepare input data - normalize values based on typical ranges
            input_data = np.array([
                sensor_data['temperature'] / 100.0,  # Normalize to 0-1 range
                sensor_data['humidity'] / 100.0,
                sensor_data['pressure'] / 1100.0,
                np.log10(max(sensor_data['gas'], 1)) / 5.0  # Log scale for gas resistance
            ], dtype=np.float32).reshape(1, 4)
            
            # Copy to GPU
            self.d_input.copy_(torch.from_numpy(input_data))
            
            # Run inference
            self.context.execute_v2(bindings=[
                int(self.d_input.data_ptr()),
                int(self.d_output.data_ptr())
            ])
            
            # Get output
            output = self.d_output.cpu().numpy()
            
            # Process output
            # Output is [validity_score, filtered_temp, filtered_humidity, filtered_pressure, filtered_gas]
            validity_score = output[0][0]
            
            if validity_score < self.threshold:
                # Data seems invalid, apply corrections
                filtered_data = {
                    'temperature': output[0][1] * 100.0,
                    'humidity': output[0][2] * 100.0,
                    'pressure': output[0][3] * 1100.0,
                    'gas': 10 ** (output[0][4] * 5.0)
                }
                return False, filtered_data
            
            return True, sensor_data
        
        except Exception as e:
            print(f"TensorRT inference failed: {e}")
            # Fallback to statistical filtering
            return self._apply_statistical_filter(sensor_data)
    
    def _apply_statistical_filter(self, sensor_data):
        """Apply statistical filtering to each sensor value"""
        valid_temp, filtered_temp = self._statistical_filter(
            sensor_data['temperature'], self.temp_buffer
        )
        valid_humidity, filtered_humidity = self._statistical_filter(
            sensor_data['humidity'], self.humidity_buffer
        )
        valid_pressure, filtered_pressure = self._statistical_filter(
            sensor_data['pressure'], self.pressure_buffer
        )
        valid_gas, filtered_gas = self._statistical_filter(
            sensor_data['gas'], self.gas_buffer
        )
        
        # If any value is invalid, apply filtering
        is_valid = valid_temp and valid_humidity and valid_pressure and valid_gas
        
        filtered_data = {
            'temperature': filtered_temp,
            'humidity': filtered_humidity,
            'pressure': filtered_pressure,
            'gas': filtered_gas
        }
        
        return is_valid, filtered_data
    
    def filter_sensor_data(self, sensor_data):
        """
        Main method to filter sensor data
        
        Args:
            sensor_data: Dictionary with temp, humidity, pressure, gas values
            
        Returns:
            Tuple of (is_data_valid, filtered_data)
        """
        # Update buffers with new values (whether filtered or not)
        self.temp_buffer.append(sensor_data['temperature'])
        self.humidity_buffer.append(sensor_data['humidity'])
        self.pressure_buffer.append(sensor_data['pressure'])
        self.gas_buffer.append(sensor_data['gas'])
        
        # Apply appropriate filtering method
        if self.engine is not None:
            return self._tensorrt_filter(sensor_data)
        else:
            return self._apply_statistical_filter(sensor_data)
    
    def add_context(self, sensor_data):
        """
        Add additional context to sensor data before storing in Neo4j
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Enriched sensor data with additional context
        """
        # Current timestamp
        timestamp = int(sensor_data.get('timestamp', time.time()))
        dt = datetime.fromtimestamp(timestamp)
        
        # Add time context
        sensor_data['hour_of_day'] = dt.hour
        sensor_data['day_of_week'] = dt.weekday()
        sensor_data['month'] = dt.month
        
        # Add derived metrics
        if 'temperature' in sensor_data and 'humidity' in sensor_data:
            # Calculate heat index
            t = sensor_data['temperature']
            h = sensor_data['humidity']
            
            # Simple heat index calculation
            if t > 26.7 and h > 40:
                heat_index = -8.784695 + 1.61139411 * t + 2.338549 * h - \
                             0.14611605 * t * h - 0.012308094 * t**2 - \
                             0.016424828 * h**2 + 0.002211732 * t**2 * h + \
                             0.00072546 * t * h**2 - 0.000003582 * t**2 * h**2
                sensor_data['heat_index'] = round(heat_index, 2)
        
        # Add data quality metrics
        is_valid, _ = self._apply_statistical_filter(sensor_data)
        sensor_data['data_quality'] = 'normal' if is_valid else 'filtered'
        
        return sensor_data

    def get_anomaly_report(self):
        """Generate a summary report of anomalies detected"""
        if len(self.temp_buffer) < self.window_size:
            return {"status": "Collecting initial data", "anomalies_detected": 0}
        
        # Calculate statistics
        anomaly_count = sum(1 for v in self.temp_buffer if abs((v - np.mean(self.temp_buffer)) / max(np.std(self.temp_buffer), 0.0001)) > 3)
        anomaly_count += sum(1 for v in self.humidity_buffer if abs((v - np.mean(self.humidity_buffer)) / max(np.std(self.humidity_buffer), 0.0001)) > 3)
        anomaly_count += sum(1 for v in self.pressure_buffer if abs((v - np.mean(self.pressure_buffer)) / max(np.std(self.pressure_buffer), 0.0001)) > 3)
        anomaly_count += sum(1 for v in self.gas_buffer if abs((v - np.mean(self.gas_buffer)) / max(np.std(self.gas_buffer), 0.0001)) > 3)
        
        return {
            "status": "Active",
            "anomalies_detected": anomaly_count,
            "temp_mean": np.mean(self.temp_buffer),
            "humidity_mean": np.mean(self.humidity_buffer),
            "pressure_mean": np.mean(self.pressure_buffer),
            "gas_mean": np.mean(self.gas_buffer)
        }

# Example usage
if __name__ == "__main__":
    # Create filter
    filter = SensorFilter(model_path="models/sensor_model.engine", window_size=20)
    
    # Test with sample data
    test_data = {
        'temperature': 25.4,
        'humidity': 65.2,
        'pressure': 1013.2,
        'gas': 12000
    }
    
    is_valid, filtered_data = filter.filter_sensor_data(test_data)
    print(f"Data valid: {is_valid}")
    print(f"Filtered data: {filtered_data}")
    
    # Add context
    enriched_data = filter.add_context(filtered_data)
    print(f"Enriched data: {enriched_data}")