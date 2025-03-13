#!/usr/bin/env python3
# sensorloader_trt.py - Read BME680 sensor data and process with TensorRT model

import time
import logging
import numpy as np
import json
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Import sensor library
from bme680 import BME680
from smbus2 import SMBus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorRT Engine handling
class TensorRTInference:
    def __init__(self, engine_path):
        logger.info(f"Initializing TensorRT engine from {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load TRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")
            
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Find input and output bindings
        self.input_idx = 0  # Assume first binding is input
        self.output_idx = 1  # Assume second binding is output
        
        # Create GPU buffers and host buffers
        self.input_shape = (1, 4)  # Batch size 1, 4 sensor values
        self.output_shape = (1, 5)  # Batch size 1, 5 output values (validity + 4 filtered values)
        
        # Create GPU buffers - Fix for numpy.int64 vs unsigned long type mismatch
        input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        
        # Create host buffers
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        logger.info("TensorRT engine initialized successfully")
    
    def infer(self, sensor_data):
        """Run inference on sensor data"""
        # Normalize input data
        normalized_data = self.normalize_data(sensor_data)
        
        # Copy to input buffer
        np.copyto(self.h_input[0], normalized_data)
        
        # Copy input data to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        bindings = [int(self.d_input), int(self.d_output)]
        
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy results back to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        return self.h_output[0]
    
    def normalize_data(self, data):
        """Normalize sensor data to model input range"""
        # Extract and normalize each sensor reading
        # These normalization values should match those used during training
        normalized = np.array([
            data['temperature'] / 100.0,  # Scale temperature
            data['humidity'] / 100.0,     # Scale humidity
            data['pressure'] / 1100.0,    # Scale pressure
            np.log10(max(data['gas_resistance'], 1)) / 5.0  # Log scale for gas resistance
        ], dtype=np.float32)
        
        return normalized
    
    def interpret_results(self, results):
        """Interpret model output"""
        validity_score = results[0]
        filtered_values = results[1:]
        
        is_valid = validity_score > 0.5
        
        return {
            'validity_score': float(validity_score),
            'is_valid': bool(is_valid),
            'filtered_temperature': float(filtered_values[0] * 100.0),
            'filtered_humidity': float(filtered_values[1] * 100.0),
            'filtered_pressure': float(filtered_values[2] * 1100.0),
            'filtered_gas_resistance': float(10 ** (filtered_values[3] * 5.0))
        }

def initialize_sensor():
    """Initialize and configure the BME680 sensor"""
    try:
        # Create an SMBus instance for I2C bus 7
        i2c_bus = SMBus(7)
        
        # Initialize BME680 with the specified bus
        sensor = BME680(i2c_device=i2c_bus)
        
        # Configure the sensor with direct values instead of enums
        sensor.set_humidity_oversample(2)  # Instead of BME680.OS_2X
        sensor.set_pressure_oversample(4)  # Instead of BME680.OS_4X
        sensor.set_temperature_oversample(8)  # Instead of BME680.OS_8X
        sensor.set_filter(3)  # Instead of BME680.FILTER_SIZE_3
        sensor.set_gas_status(1)  # Instead of BME680.ENABLE_GAS_MEAS
        
        # Set gas heater parameters for measuring VOCs
        sensor.set_gas_heater_temperature(320)
        sensor.set_gas_heater_duration(150)
        sensor.select_gas_heater_profile(0)
        
        logger.info("BME680 sensor initialized successfully")
        return sensor, False
    except Exception as e:
        logger.warning(f"Failed to initialize BME680 sensor: {e}")
        logger.warning("Running in simulation mode")
        return None, True

def read_sensor_data(sensor, simulation_mode):
    """Read data from BME680 sensor or generate simulated data"""
    # Default to simulation mode if there's any issue
    use_simulation = simulation_mode
    sensor_data = None
    
    if not use_simulation and sensor is not None:
        try:
            # Try to get sensor data
            if sensor.get_sensor_data():
                sensor_data = {
                    'temperature': sensor.data.temperature,
                    'humidity': sensor.data.humidity,
                    'pressure': sensor.data.pressure,
                    'gas_resistance': sensor.data.gas_resistance if sensor.data.heat_stable else 0
                }
                logger.debug(f"Read sensor data: {sensor_data}")
            else:
                logger.warning("Sensor returned no data, using simulation")
                use_simulation = True
        except Exception as e:
            logger.error(f"Error reading sensor: {e}")
            # Fall back to simulation if reading fails
            use_simulation = True
    else:
        use_simulation = True
    
    # Generate simulated data if in simulation mode or reading failed
    if use_simulation or sensor_data is None:
        import random
        sensor_data = {
            'temperature': round(random.uniform(18, 28), 2),
            'humidity': round(random.uniform(40, 70), 2),
            'pressure': round(random.uniform(990, 1030), 2),
            'gas_resistance': round(random.uniform(5000, 15000), 2)
        }
        logger.debug(f"Generated simulated data: {sensor_data}")
    
    # Ensure we're returning valid data
    if sensor_data is None:
        logger.error("Failed to get sensor data or generate simulation data")
        # Provide default values as a last resort
        sensor_data = {
            'temperature': 22.0,
            'humidity': 50.0,
            'pressure': 1000.0,
            'gas_resistance': 10000.0
        }
    
    return sensor_data

def write_to_neo4j_csv(data, filename='data/live_readings.csv'):
    """Write sensor readings to CSV file for Neo4j import"""
    import csv
    import os
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to determine if header needs to be written
    file_exists = os.path.isfile(filename)
    
    # Current timestamp in milliseconds
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Open CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'temperature', 'humidity', 'pressure', 'gas', 'validity_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            'timestamp': timestamp,
            'temperature': data['raw']['temperature'],
            'humidity': data['raw']['humidity'],
            'pressure': data['raw']['pressure'],
            'gas': data['raw']['gas_resistance'],
            'validity_score': data['filtered']['validity_score']
        })  # Fixed closing bracket

def main():
    # Initialize sensor
    sensor, simulation_mode = initialize_sensor()
    
    # Initialize TensorRT model
    model_path = 'models/sensor_model.engine'
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please run build_sensor_model.py first to generate the model")
        return
    
    trt_model = TensorRTInference(model_path)
    
    # Create data output directory
    os.makedirs('data', exist_ok=True)
    
    try:
        logger.info("Starting sensor data collection")
        while True:
            # Read sensor data
            raw_data = read_sensor_data(sensor, simulation_mode)
            
            if raw_data is not None:
                # Run inference
                result = trt_model.infer(raw_data)
                
                # Interpret results
                interpreted = trt_model.interpret_results(result)
                
                # Combine raw and filtered data
                combined_data = {
                    'timestamp': int(time.time() * 1000),
                    'raw': raw_data,
                    'filtered': interpreted,
                    'status': 'normal' if interpreted['is_valid'] else 'anomaly'
                }
                
                # Print status
                logger.info(f"Status: {combined_data['status']} - " +
                          f"Temp: {raw_data['temperature']:.1f}°C, " +
                          f"Humidity: {raw_data['humidity']:.1f}%, " +
                          f"Pressure: {raw_data['pressure']:.1f}hPa, " +
                          f"Gas: {raw_data['gas_resistance']:.0f}Ω, " +
                          f"Validity: {interpreted['validity_score']:.2f}")
                
                # Save data for Neo4j import
                write_to_neo4j_csv(combined_data)
            else:
                logger.error("Failed to get sensor data, skipping this reading")
            
            # Wait before next reading
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Sensor monitoring stopped by user")
    finally:
        if not simulation_mode and sensor is not None:
            # Properly close I2C bus if needed
            try:
                sensor._i2c.close()
            except:
                pass
        logger.info("Sensor monitoring finished")

if __name__ == "__main__":
    main()
