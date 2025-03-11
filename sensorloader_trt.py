from neo4j import GraphDatabase
from bme680 import BME680
import time
import json
import os
from datetime import datetime
from sensor_filter import SensorFilter

# Configuration
CONFIG = {
    "neo4j": {
        "uri": "neo4j+s://41275b2a.databases.neo4j.io",
        "user": "neo4j",
        "password": "YOUR_PASSWORD_HERE"  # Replace with your actual password
    },
    "tensorrt": {
        "model_path": "models/sensor_model.engine",
        "threshold": 0.8
    },
    "sampling": {
        "interval": 5,  # seconds between readings
        "batch_size": 10  # number of readings to batch before sending to Neo4j
    },
    "logging": {
        "file": "sensor_logs.json",
        "level": "INFO"
    }
}

# Set up the Neo4j driver
driver = GraphDatabase.driver(
    CONFIG["neo4j"]["uri"], 
    auth=(CONFIG["neo4j"]["user"], CONFIG["neo4j"]["password"])
)

# Set up the BME680 sensor
sensor = BME680()

# Set up the sensor filter
sensor_filter = SensorFilter(
    model_path=CONFIG["tensorrt"]["model_path"],
    window_size=20,
    threshold=CONFIG["tensorrt"]["threshold"]
)

# Buffer for batching data
data_buffer = []

def create_sensor_reading(tx, readings):
    """
    Create multiple sensor reading nodes in a single transaction
    
    Args:
        tx: Neo4j transaction
        readings: List of sensor reading dictionaries
    """
    for reading in readings:
        tx.run("""
            CREATE (sr:SensorReading {
                temperature: $temperature, 
                humidity: $humidity, 
                pressure: $pressure, 
                gas: $gas, 
                timestamp: $timestamp,
                data_quality: $data_quality,
                hour_of_day: $hour_of_day,
                day_of_week: $day_of_week,
                month: $month
            })
            """,
            temperature=reading.get('temperature'),
            humidity=reading.get('humidity'),
            pressure=reading.get('pressure'),
            gas=reading.get('gas'),
            timestamp=reading.get('timestamp'),
            data_quality=reading.get('data_quality', 'unknown'),
            hour_of_day=reading.get('hour_of_day', 0),
            day_of_week=reading.get('day_of_week', 0),
            month=reading.get('month', 0)
        )
        
        # If we have a heat index, add it
        if 'heat_index' in reading:
            tx.run("""
                MATCH (sr:SensorReading)
                WHERE sr.timestamp = $timestamp
                SET sr.heat_index = $heat_index
                """,
                timestamp=reading.get('timestamp'),
                heat_index=reading.get('heat_index')
            )

def log_data(data, status="normal"):
    """Log data to file for debugging and training purposes"""
    if not CONFIG["logging"]["file"]:
        return
    
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "status": status
        }
        
        with open(CONFIG["logging"]["file"], "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")

def flush_buffer():
    """Send buffered readings to Neo4j"""
    global data_buffer
    
    if not data_buffer:
        return
    
    try:
        with driver.session() as session:
            session.write_transaction(create_sensor_reading, data_buffer)
            print(f"Inserted batch of {len(data_buffer)} sensor readings")
    except Exception as e:
        print(f"Error inserting batch into Neo4j: {e}")
        log_data(data_buffer, status="db_error")
    
    data_buffer = []

def process_sensor_reading():
    """Read sensor, apply filtering, and buffer the data"""
    global data_buffer
    
    if sensor.get_sensor_data():
        # Get raw sensor data
        raw_data = {
            'temperature': round(sensor.data.temperature, 2),
            'humidity': round(sensor.data.humidity, 2),
            'pressure': round(sensor.data.pressure, 2),
            'gas': round(sensor.data.gas_resistance, 2),
            'timestamp': int(time.time())
        }
        
        # Apply TensorRT LLM filtering
        is_valid, filtered_data = sensor_filter.filter_sensor_data(raw_data)
        
        # Add context information
        enriched_data = sensor_filter.add_context(filtered_data)
        
        # Add to buffer
        data_buffer.append(enriched_data)
        
        # Log the reading
        msg = "normal" if is_valid else "filtered"
        print(f"Processed sensor reading ({msg}) - temperature: {enriched_data['temperature']}, "
              f"humidity: {enriched_data['humidity']}, pressure: {enriched_data['pressure']}, "
              f"gas: {enriched_data['gas']}")
        
        # Log for training data collection
        log_data({"raw": raw_data, "filtered": enriched_data}, status=msg)
        
        # If buffer is full, send to Neo4j
        if len(data_buffer) >= CONFIG["sampling"]["batch_size"]:
            flush_buffer()
            
            # Print anomaly report
            print("Anomaly report:", json.dumps(sensor_filter.get_anomaly_report(), indent=2))
    else:
        print("Error reading BME680 sensor data.")

def main():
    """Main function to run the sensor data collection"""
    print(f"Starting sensor data collection with TensorRT LLM filtering")
    print(f"Sampling interval: {CONFIG['sampling']['interval']} seconds")
    print(f"Batch size: {CONFIG['sampling']['batch_size']} readings")
    
    try:
        while True:
            process_sensor_reading()
            time.sleep(CONFIG["sampling"]["interval"])
    except KeyboardInterrupt:
        print("Shutting down...")
        flush_buffer()  # Ensure any remaining data is sent
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()