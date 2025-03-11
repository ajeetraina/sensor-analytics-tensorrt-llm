# Sensor Analytics with TensorRT LLM

Intelligent sensor data analytics using TensorRT LLM for filtering and anomaly detection before storing in Neo4j.

## Overview

This project demonstrates how to use NVIDIA's TensorRT LLM to process sensor data from a BME680 environmental sensor connected to a Jetson Nano, filter out anomalies and noise, and store the clean data in a Neo4j graph database.

## Features

- **Intelligent filtering** using TensorRT LLM to detect and correct anomalous sensor readings
- **Anomaly detection** to identify unusual patterns in sensor data
- **Data enrichment** with additional context and derived metrics
- **Batch processing** for more efficient Neo4j database operations
- **Auto-fallback** to statistical methods when the TensorRT engine is unavailable

## Components

- `sensor_filter.py`: Core filtering module using TensorRT LLM
- `sensorloader_trt.py`: Sensor data collection with TensorRT filtering
- `build_sensor_model.py`: Script for building and training the TensorRT LLM model
- `config.json`: Configuration for Neo4j and TensorRT settings
- `setup.py`: Helper script for dependency checks and setup

## Architecture

```
+-------------+     +----------------+     +--------------------+     +---------+
| BME680      |     | TensorRT LLM   |     | Data Enrichment &  |     | Neo4j   |
| Sensor      | --> | Filtering &    | --> | Batch Processing   | --> | Database|
| (Jetson)    |     | Anomaly Det.   |     |                    |     |         |
+-------------+     +----------------+     +--------------------+     +---------+
```

## Prerequisites

- NVIDIA Jetson Nano (2GB or 4GB model)
- BME680 environmental sensor
- Neo4j Database (cloud or local)
- NVIDIA TensorRT (installed via JetPack)
- PyTorch
- ONNX Runtime

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/ajeetraina/sensor-analytics-tensorrt-llm.git
   cd sensor-analytics-tensorrt-llm
   ```

2. Install dependencies:
   ```
   python setup.py --check
   ```

3. Configure the Neo4j connection:
   ```
   python setup.py --configure
   ```

4. Build the TensorRT model:
   ```
   python build_sensor_model.py
   ```

   ```
   python build_sensor_model.py
Building TensorRT LLM model for sensor data filtering...
2025-03-11 13:36:08,335 - INFO - No log data found. Generating synthetic training data...
Training data shape: X=(5000, 4), y=(5000, 5)
Epoch 50/500, Loss: 0.000020
Epoch 100/500, Loss: 0.000002
Epoch 150/500, Loss: 0.000000
Epoch 200/500, Loss: 0.000000
Epoch 250/500, Loss: 0.000000
Epoch 300/500, Loss: 0.000000
Epoch 350/500, Loss: 0.000000
Epoch 400/500, Loss: 0.000000
Epoch 450/500, Loss: 0.000000
Epoch 500/500, Loss: 0.000000
Training completed
ONNX model exported to models/sensor_model.onnx
[03/11/2025-13:38:34] [TRT] [W] DLA requests all profiles have same min, max, and opt value. All dla layers are falling back to GPU
TensorRT engine exported to models/sensor_model.engine
Sample data generated and saved to data/sensor_readings.csv
Neo4j import script generated at data/import_to_neo4j.cypher
Grafana example queries saved to data/grafana_example_queries.txt
Model processing complete! Engine saved at models/sensor_model.engine
You can now use the generated Neo4j data and Grafana queries for visualization.
```

5. Run the sensor data collection:
   ```
   python sensorloader_trt.py
   ```

## Example Neo4j Queries

```cypher
// Get latest sensor readings
MATCH (s:SensorReading)
RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
ORDER BY s.timestamp DESC
LIMIT 10

// Find anomalous readings
MATCH (s:SensorReading)
WHERE s.data_quality = 'filtered'
RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
ORDER BY s.timestamp DESC
LIMIT 10
```

## License

MIT License
