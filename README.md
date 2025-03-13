# Sensor Analytics with TensorRT LLM

Intelligent sensor data analytics using TensorRT LLM for filtering and anomaly detection before storing in Neo4j.

## Overview

This project demonstrates how to use NVIDIA's TensorRT LLM to process sensor data from a BME680 environmental sensor connected to a Jetson Nano, filter out anomalies and noise, and store the clean data in a Neo4j graph database.

<img width="1165" alt="image" src="https://github.com/user-attachments/assets/eb9f676b-d05c-4e75-b2e0-9705cb4a9abd" />


## How it works?


## Data Collection:

- The BME680 sensor provides raw environmental readings
- The `sensorloader_trt.py` script collects this data
- A simulation mode serves as a fallback when physical sensors aren't available


## Model Training:

- The `build_sensor_model.py` script either uses synthetic or historical data
- It trains a feed-forward neural network for filtering and anomaly detection
- The trained model is exported to ONNX format and compiled into a TensorRT engine

## Data Filtering:

- `sensor_filter.py` implements two filtering approaches:
- TensorRT model-based filtering (primary method)
- Statistical z-score based filtering (fallback method)

The filtering process adds contextual information and derived metrics


## Neo4j Integration:

- Filtered sensor data is stored as nodes in Neo4j
- Time-based and pattern relationships connect these nodes
- The database is queried by your AI assistant and visualized in Grafana


The flow of data is clearly shown with arrows, and the diagram highlights both the main processing path and fallback methods in case of component failure. This architecture demonstrates how you've built a robust system for sensor data processing with intelligent filtering using TensorRT.


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

- NVIDIA Jetson Orin Nano Super (8GB model)
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

```
python sensorloader_trt.py
2025-03-13 12:40:28,467 - INFO - BME680 sensor initialized successfully
2025-03-13 12:40:28,468 - INFO - Initializing TensorRT engine from models/sensor_model.engine
2025-03-13 12:40:28,519 - INFO - Found input tensor: input with shape (-1, 4) and dtype DataType.FLOAT
2025-03-13 12:40:28,520 - INFO - Found output tensor: output with shape (-1, 5) and dtype DataType.FLOAT
2025-03-13 12:40:28,520 - INFO - TensorRT engine initialized successfully
2025-03-13 12:40:28,520 - INFO - Starting sensor data collection
2025-03-13 12:40:28,624 - WARNING - Sensor returned no data, using simulation
2025-03-13 12:40:28,627 - INFO - Status: normal - Temp: 22.9°C, Humidity: 63.5%, Pressure: 1026.3hPa, Gas: 5325Ω, Validity: 1.00
2025-03-13 12:40:33,637 - INFO - Status: normal - Temp: 34.3°C, Humidity: 37.5%, Pressure: 921.9hPa, Gas: 0Ω, Validity: 1.00
2025-03-13 12:40:38,645 - INFO - Status: normal - Temp: 34.3°C, Humidity: 37.4%, Pressure: 921.9hPa, Gas: 79817Ω, Validity: 1.00
2025-03-13 12:40:43,654 - INFO - Status: normal - Temp: 34.3°C, Humidity: 37.3%, Pressure: 921.9hPa, Gas: 82816Ω, Validity: 1.00
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

## Using MCP and Prompts


Data Exploration Prompts:

- "Analyze the distribution of sensor readings across different classifications"
- "Find patterns in sensor data that reveal unique time-based insights"
- "Discover correlations between sensor readings and their classifications"


## Advanced Analysis Prompts:


- "Create a visualization that shows the most interesting sensor readings"
- "Identify any anomalous sensor readings that deviate from typical patterns"
- "Generate a time-based heatmap of sensor data variations"

## Complex Query Challenges:


- "Find the top 3 most frequent sensor reading classifications"
- "Develop a query that tracks sensor reading trends over different time groups"
- "Explore relationships between sensor readings and their associated classifications"
