#!/usr/bin/env python3
# chat_with_sensors.py - Interact with sensor data in Neo4j using natural language

import json
import os
import time
from datetime import datetime, timedelta
import re
import logging
import argparse
from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jConnector:
    """Connector for Neo4j database with sensor data"""
    
    def __init__(self, uri, user, password):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
        
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        
    def run_query(self, query, params=None):
        """Run a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record for record in result]

class SensorChatbot:
    """Chatbot for interacting with sensor data in Neo4j"""
    
    def __init__(self, config_path="config.json"):
        """Initialize the sensor chatbot"""
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Connect to Neo4j
        neo4j_config = self.config["neo4j"]
        self.neo4j = Neo4jConnector(
            neo4j_config["uri"],
            neo4j_config["user"],
            neo4j_config["password"]
        )
        
        # Initialize query templates
        self.query_templates = {
            "latest": """
                MATCH (s:SensorReading)
                RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
                ORDER BY s.timestamp DESC
                LIMIT {limit}
            """,
            "average": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time}
                RETURN 
                    avg(s.temperature) as avg_temp,
                    avg(s.humidity) as avg_humidity,
                    avg(s.pressure) as avg_pressure,
                    avg(s.gas) as avg_gas
            """,
            "anomalies": """
                MATCH (s:SensorReading)
                WHERE s.data_quality = 'filtered'
                RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
                ORDER BY s.timestamp DESC
                LIMIT {limit}
            """,
            "time_range": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time}
                RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
                ORDER BY s.timestamp ASC
            """,
            "max_values": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time}
                RETURN 
                    max(s.temperature) as max_temp,
                    max(s.humidity) as max_humidity,
                    max(s.pressure) as max_pressure,
                    max(s.gas) as max_gas
            """,
            "min_values": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time}
                RETURN 
                    min(s.temperature) as min_temp,
                    min(s.humidity) as min_humidity,
                    min(s.pressure) as min_pressure,
                    min(s.gas) as min_gas
            """,
            "count": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time}
                RETURN count(s) as reading_count
            """
        }
