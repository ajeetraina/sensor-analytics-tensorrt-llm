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
                LIMIT $limit
            """,
            "average": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
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
                LIMIT $limit
            """,
            "time_range": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
                RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
                ORDER BY s.timestamp ASC
            """,
            "max_values": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
                RETURN 
                    max(s.temperature) as max_temp,
                    max(s.humidity) as max_humidity,
                    max(s.pressure) as max_pressure,
                    max(s.gas) as max_gas
            """,
            "min_values": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
                RETURN 
                    min(s.temperature) as min_temp,
                    min(s.humidity) as min_humidity,
                    min(s.pressure) as min_pressure,
                    min(s.gas) as min_gas
            """,
            "count": """
                MATCH (s:SensorReading)
                WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
                RETURN count(s) as reading_count
            """
        }
    
    def generate_cypher_from_question(self, question):
        """Generate Cypher query from natural language question"""
        question = question.lower()
        
        # Default parameters
        params = {
            "limit": 5,
            "start_time": int((datetime.now() - timedelta(days=1)).timestamp() * 1000),
            "end_time": int(datetime.now().timestamp() * 1000)
        }
        
        # Determine time range from question
        if "today" in question:
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            params["start_time"] = int(start_of_day.timestamp() * 1000)
        elif "yesterday" in question:
            start_of_yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            params["start_time"] = int(start_of_yesterday.timestamp() * 1000)
            params["end_time"] = int(end_of_yesterday.timestamp() * 1000)
        elif "this week" in question:
            today = datetime.now()
            start_of_week = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            params["start_time"] = int(start_of_week.timestamp() * 1000)
        elif "last hour" in question:
            last_hour = datetime.now() - timedelta(hours=1)
            params["start_time"] = int(last_hour.timestamp() * 1000)
        
        # Number of results
        if "last" in question and any(num in question for num in ["10", "ten", "5", "five", "15", "fifteen", "20", "twenty"]):
            for num, value in [("10", 10), ("ten", 10), ("5", 5), ("five", 5), ("15", 15), ("fifteen", 15), ("20", 20), ("twenty", 20)]:
                if num in question:
                    params["limit"] = value
                    break
        
        # Match query type based on question keywords
        query_type = None
        if any(keyword in question for keyword in ["latest", "recent", "last"]):
            query_type = "latest"
        elif "average" in question or "mean" in question:
            query_type = "average"
        elif "anomal" in question or "unusual" in question or "abnormal" in question:
            query_type = "anomalies"
        elif "maximum" in question or "highest" in question or "max" in question:
            query_type = "max_values"
        elif "minimum" in question or "lowest" in question or "min" in question:
            query_type = "min_values"
        elif "how many" in question or "count" in question:
            query_type = "count"
        else:
            query_type = "time_range"
        
        # Get the corresponding query template
        query_template = self.query_templates.get(query_type, self.query_templates["latest"])
        
        return query_template, params, query_type
    
    def format_response(self, records, query_type):
        """Format Neo4j response into a readable answer"""
        if not records:
            return "I couldn't find any sensor readings matching your request."
        
        if query_type == "latest":
            response = "Here are the latest sensor readings:\n\n"
            for i, record in enumerate(records):
                timestamp = datetime.fromtimestamp(record["s.timestamp"] / 1000)
                response += f"Reading {i+1} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}):\n"
                response += f"• Temperature: {record['s.temperature']:.1f}°C\n"
                response += f"• Humidity: {record['s.humidity']:.1f}%\n"
                response += f"• Pressure: {record['s.pressure']:.1f} hPa\n"
                response += f"• Gas Resistance: {record['s.gas']:.0f} Ω\n\n"
        
        elif query_type == "average":
            record = records[0]
            response = "Here are the average sensor values for the specified period:\n\n"
            response += f"• Average Temperature: {record['avg_temp']:.1f}°C\n"
            response += f"• Average Humidity: {record['avg_humidity']:.1f}%\n"
            response += f"• Average Pressure: {record['avg_pressure']:.1f} hPa\n"
            response += f"• Average Gas Resistance: {record['avg_gas']:.0f} Ω\n"