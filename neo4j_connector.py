#!/usr/bin/env python3
# neo4j_connector.py - Connect to Neo4j and query sensor data

import logging
from neo4j import GraphDatabase
from datetime import datetime, timedelta

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
        logger.info("Neo4j connection closed")
        
    def run_query(self, query, params=None):
        """Run a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record for record in result]
    
    def get_latest_readings(self, limit=5):
        """Get the latest sensor readings"""
        query = """
            MATCH (s:SensorReading)
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp DESC
            LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})
    
    def get_average_readings(self, start_time=None, end_time=None):
        """Get average sensor readings for a time period"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN 
                avg(s.temperature) as avg_temp,
                avg(s.humidity) as avg_humidity,
                avg(s.pressure) as avg_pressure,
                avg(s.gas) as avg_gas
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
    
    def get_anomalies(self, limit=5):
        """Get anomalous sensor readings"""
        query = """
            MATCH (s:SensorReading)
            WHERE s.data_quality = 'filtered'
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp DESC
            LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})
    
    def get_readings_in_range(self, start_time=None, end_time=None):
        """Get sensor readings for a specific time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp ASC
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
    
    def get_max_values(self, start_time=None, end_time=None):
        """Get maximum sensor values for a time period"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN 
                max(s.temperature) as max_temp,
                max(s.humidity) as max_humidity,
                max(s.pressure) as max_pressure,
                max(s.gas) as max_gas
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
    
    def get_min_values(self, start_time=None, end_time=None):
        """Get minimum sensor values for a time period"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN 
                min(s.temperature) as min_temp,
                min(s.humidity) as min_humidity,
                min(s.pressure) as min_pressure,
                min(s.gas) as min_gas
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
    
    def count_readings(self, start_time=None, end_time=None):
        """Count sensor readings for a time period"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN count(s) as reading_count
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
    
    def get_sensor_metadata(self):
        """Get metadata about sensor readings in the database"""
        query = """
            MATCH (s:SensorReading)
            RETURN 
                count(s) as total_readings,
                min(s.timestamp) as first_reading,
                max(s.timestamp) as last_reading,
                avg(s.temperature) as avg_temp,
                avg(s.humidity) as avg_humidity
        """
        return self.run_query(query)
