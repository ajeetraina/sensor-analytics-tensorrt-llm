#!/usr/bin/env python3
# neo4j_connector.py - Neo4j database connector for sensor data

import logging
from neo4j import GraphDatabase

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
            
    def get_latest_readings(self, limit=5):
        """Get latest sensor readings"""
        query = """
            MATCH (s:SensorReading)
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp DESC
            LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})
        
    def get_anomalies(self, limit=5):
        """Get anomalous readings"""
        query = """
            MATCH (s:SensorReading)
            WHERE s.data_quality = 'filtered'
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp DESC
            LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})
        
    def get_average_readings(self, start_time, end_time):
        """Get average readings for time period"""
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
        
    def get_time_range_readings(self, start_time, end_time):
        """Get readings for time period"""
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas
            ORDER BY s.timestamp ASC
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
        
    def get_min_max_readings(self, start_time, end_time):
        """Get min and max readings for time period"""
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN 
                min(s.temperature) as min_temp,
                max(s.temperature) as max_temp,
                min(s.humidity) as min_humidity,
                max(s.humidity) as max_humidity,
                min(s.pressure) as min_pressure,
                max(s.pressure) as max_pressure,
                min(s.gas) as min_gas,
                max(s.gas) as max_gas
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
        
    def get_reading_count(self, start_time, end_time):
        """Get count of readings for time period"""
        query = """
            MATCH (s:SensorReading)
            WHERE s.timestamp >= $start_time AND s.timestamp <= $end_time
            RETURN count(s) as reading_count
        """
        return self.run_query(query, {"start_time": start_time, "end_time": end_time})
