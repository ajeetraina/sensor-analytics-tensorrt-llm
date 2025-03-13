#!/usr/bin/env python3
# sensor_chat.py - Chat interface for Neo4j sensor data using TensorRT-LLM

import os
import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from neo4j import GraphDatabase
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jSensorChat:
    """Chat interface for Neo4j sensor data"""
    
    def __init__(self, config_path="config.json"):
        """Initialize the sensor chat interface"""
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Connect to Neo4j
        neo4j_config = self.config["neo4j"]
        self.neo4j_uri = neo4j_config["uri"]
        self.neo4j_user = neo4j_config["user"]
        self.neo4j_password = neo4j_config["password"]
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
        
        # Define common query patterns
        self.query_patterns = {
            "latest": self._get_latest_readings_query,
            "average": self._get_average_readings_query,
            "anomalies": self._get_anomalies_query,
            "range": self._get_readings_in_range_query,
            "max": self._get_max_values_query,
            "min": self._get_min_values_query,
            "count": self._get_count_query
        }
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def run_query(self, query, params=None):
        """Execute a Cypher query against Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            return []
    
    def _get_latest_readings_query(self, limit=5):
        """Generate query for latest readings"""
        return (
            "MATCH (s:SensorReading) "
            "RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas "
            "ORDER BY s.timestamp DESC "
            f"LIMIT {limit}"
        )
    
    def _get_average_readings_query(self, start_time=None, end_time=None):
        """Generate query for average readings in a time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        return (
            "MATCH (s:SensorReading) "
            f"WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time} "
            "RETURN "
            "    avg(s.temperature) as avg_temp, "
            "    avg(s.humidity) as avg_humidity, "
            "    avg(s.pressure) as avg_pressure, "
            "    avg(s.gas) as avg_gas"
        )
    
    def _get_anomalies_query(self, limit=5):
        """Generate query for anomalous readings"""
        return (
            "MATCH (s:SensorReading) "
            "WHERE s.data_quality = 'filtered' "
            "RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas "
            "ORDER BY s.timestamp DESC "
            f"LIMIT {limit}"
        )
    
    def _get_readings_in_range_query(self, start_time=None, end_time=None):
        """Generate query for readings in a time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        return (
            "MATCH (s:SensorReading) "
            f"WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time} "
            "RETURN s.timestamp, s.temperature, s.humidity, s.pressure, s.gas "
            "ORDER BY s.timestamp ASC"
        )
    
    def _get_max_values_query(self, start_time=None, end_time=None):
        """Generate query for maximum values in a time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        return (
            "MATCH (s:SensorReading) "
            f"WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time} "
            "RETURN "
            "    max(s.temperature) as max_temp, "
            "    max(s.humidity) as max_humidity, "
            "    max(s.pressure) as max_pressure, "
            "    max(s.gas) as max_gas"
        )
    
    def _get_min_values_query(self, start_time=None, end_time=None):
        """Generate query for minimum values in a time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        return (
            "MATCH (s:SensorReading) "
            f"WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time} "
            "RETURN "
            "    min(s.temperature) as min_temp, "
            "    min(s.humidity) as min_humidity, "
            "    min(s.pressure) as min_pressure, "
            "    min(s.gas) as min_gas"
        )
    
    def _get_count_query(self, start_time=None, end_time=None):
        """Generate query to count readings in a time range"""
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)
            
        return (
            "MATCH (s:SensorReading) "
            f"WHERE s.timestamp >= {start_time} AND s.timestamp <= {end_time} "
            "RETURN count(s) as reading_count"
        )
    
    def get_latest_readings(self, limit=5):
        """Get the latest sensor readings"""
        query = self._get_latest_readings_query(limit)
        return self.run_query(query)
    
    def get_average_readings(self, start_time=None, end_time=None):
        """Get average sensor readings for a time period"""
        query = self._get_average_readings_query(start_time, end_time)
        return self.run_query(query)
    
    def get_anomalies(self, limit=5):
        """Get anomalous sensor readings"""
        query = self._get_anomalies_query(limit)
        return self.run_query(query)
    
    def get_readings_in_range(self, start_time=None, end_time=None):
        """Get sensor readings for a specific time range"""
        query = self._get_readings_in_range_query(start_time, end_time)
        return self.run_query(query)
    
    def process_natural_language_query(self, question):
        """Process a natural language question about sensor data"""
        # Simple keyword matching for now
        question = question.lower()
        
        # Define time range
        start_time = None
        end_time = None
        
        # Time range extraction
        if "today" in question:
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = int(start_of_day.timestamp() * 1000)
        elif "yesterday" in question:
            start_of_yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = int(start_of_yesterday.timestamp() * 1000)
            end_time = int(end_of_yesterday.timestamp() * 1000)
        elif "this week" in question:
            today = datetime.now()
            start_of_week = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = int(start_of_week.timestamp() * 1000)
        elif "last hour" in question:
            last_hour = datetime.now() - timedelta(hours=1)
            start_time = int(last_hour.timestamp() * 1000)
        
        # Determine query type
        if any(word in question for word in ["latest", "recent", "last"]):
            limit = 5
            # Check for explicit limits
            if "last 10" in question or "last ten" in question:
                limit = 10
            elif "last 20" in question or "last twenty" in question:
                limit = 20
                
            results = self.get_latest_readings(limit)
            return self._format_latest_readings(results)
            
        elif "average" in question or "mean" in question:
            results = self.get_average_readings(start_time, end_time)
            return self._format_average_readings(results)
            
        elif "anomaly" in question or "anomalies" in question or "unusual" in question:
            limit = 5
            if "all" in question:
                limit = 100
            results = self.get_anomalies(limit)
            return self._format_anomalies(results)
            
        elif "maximum" in question or "highest" in question or "max" in question:
            results = self.get_average_readings(start_time, end_time)
            return self._format_max_readings(results)
            
        elif "minimum" in question or "lowest" in question or "min" in question:
            results = self.get_average_readings(start_time, end_time)
            return self._format_min_readings(results)
            
        else:
            # Default to latest readings
            results = self.get_latest_readings(5)
            return self._format_latest_readings(results)
    
    def _format_latest_readings(self, results):
        """Format latest readings results"""
        if not results:
            return "I don't have any sensor readings to show you."
            
        response = "Here are the latest sensor readings:\n\n"
        for i, record in enumerate(results):
            timestamp = datetime.fromtimestamp(record["s.timestamp"] / 1000)
            response += f"Reading {i+1} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}):\n"
            response += f"• Temperature: {record['s.temperature']:.1f}°C\n"
            response += f"• Humidity: {record['s.humidity']:.1f}%\n"
            response += f"• Pressure: {record['s.pressure']:.1f} hPa\n"
            response += f"• Gas Resistance: {record['s.gas']:.0f} Ω\n\n"
            
        return response
    
    def _format_average_readings(self, results):
        """Format average readings results"""
        if not results or not results[0]:
            return "I couldn't calculate average readings for the specified period."
            
        record = results[0]
        response = "Here are the average sensor values for the specified period:\n\n"
        response += f"• Average Temperature: {record['avg_temp']:.1f}°C\n"
        response += f"• Average Humidity: {record['avg_humidity']:.1f}%\n"
        response += f"• Average Pressure: {record['avg_pressure']:.1f} hPa\n"
        response += f"• Average Gas Resistance: {record['avg_gas']:.0f} Ω\n"
        
        return response
    
    def _format_anomalies(self, results):
        """Format anomalies results"""
        if not results:
            return "I didn't find any anomalous readings in the specified period."
            
        if len(results) == 0:
            return "Great news! I didn't detect any anomalies in the sensor data."
            
        response = "I found the following anomalous readings:\n\n"
        for i, record in enumerate(results):
            timestamp = datetime.fromtimestamp(record["s.timestamp"] / 1000)
            response += f"Anomaly {i+1} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}):\n"
            response += f"• Temperature: {record['s.temperature']:.1f}°C\n"
            response += f"• Humidity: {record['s.humidity']:.1f}%\n"
            response += f"• Pressure: {record['s.pressure']:.1f} hPa\n"
            response += f"• Gas Resistance: {record['s.gas']:.0f} Ω\n\n"
            
        return response
    
    def _format_max_readings(self, results):
        """Format maximum readings results"""
        if not results or not results[0]:
            return "I couldn't determine maximum values for the specified period."
            
        record = results[0]
        response = "Here are the maximum sensor values for the specified period:\n\n"
        response += f"• Maximum Temperature: {record['max_temp']:.1f}°C\n"
        response += f"• Maximum Humidity: {record['max_humidity']:.1f}%\n"
        response += f"• Maximum Pressure: {record['max_pressure']:.1f} hPa\n"
        response += f"• Maximum Gas Resistance: {record['max_gas']:.0f} Ω\n"
        
        return response
    
    def _format_min_readings(self, results):
        """Format minimum readings results"""
        if not results or not results[0]:
            return "I couldn't determine minimum values for the specified period."
            
        record = results[0]
        response = "Here are the minimum sensor values for the specified period:\n\n"
        response += f"• Minimum Temperature: {record['min_temp']:.1f}°C\n"
        response += f"• Minimum Humidity: {record['min_humidity']:.1f}%\n"
        response += f"• Minimum Pressure: {record['min_pressure']:.1f} hPa\n"
        response += f"• Minimum Gas Resistance: {record['min_gas']:.0f} Ω\n"
        
        return response
    
    def create_visualization(self, start_time=None, end_time=None, output_path="sensor_visualization.png"):
        """Create a visualization of sensor data"""
        # Get data for visualization
        results = self.get_readings_in_range(start_time, end_time)
        
        if not results:
            return "No data available for visualization."
        
        # Extract data
        timestamps = [datetime.fromtimestamp(r["s.timestamp"] / 1000) for r in results]
        temperatures = [r["s.temperature"] for r in results]
        humidities = [r["s.humidity"] for r in results]
        pressures = [r["s.pressure"] for r in results]
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Temperature
        ax1.plot(timestamps, temperatures, 'r-', label='Temperature')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Sensor Data Visualization')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Humidity
        ax2.plot(timestamps, humidities, 'b-', label='Humidity')
        ax2.set_ylabel('Humidity (%)')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        # Pressure
        ax3.plot(timestamps, pressures, 'g-', label='Pressure')
        ax3.set_ylabel('Pressure (hPa)')
        ax3.set_xlabel('Time')
        ax3.legend(loc='upper right')
        ax3.grid(True)
        
        # Format x-axis
        date_format = DateFormatter('%Y-%m-%d %H:%M')
        ax3.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return f"Visualization created and saved to {output_path}"
    
    def interactive_mode(self):
        """Run an interactive chat session"""
        print("\nSensor Data Chat - Interactive Mode")
        print("----------------------------------")
        print("Ask questions about your sensor data or type 'exit' to quit.")
        print("Examples:")
        print("  - What are the latest sensor readings?")
        print("  - Show me any anomalies from today")
        print("  - What was the average temperature yesterday?")
        print("  - Create a visualization of this week's data")
        
        while True:
            question = input("\nYou: ")
            if question.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye!")
                break
            
            start_time = time.time()
            response = self.process_natural_language_query(question)
            end_time = time.time()
            
            print(f"\nChatbot: {response}")
            print(f"(Response time: {end_time - start_time:.2f} seconds)")

def main():
    parser = argparse.ArgumentParser(description="Chat with Neo4j sensor data")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--query", help="Run a single query instead of interactive mode")
    parser.add_argument("--visualize", action="store_true", help="Create a visualization of the sensor data")
    
    args = parser.parse_args()
    
    try:
        chat = Neo4jSensorChat(args.config)
        
        if args.visualize:
            result = chat.create_visualization()
            print(result)
        elif args.query:
            response = chat.process_natural_language_query(args.query)
            print(response)
        else:
            chat.interactive_mode()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'chat' in locals():
            chat.close()

if __name__ == "__main__":
    main()
