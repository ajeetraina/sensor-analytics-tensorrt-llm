#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "torch", 
        "tensorrt", 
        "onnx", 
        "neo4j", 
        "numpy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        install = input("Would you like to install them now? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Dependencies installed.")
        else:
            print("Please install the missing dependencies to continue.")
            sys.exit(1)
    else:
        print("All dependencies are installed.")

def configure_neo4j():
    """Configure Neo4j connection settings"""
    config_file = "config.json"
    
    # Load existing config if it exists
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Create default config
        config = {
            "neo4j": {
                "uri": "neo4j+s://localhost:7687",
                "user": "neo4j",
                "password": "password"
            },
            "tensorrt": {
                "model_path": "models/sensor_model.engine",
                "threshold": 0.75,
                "enable_filtering": True
            },
            "sampling": {
                "interval": 5,
                "batch_size": 10
            },
            "logging": {
                "file": "sensor_logs.json",
                "level": "INFO",
                "enable_console": True
            },
            "features": {
                "anomaly_detection": True,
                "data_enrichment": True,
                "notifications": False
            }
        }
    
    # Get Neo4j connection details
    print("\nNeo4j Configuration")
    print("-----------------")
    
    config["neo4j"]["uri"] = input(f"Neo4j URI [{config['neo4j']['uri']}]: ") or config["neo4j"]["uri"]
    config["neo4j"]["user"] = input(f"Username [{config['neo4j']['user']}]: ") or config["neo4j"]["user"]
    config["neo4j"]["password"] = input(f"Password: ") or config["neo4j"]["password"]
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")

def setup_directories():
    """Create necessary directories"""
    directories = ["models", "logs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def build_model():
    """Build the TensorRT model"""
    if os.path.exists("build_sensor_model.py"):
        print("\nBuilding TensorRT model...")
        try:
            subprocess.run([sys.executable, "build_sensor_model.py"], check=True)
            print("Model built successfully.")
        except subprocess.CalledProcessError:
            print("Model building failed. See error output above.")
    else:
        print("build_sensor_model.py not found. Cannot build model.")

def test_connection():
    """Test Neo4j connection"""
    if not os.path.exists("config.json"):
        print("Configuration file not found. Please run setup with --configure first.")
        return
    
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    print("\nTesting Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            config["neo4j"]["uri"],
            auth=(config["neo4j"]["user"], config["neo4j"]["password"])
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' AS message")
            message = result.single()["message"]
            print(message)
        
        driver.close()
    except Exception as e:
        print(f"Connection failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup BME680 Sensor TensorRT LLM Integration")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    parser.add_argument("--configure", action="store_true", help="Configure Neo4j connection")
    parser.add_argument("--build-model", action="store_true", help="Build TensorRT model")
    parser.add_argument("--test-connection", action="store_true", help="Test Neo4j connection")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.all or args.check:
        check_dependencies()
    
    if args.all:
        setup_directories()
    
    if args.all or args.configure:
        configure_neo4j()
    
    if args.all or args.build_model:
        build_model()
    
    if args.all or args.test_connection:
        test_connection()
    
    if args.all:
        print("\nSetup completed. You can now run the sensor data collection with:")
        print("  python sensorloader_trt.py")

if __name__ == "__main__":
    main()