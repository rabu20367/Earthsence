"""IoT sensor data ingestion module for EarthSense."""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import paho.mqtt.client as mqtt
from paho.mqtt import subscribe
import pytz
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class IoTDataIngestor:
    """Class for ingesting and processing IoT sensor data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the IoTDataIngestor with configuration.
        
        Args:
            config: Optional configuration dictionary. If not provided, uses settings from environment variables.
        """
        self.config = config or {}
        self.mqtt_client = None
        self.influx_client = None
        self.write_api = None
        self.callbacks = []
        
        # Initialize timezone
        self.timezone = pytz.timezone(self.config.get('timezone', 'UTC'))
    
    def connect_mqtt(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Get MQTT configuration
            mqtt_config = self.config.get('mqtt', {})
            host = mqtt_config.get('host', 'localhost')
            port = mqtt_config.get('port', 1883)
            username = mqtt_config.get('username')
            password = mqtt_config.get('password')
            
            # Create MQTT client
            self.mqtt_client = mqtt.Client()
            
            # Set credentials if provided
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            
            # Set TLS if enabled
            if mqtt_config.get('tls', False):
                self.mqtt_client.tls_set()
            
            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to broker
            self.mqtt_client.connect(host, port, 60)
            logger.info(f"Connected to MQTT broker at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def connect_influxdb(self) -> bool:
        """
        Connect to InfluxDB.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Get InfluxDB configuration
            influx_config = self.config.get('influxdb', {})
            url = influx_config.get('url', 'http://localhost:8086')
            token = influx_config.get('token')
            org = influx_config.get('org', 'earthsense')
            
            # Create InfluxDB client
            self.influx_client = InfluxDBClient(
                url=url,
                token=token,
                org=org
            )
            
            # Create write API
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            # Test the connection
            health = self.influx_client.health()
            if health.status == 'pass':
                logger.info(f"Connected to InfluxDB at {url}")
                return True
            else:
                logger.error(f"InfluxDB health check failed: {health.message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            return False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for when the MQTT client connects to the broker."""
        if rc == 0:
            logger.info("MQTT client connected successfully")
            # Resubscribe to topics if needed
            for topic in self.config.get('mqtt', {}).get('topics', []):
                client.subscribe(topic)
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Callback for when a message is received on a subscribed topic."""
        try:
            # Parse the message payload
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON payload: {msg.payload}")
                return
            
            # Add metadata
            message_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'topic': msg.topic,
                'qos': msg.qos,
                'retain': msg.retain,
                'payload': payload
            }
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(message_data)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
            
            # Write to InfluxDB if connected
            if self.write_api and self.influx_client:
                self._write_to_influxdb(msg.topic, payload)
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for when the MQTT client disconnects from the broker."""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (rc: {rc}). Attempting to reconnect...")
            # Try to reconnect
            self.connect_mqtt()
    
    def _write_to_influxdb(self, topic: str, payload: Dict):
        """
        Write sensor data to InfluxDB.
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            from influxdb_client import Point
            
            # Extract measurement name from topic (last part after /)
            measurement = topic.split('/')[-1]
            
            # Create a data point
            point = Point(measurement)
            
            # Add tags (metadata)
            if 'sensor_id' in payload:
                point.tag('sensor_id', str(payload['sensor_id']))
            if 'location' in payload:
                point.tag('location', str(payload['location']))
            
            # Add fields (numeric values)
            numeric_fields = {k: v for k, v in payload.items() 
                            if isinstance(v, (int, float)) and not k.endswith('_id')}
            for field, value in numeric_fields.items():
                point.field(field, value)
            
            # Add timestamp if provided, otherwise use current time
            timestamp = payload.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        # Try to parse ISO format
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        point.time(dt)
                    except ValueError:
                        logger.warning(f"Failed to parse timestamp: {timestamp}")
                elif isinstance(timestamp, (int, float)):
                    # Assume Unix timestamp in seconds or milliseconds
                    point.time(int(timestamp * 1e9))  # Convert to nanoseconds
            
            # Write the point to InfluxDB
            self.write_api.write(
                bucket=self.config.get('influxdb', {}).get('bucket', 'sensor_data'),
                record=point
            )
            
        except Exception as e:
            logger.error(f"Failed to write to InfluxDB: {e}")
    
    def add_callback(self, callback):
        """
        Add a callback function to be called when a message is received.
        
        Args:
            callback: Function that takes a single argument (the message data)
        """
        if callable(callback):
            self.callbacks.append(callback)
        else:
            logger.warning("Callback is not callable")
    
    def start(self):
        """Start the MQTT client loop."""
        if self.mqtt_client:
            self.mqtt_client.loop_start()
        else:
            logger.error("MQTT client not initialized. Call connect_mqtt() first.")
    
    def stop(self):
        """Stop the MQTT client loop and clean up resources."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("MQTT client stopped")
        
        if self.influx_client:
            self.write_api.close()
            self.influx_client.close()
            logger.info("InfluxDB client closed")
    
    def query_sensor_data(
        self,
        measurement: str,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        filters: Optional[Dict[str, str]] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Query sensor data from InfluxDB.
        
        Args:
            measurement: Name of the measurement to query
            start_time: Start time for the query (ISO format string or datetime object)
            end_time: End time for the query (ISO format string or datetime object)
            filters: Dictionary of tag filters (e.g., {'sensor_id': 'sensor1'})
            limit: Maximum number of points to return
            
        Returns:
            DataFrame containing the query results, or None if the query failed
        """
        if not self.influx_client:
            logger.error("InfluxDB client not initialized. Call connect_influxdb() first.")
            return None
        
        try:
            from influxdb_client.client.flux_table import FluxStructureEncoder
            
            # Build the Flux query
            query = f'from(bucket: "sensor_data") '
            query += f'|> range(start: {self._format_flux_time(start_time) or "-1h"}, stop: {self._format_flux_time(end_time) or "now()"}) '
            query += f'|> filter(fn: (r) => r._measurement == "{measurement}") '
            
            # Add tag filters
            if filters:
                for key, value in filters.items():
                    query += f'|> filter(fn: (r) => r["{key}"] == "{value}") '
            
            # Add limit and sort
            query += f'|> limit(n: {limit}) '
            query += '|> sort(columns: ["_time"], desc: false)'
            
            # Execute the query
            result = self.influx_client.query_api().query_data_frame(query)
            
            # Process the result
            if isinstance(result, list):
                # If multiple tables are returned, concatenate them
                df = pd.concat(result, ignore_index=True)
            else:
                df = result
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")
            return None
    
    def _format_flux_time(self, time_value: Optional[Union[str, datetime]]) -> Optional[str]:
        """Format a time value for Flux query."""
        if time_value is None:
            return None
            
        if isinstance(time_value, str):
            # Assume ISO format, ensure it's properly quoted
            return f'"{time_value}"'
        elif isinstance(time_value, datetime):
            # Convert datetime to RFC3339 format
            return f'"{time_value.isoformat()}"'
        return None

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'mqtt': {
            'host': 'mqtt.example.com',
            'port': 1883,
            'username': 'user',
            'password': 'password',
            'topics': [
                'sensors/+/temperature',
                'sensors/+/humidity',
                'sensors/+/air_quality'
            ]
        },
        'influxdb': {
            'url': 'http://localhost:8086',
            'token': 'your-influxdb-token',
            'org': 'earthsense',
            'bucket': 'sensor_data'
        },
        'timezone': 'UTC'
    }
    
    # Create and start the ingestor
    ingestor = IoTDataIngestor(config)
    
    # Connect to MQTT and InfluxDB
    if ingestor.connect_mqtt() and ingestor.connect_influxdb():
        try:
            # Add a callback to print received messages
            def on_message(message_data):
                print(f"Received message on {message_data['topic']}: {message_data['payload']}")
            
            ingestor.add_callback(on_message)
            
            # Start the MQTT client loop
            ingestor.start()
            
            # Keep the script running
            print("Listening for MQTT messages. Press Ctrl+C to exit.")
            import time
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Stopping...")
            ingestor.stop()
    else:
        print("Failed to connect to one or more services.")
