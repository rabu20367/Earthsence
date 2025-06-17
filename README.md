# 🌍 EarthSense AI

An AI-powered environmental monitoring system for detecting ecological threats using satellite imagery, drone data, and IoT sensors.

## 🚀 Features

- Real-time environmental monitoring
- Multi-modal data integration (satellite, drone, IoT)
- Threat detection and alerting
- Interactive dashboard
- RESTful API

## 🛠️ Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/earthsense-ai.git
   cd earthsense-ai
   ```

2. **Set up the environment**
   - On Windows:
     ```powershell
     .\setup_environment.ps1
     ```
   - On Linux/macOS:
     ```bash
     chmod +x setup_environment.sh
     ./setup_environment.sh
     ```

3. **Initialize the database and directories**
   ```bash
   python init_db.py
   ```

4. **Start the development server**
   - On Windows:
     ```powershell
     .\run.ps1
     ```
   - On Linux/macOS:
     ```bash
     ./run.sh
     ```

5. **Access the application**
   - API Documentation: http://127.0.0.1:8000/docs
   - Web Interface: http://127.0.0.1:8000

## 📂 Project Structure

```
earthsense-ai/
├── data/                   # Data storage
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data_ingestion/     # Data collection
│   ├── data_processing/    # Data processing
│   └── models/             # ML models
├── static/                 # Static files
├── templates/              # HTML templates
├── tests/                  # Test files
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
├── web_requirements.txt   # Web-specific dependencies
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please contact [atm.hasibur.rashid20367@gmail.com]Sense AI

An AI-driven environmental monitoring system for detecting and predicting ecological threats using multi-modal data streams from satellites, drones, and IoT sensors.

## 🌍 Project Overview

EarthSense is designed to monitor and predict environmental threats such as:
- Illegal logging detection
- Wildfire detection and prediction
- Water pollution monitoring
- Air quality assessment

## 🚀 Features

- **Multi-modal Data Integration**: Combines satellite imagery, drone data, and IoT sensor data
- **Real-time Processing**: Processes environmental data in real-time
- **AI-Powered Analysis**: Uses advanced computer vision and time-series analysis
- **Scalable Architecture**: Deployable on both cloud and edge devices
- **Interactive Dashboard**: Web-based visualization of environmental metrics

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/earthsense-ai.git
   cd earthsense-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🚦 Quick Start

1. Run the API server:
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs`

## 📂 Project Structure

```
earthsense/
├── config/               # Configuration files
├── data/                 # Data storage
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── api/              # FastAPI application
│   ├── data_ingestion/   # Data collection modules
│   ├── data_processing/  # Data cleaning and transformation
│   └── models/           # ML model definitions
└── tests/                # Test files
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue or contact us at [atm.hasibur.rashid20367@gmail.com](atm.hasibur.rashic20367@gmail.com)
