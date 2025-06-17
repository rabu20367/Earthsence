import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at beginning to prioritize local modules

# Initialize FastAPI app
app = FastAPI(title="EarthSense AI - Sensor Data Analysis")

# Set up directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
STATIC_DIR.mkdir(exist_ok=True, parents=True)
TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Set up directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize processor
processor = SensorDataProcessor()

# HTML Templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EarthSense AI - Sensor Data Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container { margin-top: 20px; }
        .plot-container { margin-bottom: 30px; }
        .upload-section { margin-bottom: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">EarthSense AI - Sensor Data Analysis</h1>
        
        <div class="upload-section">
            <h3>Upload Sensor Data</h3>
            <form id="uploadForm" enctype="multipart/form-data" onsubmit="return uploadFile(event)">
                <div class="mb-3">
                    <input type="file" class="form-control" id="fileInput" name="file" accept=".csv,.json,.parquet,.feather" required>
                </div>
                <div class="mb-3">
                    <label class="form-label">File Type:</label>
                    <select name="file_format" class="form-select">
                        <option value="csv">CSV</option>
                        <option value="json">JSON</option>
                        <option value="parquet">Parquet</option>
                        <option value="feather">Feather</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-primary">Upload and Process</button>
            </form>
        </div>
        
        <div id="results" class="mt-4">
            <!-- Results will be displayed here -->
        </div>
    </div>
    
    <script>
    async function uploadFile(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Display the plot
                const plotDiv = document.createElement('div');
                plotDiv.id = 'plot';
                document.getElementById('results').innerHTML = '';
                document.getElementById('results').appendChild(plotDiv);
                
                // Parse the plot data
                const plotData = JSON.parse(result.plot_data);
                Plotly.newPlot('plot', plotData.data, plotData.layout);
                
                // Display anomaly info if available
                if (result.anomaly_info) {
                    const anomalyInfo = document.createElement('div');
                    anomalyInfo.className = 'alert alert-info mt-3';
                    anomalyInfo.innerHTML = `
                        <h4>Anomaly Detection Results</h4>
                        <p>Detected ${result.anomaly_count} anomalies in the data.</p>
                        <pre>${JSON.stringify(result.anomaly_info, null, 2)}</pre>
                    `;
                    document.getElementById('results').appendChild(anomalyInfo);
                }
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the file.');
        }
        
        return false;
    }
    </script>
</body>
</html>
"""

# Import SensorDataProcessor after setting up paths
from src.data_processing.sensor_processor import SensorDataProcessor

# Initialize processor
processor = SensorDataProcessor()

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_format: str = Form(...)
):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the file
        df = processor.load_sensor_data(str(file_path), file_format=file_format)
        
        if df is None or df.empty:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Failed to load or process the file"}
            )
        
        # Preprocess data
        processed_df = processor.preprocess_data(df, freq='1H')
        
        # Detect anomalies
        anomalies, anomaly_info = processor.detect_anomalies(processed_df)
        
        # Create interactive plot
        fig = make_subplots(rows=1, cols=1)
        
        # Add original data
        for col in processed_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=processed_df.index,
                    y=processed_df[col],
                    mode='lines',
                    name=f"{col} (Original)"
                )
            )
        
        # Add anomalies if any
        if not anomalies.empty:
            for col in processed_df.columns:
                anomaly_col = f'anomaly_{col}'
                if anomaly_col in anomalies.columns:
                    anomaly_points = processed_df[anomalies[anomaly_col]]
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_points.index,
                            y=anomaly_points[col],
                            mode='markers',
                            marker=dict(color='red', size=10),
                            name=f"{col} (Anomaly)",
                            showlegend=True
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title='Sensor Data with Anomalies',
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title='Legend',
            hovermode='x unified'
        )
        
        # Convert plot to JSON
        plot_json = fig.to_json()
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "plot_data": plot_json,
            "anomaly_info": anomaly_info,
            "anomaly_count": sum(len(v['anomaly_indices']) for v in anomaly_info.values())
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
