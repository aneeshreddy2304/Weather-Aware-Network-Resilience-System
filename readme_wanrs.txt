# Weather Aware Network Resilience System (WANRS)

## Project Overview

An AI-driven system that predicts weather-induced network degradation and recommends mitigation strategies for heterogeneous networks (mmWave, Sub-6 GHz, Satellite). The system uses ensemble ML models (Random Forest + Gradient Boosting) for current predictions and LSTM with attention mechanism for 3-hour forecasting.

## Features

- **Real-time Weather Integration**: Fetches live weather data from Open-Meteo API
- **Multi-Network Prediction**: Predicts throughput drop, latency increase, and packet loss for three network types
- **Ensemble ML Models**: Combines Random Forest and Gradient Boosting for robust predictions
- **LSTM Forecasting**: 3-hour ahead predictions using deep learning with attention mechanism
- **Interactive Dashboard**: Streamlit-based UI with visualizations and historical trends
- **Risk Assessment**: Automatic risk level computation and mitigation recommendations
- **History Tracking**: Persistent storage of predictions for trend analysis

---

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: 2GB free disk space
- **Internet**: Stable connection for API calls

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8, 3.9, 3.10, or 3.11 (3.9+ recommended)
- **pip**: Latest version (upgrade with `pip install --upgrade pip`)

---

## Installation Instructions

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd WANRS

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If requirements.txt is not available, install manually:**
```bash
pip install numpy pandas scikit-learn tensorflow streamlit plotly requests joblib
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
```

---

## Project Structure

```
WANRS/
│
├── core.py                 # Core utilities (weather API, simulation, risk engine)
├── train_models.py         # Model training script (RF, GB, LSTM)
├── app.py                  # Streamlit dashboard application
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Created automatically
│   └── history.csv        # Prediction history (generated at runtime)
│
└── models/                # Created automatically
    ├── rf_multi_output_pipeline.pkl    # Random Forest model
    ├── gb_multi_output_pipeline.pkl    # Gradient Boosting model
    └── lstm_model.keras                # LSTM forecasting model
```

---

## Usage Instructions

### Phase 1: Train the Models (Required - First Time Only)

Before running the dashboard, you must train the ML models:

```bash
python train_models.py
```

**Expected Output:**
```
Building training dataset from simulated metrics (30 days)...
Total dataset size: 8928 rows
Training RandomForest multi-output model...
Training GradientBoosting multi-output model...
Evaluating models...

RandomForest performance:
  throughput_drop: MAE=0.005, RMSE=0.045
  latency_increase: MAE=0.015, RMSE=0.074
  packet_loss: MAE=0.002, RMSE=0.009

GradientBoosting performance:
  throughput_drop: MAE=0.078, RMSE=0.137
  latency_increase: MAE=0.044, RMSE=0.069
  packet_loss: MAE=0.006, RMSE=0.010

Building sequence dataset for LSTM (30 days)...
Built sequence dataset: X=(8832, 6, 9), y=(8832, 3, 3)

Training LSTM + attention model with early stopping...
[Training progress displayed...]

Saved LSTM model to: models/lstm_model.keras
```

**Training Time:**
- Ensemble Models: 1-3 minutes
- LSTM Model: 5-10 minutes (depending on hardware)
- Total: ~10-15 minutes

**Note:** This step only needs to be run once. The trained models are saved and reused by the dashboard.

### Phase 2: Launch the Dashboard

```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The dashboard will automatically open in your default web browser. If not, manually navigate to `http://localhost:8501`.

---

## Using the Dashboard

### 1. City Selection
- Use the dropdown menu to select a city (New York, London, Tokyo, Delhi)
- Weather data and predictions update automatically

### 2. Current Weather Display
- View real-time weather conditions from Open-Meteo API
- Shows temperature, humidity, pressure, wind speed, rain, and cloud cover

### 3. Current Network Impact Predictions
- Table showing predictions for all three network types
- Displays throughput drop (%), latency increase (ms), and packet loss (%)
- Risk level classification (Low/Medium/High)
- Two recommended mitigation actions per network type

### 4. Visualizations
- **Bar Charts**: Compare degradation metrics across network types
- **Historical Trends**: Line charts showing metric evolution over time
- Select different metrics from the dropdown menu

### 5. 3-Hour Forecast
- LSTM-based predictions for the next 3 hours
- Shows future trends for all network types
- Includes forecasted risk levels and actions
- **Note**: Requires at least 6 historical data points to generate forecasts

### 6. Automatic History Logging
- All predictions are automatically saved to `data/history.csv`
- History accumulates over time for better forecasting
- View historical trends through the dashboard interface

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow
```

For Apple Silicon Macs:
```bash
pip install tensorflow-macos tensorflow-metal
```

### Issue: "FileNotFoundError: LSTM model file not found"

**Solution:**
Run the training script first:
```bash
python train_models.py
```

### Issue: "requests.exceptions.ConnectionError" or Weather API fails

**Possible Causes:**
- No internet connection
- Open-Meteo API temporarily unavailable
- Firewall blocking requests

**Solution:**
- Check internet connectivity
- Wait a few minutes and try again
- Check firewall settings

### Issue: Dashboard is slow or unresponsive

**Solutions:**
- Ensure only one instance of the dashboard is running
- Close other resource-intensive applications
- Clear browser cache
- Restart the Streamlit server (`Ctrl+C` and rerun)

### Issue: "AttributeError: module 'tensorflow' has no attribute 'keras'"

**Solution:**
Upgrade TensorFlow to a compatible version:
```bash
pip install --upgrade tensorflow>=2.10.0
```

### Issue: Models train but predictions seem incorrect

**Solution:**
- Delete the `models/` folder
- Rerun `python train_models.py` to retrain from scratch
- Verify training metrics match expected values

### Issue: Port 8501 already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## Configuration & Customization

### Adding More Cities

Edit `core.py` and add to `CITY_COORDS`:
```python
CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Your City": (latitude, longitude),  # Add here
}
```

Then retrain models: `python train_models.py`

### Adjusting Network Types

Edit `core.py` and modify `NETWORK_TYPES`:
```python
NETWORK_TYPES = ["mmWave", "Sub6", "Satellite", "LTE"]  # Add types
```

Update simulation rules in `simulate_network_metrics_for_row()` function.

### Changing Forecast Horizon

Edit `train_models.py`:
```python
# Change from 3 to desired hours
seq_len = 6
horizon = 6  # Now forecasts 6 hours ahead
```

Retrain LSTM model after changes.

### Modifying Risk Thresholds

Edit `core.py` in `compute_risk_level()`:
```python
def compute_risk_level(throughput_drop, packet_loss):
    if throughput_drop > 40 or packet_loss > 8:  # Adjust thresholds
        return "High"
    # ... modify as needed
```

---

## API Documentation

### Open-Meteo API
- **Endpoint**: https://api.open-meteo.com/v1/forecast
- **Rate Limit**: 10,000 requests/day (free tier)
- **Documentation**: https://open-meteo.com/en/docs

The system automatically handles API requests. No API key required.

---

## Performance Benchmarks

### Model Performance (on synthetic data)

**Random Forest:**
- Throughput Drop: MAE=0.005, RMSE=0.045
- Latency Increase: MAE=0.015, RMSE=0.074
- Packet Loss: MAE=0.002, RMSE=0.009

**Gradient Boosting:**
- Throughput Drop: MAE=0.078, RMSE=0.137
- Latency Increase: MAE=0.044, RMSE=0.069
- Packet Loss: MAE=0.006, RMSE=0.010

**LSTM:**
- Best Validation Loss: 3.84
- Validation MAE: 0.90
- Training Time: ~5-10 minutes

---

## Known Limitations

1. **Synthetic Training Data**: Models are trained on simulated data, not real network metrics
2. **Single-City Operation**: Dashboard operates on one city at a time
3. **Limited Weather Features**: Only 6 weather parameters currently used
4. **3-Hour Forecast Limit**: LSTM horizon limited to 3 hours
5. **No Real-Time Network Data**: System doesn't integrate with actual network monitoring tools

---

## Future Enhancements

- Integration with real network monitoring systems
- Extended forecast horizon (24-48 hours)
- Additional weather features (lightning, storms, visibility)
- Multi-city simultaneous monitoring
- Cloud deployment with REST API
- Reinforcement learning for adaptive mitigation
- Mobile application interface

---

## Dependencies

**Core Libraries:**
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - ML algorithms
- `tensorflow>=2.10.0` - Deep learning
- `streamlit>=1.20.0` - Web dashboard
- `plotly>=5.0.0` - Interactive visualizations
- `requests>=2.27.0` - HTTP requests
- `joblib>=1.1.0` - Model serialization

**Full requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
streamlit>=1.20.0
plotly>=5.0.0
requests>=2.27.0
joblib>=1.1.0
```

---

## Support & Contact

For issues, questions, or contributions:
- **Issues**: Report bugs or request features via project issues
- **Documentation**: Refer to code comments and this README
- **Updates**: Check for new releases and updates

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- **Open-Meteo**: Free weather API service
- **ITU-R Standards**: Physics-based simulation models (P.838, P.840)
- **Streamlit**: Interactive dashboard framework
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning library

---

## Quick Start Summary

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (first time only)
python train_models.py

# 4. Run dashboard
streamlit run app.py

# 5. Open browser to http://localhost:8501
```

**Enjoy using WANRS! 🌦️📡**