"""
core.py

Shared utilities for:
- Weather fetching (Open-Meteo)
- Network metric simulation
- Risk evaluation & decision engine
- Historical storage (CSV)
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# ---------- CONFIG ----------

DATA_DIR = "data"
MODELS_DIR = "models"
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 10 US cities from different regions (lat, lon)
CITY_COORDS = {
    "New York": (40.7128, -74.0060),       # Northeast
    "Los Angeles": (34.0522, -118.2437),   # West Coast
    "Chicago": (41.8781, -87.6298),        # Midwest
    "Houston": (29.7604, -95.3698),        # South
    "Phoenix": (33.4484, -112.0740),       # Southwest
    "Miami": (25.7617, -80.1918),          # Southeast
    "Seattle": (47.6062, -122.3321),       # Pacific Northwest
    "Denver": (39.7392, -104.9903),        # Mountain
    "Boston": (42.3601, -71.0589),         # Northeast
    "Atlanta": (33.7490, -84.3880),        # Southeast
}

NETWORK_TYPES = ["mmWave", "Sub6", "Satellite"]

# Variables to request from Open-Meteo
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain",
    "cloud_cover",
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------- WEATHER FETCHING ----------

def fetch_historical_weather(city: str, past_days: int = 14) -> pd.DataFrame:
    """
    Fetch hourly historical & near-future weather data for a city.
    Uses Open-Meteo's 'past_days' and 'forecast_days=1' as a simple approach.
    Returns a DataFrame with timestamp + basic weather columns.
    """
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Add it to CITY_COORDS in core.py")

    lat, lon = CITY_COORDS[city]

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    times = data["hourly"]["time"]
    df = pd.DataFrame({"timestamp": pd.to_datetime(times)})

    mapping = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed",
        "rain": "rain",
        "cloud_cover": "cloud_cover",
    }

    for api_name, col_name in mapping.items():
        df[col_name] = data["hourly"][api_name]

    df["city"] = city
    return df


def fetch_current_weather(city: str) -> dict:
    """
    Fetch current weather for a city using Open-Meteo.
    Returns a dict with temperature, humidity, pressure, wind_speed, rain, cloud_cover.
    We'll simply take the latest hourly value.
    """
    df = fetch_historical_weather(city, past_days=0)
    latest = df.sort_values("timestamp").iloc[-1]

    return {
        "temperature": float(latest["temperature"]),
        "humidity": float(latest["humidity"]),
        "pressure": float(latest["pressure"]),
        "wind_speed": float(latest["wind_speed"]),
        "rain": float(latest["rain"]),
        "cloud_cover": float(latest["cloud_cover"]),
    }


def fetch_next_hours_weather(city: str, hours_ahead: int = 3) -> pd.DataFrame:
    """
    Fetch the next few hours of forecast weather for a city.
    We'll fetch 1 forecast day and then keep only 'now → now+hours_ahead'.
    """
    df = fetch_historical_weather(city, past_days=0)  # includes some forecast
    now = datetime.utcnow()
    end = now + timedelta(hours=hours_ahead)

    mask = (df["timestamp"] >= now) & (df["timestamp"] <= end)
    df_future = df.loc[mask].copy()
    df_future.reset_index(drop=True, inplace=True)
    return df_future


# ---------- NETWORK METRIC SIMULATION ----------

def simulate_network_metrics_for_row(row, network_type: str) -> dict:
    """
    Rule-based simulation of network metrics for a single (weather, network_type) pair.
    Returns dict with throughput_drop, latency_increase, packet_loss.
    These rules are simple and just for demonstration.
    """
    rain = row["rain"]
    wind = row["wind_speed"]
    cloud = row["cloud_cover"]

    throughput_drop = 0.0
    latency_increase = 5.0   # ms
    packet_loss = 0.1        # %

    if network_type == "mmWave":
        throughput_drop += rain * 3.0 + cloud * 0.1
        latency_increase += rain * 1.0 + wind * 0.5
        packet_loss += rain * 0.3 + wind * 0.05
    elif network_type == "Sub6":
        throughput_drop += rain * 1.5 + cloud * 0.05
        latency_increase += rain * 0.5 + wind * 0.2
        packet_loss += rain * 0.1 + wind * 0.02
    else:  # Satellite
        throughput_drop += rain * 1.0 + cloud * 0.2
        latency_increase += rain * 1.5 + wind * 0.3
        packet_loss += rain * 0.2 + wind * 0.03


    return {
        "throughput_drop": throughput_drop,
        "latency_increase": latency_increase,
        "packet_loss": packet_loss,
    }


def build_training_dataset(cities, past_days: int = 14) -> pd.DataFrame:
    """
    For each city, fetch past_days weather, simulate metrics
    for each network_type, and return a combined training DataFrame.
    """
    all_records = []

    for city in cities:
        weather_df = fetch_historical_weather(city, past_days=past_days)
        for _, row in weather_df.iterrows():
            for net in NETWORK_TYPES:
                metrics = simulate_network_metrics_for_row(row, net)
                record = {
                    "timestamp": row["timestamp"],
                    "city": city,
                    "network_type": net,
                    "temperature": row["temperature"],
                    "humidity": row["humidity"],
                    "pressure": row["pressure"],
                    "wind_speed": row["wind_speed"],
                    "rain": row["rain"],
                    "cloud_cover": row["cloud_cover"],
                    "throughput_drop": metrics["throughput_drop"],
                    "latency_increase": metrics["latency_increase"],
                    "packet_loss": metrics["packet_loss"],
                }
                all_records.append(record)

    df = pd.DataFrame(all_records)
    return df


# ---------- RISK & DECISION ENGINE ----------

def compute_risk_level(throughput_drop: float, packet_loss: float) -> str:
    """
    Derive risk level from predicted metrics.
    You can adjust thresholds as needed.
    """
    if throughput_drop > 30 or packet_loss > 5:
        return "High"
    elif throughput_drop > 10 or packet_loss > 1:
        return "Medium"
    else:
        return "Low"


def decision_engine(network_type: str, risk_level: str):
    """
    Returns two recommended actions for a given network_type and risk_level.
    """
    if network_type == "mmWave":
        if risk_level == "High":
            return [
                "Shift critical traffic to Sub6 fallback.",
                "Enable additional error correction / redundancy.",
            ]
        elif risk_level == "Medium":
            return [
                "Monitor mmWave KPIs more frequently.",
                "Pre-warm Sub6 resources in case of further degradation.",
            ]
        else:
            return [
                "Maintain standard configuration.",
                "Log weather and metrics for trend analysis.",
            ]
    elif network_type == "Sub6":
        if risk_level == "High":
            return [
                "Offload non-critical traffic to Satellite.",
                "Increase scheduling priority for critical services.",
            ]
        elif risk_level == "Medium":
            return [
                "Enable moderate congestion control policies.",
                "Alert NOC to potential risk in this region.",
            ]
        else:
            return [
                "Keep normal scheduling and QoS.",
                "Periodically verify latency and packet loss.",
            ]
    else:  # Satellite
        if risk_level == "High":
            return [
                "Limit bandwidth to essential services only.",
                "Route bulk traffic during less severe weather windows.",
            ]
        elif risk_level == "Medium":
            return [
                "Notify customers about potential temporary degradation.",
                "Plan reroutes using terrestrial links where available.",
            ]
        else:
            return [
                "Maintain usual satellite link configuration.",
                "Track performance for any anomalies.",
            ]


# ---------- HISTORY STORAGE ----------

def append_history(rows_df: pd.DataFrame):
    """
    Append rows to history CSV. Creates the file with header if missing.
    """
    file_exists = os.path.exists(HISTORY_PATH)
    mode = "a" if file_exists else "w"
    header = not file_exists
    rows_df.to_csv(HISTORY_PATH, mode=mode, header=header, index=False)


def load_history() -> pd.DataFrame:
    """
    Load full history CSV. Returns empty DataFrame if not present.
    """
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "city",
                "network_type",
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "rain",
                "cloud_cover",
                "throughput_drop",
                "latency_increase",
                "packet_loss",
                "risk_level",
                "action1",
                "action2",
            ]
        )
    df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
    return df
