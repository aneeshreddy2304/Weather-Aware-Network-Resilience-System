"""
app.py

Streamlit dashboard for:
- Real-time weather & network impact prediction (ensemble models)
- Near-future network degradation forecast (LSTM, if available)
- Recommended actions per network type
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import tensorflow as tf


from core import (
    CITY_COORDS,
    NETWORK_TYPES,
    fetch_current_weather,
    compute_risk_level,
    decision_engine,
    append_history,
    load_history,
    MODELS_DIR,
)


# ---------- CACHED MODEL LOADERS ----------

@st.cache_resource
def load_ensemble_models():
    rf_path = os.path.join(MODELS_DIR, "rf_multi_output_pipeline.pkl")
    gb_path = os.path.join(MODELS_DIR, "gb_multi_output_pipeline.pkl")
    rf_pipeline = joblib.load(rf_path)
    gb_pipeline = joblib.load(gb_path)
    return rf_pipeline, gb_pipeline


@st.cache_resource
def load_lstm_model():
    """
    Load LSTM model (MANDATORY).
    Raises FileNotFoundError if the model file is missing.
    """
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(
            f"LSTM model file not found at {lstm_path}. "
            "Run `python train_models.py` to train and save the LSTM model."
        )

    model = tf.keras.models.load_model(lstm_path)
    return model


# ---------- PREDICTION HELPERS ----------

def predict_metrics_and_risk_for_network(
    current_weather: dict,
    network_type: str,
    rf_pipeline,
    gb_pipeline,
):
    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "cloud_cover",
        "network_type",
    ]

    row = {
        "temperature": current_weather["temperature"],
        "humidity": current_weather["humidity"],
        "pressure": current_weather["pressure"],
        "wind_speed": current_weather["wind_speed"],
        "rain": current_weather["rain"],
        "cloud_cover": current_weather["cloud_cover"],
        "network_type": network_type,
    }

    X_input = pd.DataFrame([row], columns=feature_cols)

    y_rf = rf_pipeline.predict(X_input)
    y_gb = gb_pipeline.predict(X_input)

    y_pred = (y_rf + y_gb) / 2.0
    throughput_drop, latency_increase, packet_loss = y_pred[0]

    risk_level = compute_risk_level(throughput_drop, packet_loss)
    action1, action2 = decision_engine(network_type, risk_level)

    result = {
        "throughput_drop": float(throughput_drop),
        "latency_increase": float(latency_increase),
        "packet_loss": float(packet_loss),
        "risk_level": risk_level,
        "action1": action1,
        "action2": action2,
    }
    return result


def build_lstm_sequence_for_city(df_history, city, network_type, seq_len=6):
    """
    Build the most recent sequence for (city, network_type) from history.
    If there are fewer than seq_len points, pad with the last available row
    so we can still feed a full-length sequence into the LSTM.
    """
    sub = df_history[
        (df_history["city"] == city)
        & (df_history["network_type"] == network_type)
    ].sort_values("timestamp")

    # No data at all for this city/network_type
    if len(sub) == 0:
        return None

    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "cloud_cover",
        "throughput_drop",
        "latency_increase",
        "packet_loss",
    ]

    feats = sub[feature_cols].values  # shape: (N, num_features)

    # If we have fewer than seq_len points, pad using the last row
    if len(sub) < seq_len:
        last_row = feats[-1]                      # (num_features,)
        pad_count = seq_len - len(sub)
        pad_block = np.tile(last_row, (pad_count, 1))  # (pad_count, num_features)
        feats = np.vstack([feats, pad_block])

    # Take the last seq_len rows (either original or padded)
    seq = feats[-seq_len:]                        # (seq_len, num_features)
    seq = np.expand_dims(seq, axis=0)             # (1, seq_len, num_features)
    return seq



def forecast_future_metrics_for_network(
    lstm_model,
    df_history,
    city,
    network_type,
    seq_len=6,
    horizon=3,
):
    """
    Use LSTM model to forecast next 'horizon' steps for one network type.
    Returns a DataFrame with predicted metrics per future step.
    If lstm_model is None or not enough history, returns None.
    """
    if lstm_model is None:
        return None

    seq = build_lstm_sequence_for_city(df_history, city, network_type, seq_len=seq_len)
    if seq is None:
        return None

    y_pred = lstm_model.predict(seq)  # shape (1, horizon, 3)
    y_pred = y_pred[0]  # (horizon, 3)

    now = datetime.now(timezone.utc)
    future_rows = []
    for i in range(horizon):
        future_time = now + pd.Timedelta(hours=i + 1)
        throughput_drop, latency_increase, packet_loss = y_pred[i]

        risk_level = compute_risk_level(throughput_drop, packet_loss)
        action1, action2 = decision_engine(network_type, risk_level)

        future_rows.append(
            {
                "timestamp": future_time,
                "city": city,
                "network_type": network_type,
                "throughput_drop": float(throughput_drop),
                "latency_increase": float(latency_increase),
                "packet_loss": float(packet_loss),
                "risk_level": risk_level,
                "action1": action1,
                "action2": action2,
            }
        )

    df_future = pd.DataFrame(future_rows)
    return df_future


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(
        page_title="Weather-Aware Network Resilience",
        layout="wide",
    )

    st.title(" AI-Driven Weather-Aware Network Resilience Dashboard")

    cities = list(CITY_COORDS.keys())
    city = st.selectbox("Select a city", cities, index=0)

    rf_pipeline, gb_pipeline = load_ensemble_models()
    lstm_model = load_lstm_model()

    st.subheader(f"Current Weather in {city}")
    current_weather = fetch_current_weather(city)
    st.write(current_weather)

    # Predict current metrics + risk for each network type
    results = []
    now = datetime.now(timezone.utc)

    for net in NETWORK_TYPES:
        res = predict_metrics_and_risk_for_network(
            current_weather=current_weather,
            network_type=net,
            rf_pipeline=rf_pipeline,
            gb_pipeline=gb_pipeline,
        )
        res_row = {
            "timestamp": now,
            "city": city,
            "network_type": net,
            "temperature": current_weather["temperature"],
            "humidity": current_weather["humidity"],
            "pressure": current_weather["pressure"],
            "wind_speed": current_weather["wind_speed"],
            "rain": current_weather["rain"],
            "cloud_cover": current_weather["cloud_cover"],
            **res,
        }
        results.append(res_row)

    df_current = pd.DataFrame(results)
    st.subheader("Current Network Impact (Predicted)")
    st.dataframe(
        df_current[
            [
                "network_type",
                "throughput_drop",
                "latency_increase",
                "packet_loss",
                "risk_level",
                "action1",
                "action2",
            ]
        ]
    )

    # Append to history
    append_history(df_current)

    # Bar chart for current impact
    st.subheader("Current Impact Comparison Across Network Types")
    for metric in ["throughput_drop", "latency_increase", "packet_loss"]:
        fig = px.bar(
            df_current,
            x="network_type",
            y=metric,
            color="network_type",
            title=f"Current {metric} by Network Type",
        )
        st.plotly_chart(fig, width='stretch')

    # Load history for trends and forecasting
    df_history = load_history()

    st.subheader("Historical Trends (All Network Types)")

    if not df_history.empty:
        metric_choice = st.selectbox(
            "Select metric to view trend",
            ["throughput_drop", "latency_increase", "packet_loss"],
        )

        fig_hist = px.line(
            df_history.sort_values("timestamp"),
            x="timestamp",
            y=metric_choice,
            color="network_type",
            title=f"Historical {metric_choice} Trends",
        )
        st.plotly_chart(fig_hist, width='stretch')

    # Forecast future metrics using LSTM (if available)
    st.subheader("Forecasted Network Degradation (Next 3 Hours)")

    all_future = []
    for net in NETWORK_TYPES:
        df_future = forecast_future_metrics_for_network(
            lstm_model=lstm_model,
            df_history=df_history,
            city=city,
            network_type=net,
            seq_len=6,
            horizon=3,
        )
        if df_future is not None:
            all_future.append(df_future)

    if all_future:
        df_future_all = pd.concat(all_future, ignore_index=True)
        st.write("Forecast table:")
        st.dataframe(
            df_future_all[
                [
                    "timestamp",
                    "network_type",
                    "throughput_drop",
                    "latency_increase",
                    "packet_loss",
                    "risk_level",
                    "action1",
                    "action2",
                ]
            ]
        )

        fig_future = px.line(
            df_future_all,
            x="timestamp",
            y="throughput_drop",
            color="network_type",
            title="Forecasted Throughput Drop (Next 3 Hours)",
        )
        st.plotly_chart(fig_future, width='stretch')
    else:
        if lstm_model is None:
            st.info(
                "LSTM model not available (TensorFlow not installed or model not trained). "
                "The dashboard still shows current predictions and historical trends."
            )
        else:
            st.info(
                "Not enough historical data to run LSTM forecast yet. "
                "Let the system collect more history over time."
            )


if __name__ == "__main__":
    main()