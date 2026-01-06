"""
train_models.py

- Build training dataset (weather + simulated metrics)
- Train multi-output ensemble (RF + GB)
- Train LSTM + attention for time-series forecasting (if TensorFlow is available)
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from core import (
    build_training_dataset,
    CITY_COORDS,
    NETWORK_TYPES,
    MODELS_DIR,
)


# ------------- PHASE 1: ENSEMBLE MULTI-OUTPUT REGRESSION -------------


def train_ensemble_models():
    print("Building training dataset from simulated metrics (30 days)...")
    cities = list(CITY_COORDS.keys())
    df = build_training_dataset(cities, past_days=30)

    print(f"Total dataset size: {len(df)} rows")

    feature_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "cloud_cover",
        "network_type",
    ]

    target_cols = [
        "throughput_drop",
        "latency_increase",
        "packet_loss",
    ]

    X = df[feature_cols]
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain",
        "cloud_cover",
    ]
    categorical_features = ["network_type"]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf_base = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    gb_base = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )

    rf_multi = MultiOutputRegressor(rf_base)
    gb_multi = MultiOutputRegressor(gb_base)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_multi),
        ]
    )
    gb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gb_multi),
        ]
    )

    print("Training RandomForest multi-output model...")
    rf_pipeline.fit(X_train, y_train)

    print("Training GradientBoosting multi-output model...")
    gb_pipeline.fit(X_train, y_train)

    print("Evaluating models...")

    def print_regression_metrics(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
        print(f"\n{name} performance:")
        for i, col in enumerate(target_cols):
            print(f"  {col}: MAE={mae[i]:.3f}, RMSE={rmse[i]:.3f}")

    y_pred_rf = rf_pipeline.predict(X_test)
    y_pred_gb = gb_pipeline.predict(X_test)

    print_regression_metrics("RandomForest", y_test, y_pred_rf)
    print_regression_metrics("GradientBoosting", y_test, y_pred_gb)

    rf_path = os.path.join(MODELS_DIR, "rf_multi_output_pipeline.pkl")
    gb_path = os.path.join(MODELS_DIR, "gb_multi_output_pipeline.pkl")

    joblib.dump(rf_pipeline, rf_path)
    joblib.dump(gb_pipeline, gb_path)

    print(f"\nSaved RF model to: {rf_path}")
    print(f"Saved GB model to: {gb_path}")


# ------------- PHASE 2: LSTM + ATTENTION TIME-SERIES FORECASTING -------------


def build_sequence_data(seq_len=6, horizon=3):
    """
    Build time-series sequences from training-style data (simulated).
    """
    print("Building sequence dataset for LSTM (30 days)...")

    cities = list(CITY_COORDS.keys())
    df = build_training_dataset(cities, past_days=30)

    df = df.sort_values(["city", "network_type", "timestamp"]).reset_index(drop=True)

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

    target_cols = [
        "throughput_drop",
        "latency_increase",
        "packet_loss",
    ]

    sequences_X = []
    sequences_y = []

    for city in cities:
        for net in NETWORK_TYPES:
            sub = df[(df["city"] == city) & (df["network_type"] == net)].reset_index(
                drop=True
            )
            if len(sub) < seq_len + horizon:
                continue

            feats = sub[feature_cols].values

            for i in range(len(sub) - seq_len - horizon + 1):
                seq_x = feats[i: i + seq_len]
                future_targets = sub[target_cols].iloc[
                    i + seq_len: i + seq_len + horizon
                ].values  # (horizon, 3)

                sequences_X.append(seq_x)
                sequences_y.append(future_targets)

    X_array = np.array(sequences_X)  # (N, seq_len, num_features)
    y_array = np.array(sequences_y)  # (N, horizon, num_targets)

    print(f"Built sequence dataset: X={X_array.shape}, y={y_array.shape}")
    return X_array, y_array


def train_lstm_model():
    """
    Train LSTM + attention model with 2 LSTM layers, 128 units, and early stopping.
    """

    def build_lstm_attention_model(seq_len, num_features, horizon, num_targets):
        inputs = layers.Input(shape=(seq_len, num_features))

        # First LSTM layer
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)

        # Second LSTM layer
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)

        # Attention mechanism
        score = layers.Dense(1, activation="tanh")(x)
        attention_weights = layers.Softmax(axis=1)(score)
        context = tf.reduce_sum(attention_weights * x, axis=1)

        # Dense layers
        x2 = layers.Dense(128, activation="relu")(context)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.Dense(64, activation="relu")(x2)

        outputs = layers.Dense(horizon * num_targets)(x2)
        outputs_reshaped = layers.Reshape((horizon, num_targets))(outputs)

        model = models.Model(inputs=inputs, outputs=outputs_reshaped)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    seq_len = 6
    horizon = 3

    X, y = build_sequence_data(seq_len=seq_len, horizon=horizon)
    if len(X) == 0:
        raise RuntimeError(
            "Not enough data to train LSTM. "
            "Increase past_days in build_sequence_data() or generate more simulated data."
        )

    num_features = X.shape[2]
    num_targets = y.shape[2]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_lstm_attention_model(
        seq_len=seq_len,
        num_features=num_features,
        horizon=horizon,
        num_targets=num_targets,
    )

    # Callbacks for early stopping and learning rate reduction
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    print("\nTraining LSTM + attention model with early stopping...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Set high, early stopping will stop when appropriate
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    model.save(lstm_path)
    print(f"\nSaved LSTM model to: {lstm_path}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")


if __name__ == "__main__":
    train_ensemble_models()
    train_lstm_model()