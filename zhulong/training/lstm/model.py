"""LSTM 二分类模型构建。"""

from __future__ import annotations

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    seq_len: int = 60,
    n_features: int = 5,
    lstm_units: tuple[int, int] = (64, 32),
) -> Sequential:
    model = Sequential(
        [
            LSTM(lstm_units[0], input_shape=(seq_len, n_features), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(lstm_units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


def default_callbacks(model_path: str) -> list:
    return [
        EarlyStopping(monitor="val_AUC", mode="max", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_AUC", mode="max", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_path, monitor="val_AUC", mode="max", save_best_only=True, verbose=1),
    ]


CLASS_WEIGHT = {0: 1.0, 1: 1.8}
