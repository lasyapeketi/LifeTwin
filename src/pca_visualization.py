import os
import joblib
import numpy as np



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "behavioral"
)

PCA_PATH = os.path.join(
    MODEL_DIR,
    "pca.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

scaler = joblib.load(SCALER_PATH)

pca = joblib.load(PCA_PATH)

BEHAVIORAL_FEATURES = [
    "sleep_hours",
    "sleep_quality",
    "wake_time_consistency",
    "study_work_hours",
    "focus_sessions",
    "screen_time_hours",
    "social_media_hours",
    "notification_interruptions",
    "exercise_minutes",
    "step_count",
    "stress_level",
    "mood_score",
    "energy_level",
    "caffeine_intake",
    "break_minutes"
]


def transform_to_pca(user_input):
    values = np.array([
        float(user_input[feature])
        for feature in BEHAVIORAL_FEATURES
    ]).reshape(1, -1)

    scaled_values = scaler.transform(
        values
    )

    transformed = pca.transform(
        scaled_values
    )

    return {
        "pca_1": float(transformed[0][0]),
        "pca_2": float(transformed[0][1])
    }