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

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

KMEANS_PATH = os.path.join(
    MODEL_DIR,
    "kmeans.pkl"
)


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


scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)


PROFILE_NAMES = {
    0: "High Work / High Stress Pattern",
    1: "Balanced / Active Pattern",
    2: "High Screen / Social Pattern"
}


def predict_behavioral_profile(user_input):

    values = np.array([
        float(user_input[feature])
        for feature in BEHAVIORAL_FEATURES
    ]).reshape(1, -1)

    scaled_values = scaler.transform(values)

    cluster = int(
        kmeans.predict(scaled_values)[0]
    )

    profile_name = PROFILE_NAMES.get(
        cluster,
        f"Behavioral Cluster {cluster}"
    )

    return {
        "cluster": cluster,
        "profile": profile_name
    }