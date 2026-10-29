import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(
    MODEL_DIR, "preprocessing", "scaler.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR, "preprocessing", "encoder.pkl"
)

REGRESSOR_PATH = os.path.join(
    MODEL_DIR, "regression", "best_regressor.pkl"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR, "classification", "best_classifier.pkl"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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

CATEGORICAL_FEATURES = [
    "age_group",
    "day_of_week"
]


# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

regressor = joblib.load(REGRESSOR_PATH)
classifier = joblib.load(CLASSIFIER_PATH)


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(user_input):

    df = pd.DataFrame([user_input])

    # Numerical features
    numerical_data = df[BEHAVIORAL_FEATURES]

    # Categorical features
    categorical_data = df[CATEGORICAL_FEATURES]

    # Apply training-fitted transformations
    numerical_scaled = scaler.transform(
        numerical_data
    )

    categorical_encoded = encoder.transform(
        categorical_data
    )

    # Combine exactly like training
    processed_data = np.hstack([
        numerical_scaled,
        categorical_encoded
    ])

    return processed_data


# ============================================================
# PREDICT PRODUCTIVITY
# ============================================================

def predict_productivity(user_input):

    processed_data = preprocess_input(user_input)

    productivity_score = regressor.predict(
        processed_data
    )[0]

    productive_class = classifier.predict(
        processed_data
    )[0]

    return {
        "productivity_score": float(productivity_score),
        "productive_class": int(productive_class)
    }