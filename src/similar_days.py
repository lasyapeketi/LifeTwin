import numpy as np


# ============================================================
# FEATURES USED FOR SIMILAR-DAY COMPARISON
# ============================================================

BEHAVIOUR_FEATURES = [
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


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def calculate_distance(current_day, historical_day):

    distances = []

    for feature in BEHAVIOUR_FEATURES:

        current_value = float(
            current_day[feature]
        )

        historical_value = float(
            historical_day[feature]
        )

        distances.append(
            (current_value - historical_value) ** 2
        )

    return float(
        np.sqrt(sum(distances))
    )


# ============================================================
# FIND SIMILAR DAYS
# ============================================================

def find_similar_days(
    current_day,
    history,
    n_neighbors=3
):

    candidates = []

    for day in history:

        # Only use completed days
        if day.get("actual_productivity") is None:
            continue

        distance = calculate_distance(
            current_day,
            day
        )

        candidates.append({
            **day,
            "distance": distance
        })

    # Closest behavioural matches first
    candidates.sort(
        key=lambda x: x["distance"]
    )

    return candidates[:n_neighbors]


# ============================================================
# SIMILAR-DAY SUMMARY
# ============================================================

def summarize_similar_days(similar_days):

    if not similar_days:

        return {
            "count": 0,
            "average_productivity": None,
            "best_productivity": None,
            "worst_productivity": None
        }

    productivity_values = [
        day["actual_productivity"]
        for day in similar_days
    ]

    return {
        "count": len(similar_days),

        "average_productivity": float(
            np.mean(productivity_values)
        ),

        "best_productivity": float(
            np.max(productivity_values)
        ),

        "worst_productivity": float(
            np.min(productivity_values)
        )
    }