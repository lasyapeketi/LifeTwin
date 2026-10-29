import numpy as np


# ============================================================
# PERSONALIZATION STAGES
# ============================================================

def get_personalization_stage(history_count):

    if history_count == 0:
        return {
            "stage": "NEW USER",
            "description": "Population-level baseline prediction",
            "personalized": False
        }

    elif history_count < 7:
        return {
            "stage": "EARLY HISTORY",
            "description": "Limited personal context available",
            "personalized": False
        }

    elif history_count < 21:
        return {
            "stage": "BUILDING PERSONAL PROFILE",
            "description": "Personal history is beginning to influence predictions",
            "personalized": True
        }

    elif history_count < 60:
        return {
            "stage": "SUFFICIENT PERSONAL HISTORY",
            "description": "Strong personal behavioural context available",
            "personalized": True
        }

    else:
        return {
            "stage": "LONG-TERM PERSONAL TWIN",
            "description": "Rich historical behavioural context available",
            "personalized": True
        }


# ============================================================
# PERSONAL BIAS
# ============================================================

def calculate_personal_bias(history):

    """
    Calculate the user's historical prediction bias.

    Error = Actual productivity - Predicted productivity

    Positive bias:
        Model tends to underestimate the user.

    Negative bias:
        Model tends to overestimate the user.
    """

    errors = []

    for record in history:

        predicted = record["productivity_score"]
        actual = record["actual_productivity"]

        if actual is not None and predicted is not None:

            error = actual - predicted
            errors.append(error)

    if not errors:
        return 0.0

    # Shrink the estimate when history is small.
    # This prevents one unusual day from causing
    # a large personalization jump.

    mean_error = np.mean(errors)

    n = len(errors)

    shrinkage_strength = 5

    personal_bias = (
        mean_error * n
        / (n + shrinkage_strength)
    )

    return float(personal_bias)


# ============================================================
# PERSONALIZED PREDICTION
# ============================================================

def personalize_prediction(
    population_prediction,
    history
):

    valid_history = [
        record
        for record in history
        if record["actual_productivity"] is not None
    ]

    history_count = len(valid_history)

    stage = get_personalization_stage(
        history_count
    )

    # Not enough history → keep population prediction
    if not stage["personalized"]:

        return {
            "prediction": float(population_prediction),
            "population_prediction": float(population_prediction),
            "personal_bias": 0.0,
            "stage": stage["stage"],
            "description": stage["description"],
            "history_count": history_count
        }

    # Calculate personal bias
    personal_bias = calculate_personal_bias(
        valid_history
    )

    personalized_prediction = (
        population_prediction + personal_bias
    )

    # Keep score inside valid range
    personalized_prediction = np.clip(
        personalized_prediction,
        0,
        100
    )

    return {
        "prediction": float(personalized_prediction),
        "population_prediction": float(population_prediction),
        "personal_bias": float(personal_bias),
        "stage": stage["stage"],
        "description": stage["description"],
        "history_count": history_count
    }