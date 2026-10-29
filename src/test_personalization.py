from personalization import (
    get_personalization_stage,
    calculate_personal_bias,
    personalize_prediction
)


def make_history(n):
    history = []

    for i in range(n):
        history.append({
            "productivity_score": 45.0,
            "actual_productivity": 50.0,
            "record_date": f"2026-09-{i+1:02d}"
        })

    return history


print("\n========== PERSONALIZATION TEST ==========\n")


# ------------------------------------------------------------
# TEST 1 — NEW USER
# ------------------------------------------------------------

result = personalize_prediction(
    population_prediction=46.5,
    history=[]
)

print("0 days:")
print(result)


# ------------------------------------------------------------
# TEST 2 — EARLY HISTORY
# ------------------------------------------------------------

result = personalize_prediction(
    population_prediction=46.5,
    history=make_history(3)
)

print("\n3 completed days:")
print(result)


# ------------------------------------------------------------
# TEST 3 — BUILDING PERSONAL PROFILE
# ------------------------------------------------------------

result = personalize_prediction(
    population_prediction=46.5,
    history=make_history(7)
)

print("\n7 completed days:")
print(result)


# ------------------------------------------------------------
# TEST 4 — SUFFICIENT HISTORY
# ------------------------------------------------------------

result = personalize_prediction(
    population_prediction=46.5,
    history=make_history(21)
)

print("\n21 completed days:")
print(result)


# ------------------------------------------------------------
# TEST 5 — LONG TERM
# ------------------------------------------------------------

result = personalize_prediction(
    population_prediction=46.5,
    history=make_history(60)
)

print("\n60 completed days:")
print(result)


print("\n===========================================\n")