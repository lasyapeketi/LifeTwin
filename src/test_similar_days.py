from similar_days import (
    find_similar_days,
    summarize_similar_days
)


current_day = {

    "sleep_hours": 7.0,
    "sleep_quality": 4.0,
    "wake_time_consistency": 0.8,

    "study_work_hours": 6.0,
    "focus_sessions": 4,

    "screen_time_hours": 5.0,
    "social_media_hours": 2.0,
    "notification_interruptions": 10,

    "exercise_minutes": 30,
    "step_count": 7000,

    "stress_level": 3.0,
    "mood_score": 4.0,
    "energy_level": 4.0,

    "caffeine_intake": 2,
    "break_minutes": 60
}


history = [

    {
        **current_day,
        "record_date": "2026-09-01",
        "productivity_score": 48,
        "actual_productivity": 52
    },

    {
        **current_day,
        "record_date": "2026-09-02",
        "productivity_score": 55,
        "actual_productivity": 60
    },

    {
        **current_day,
        "record_date": "2026-09-03",
        "productivity_score": 40,
        "actual_productivity": 42
    }
]


similar_days = find_similar_days(
    current_day,
    history,
    n_neighbors=3
)


print("\n========== SIMILAR DAY TEST ==========\n")

for day in similar_days:

    print(
        f"Date: {day['record_date']} | "
        f"Actual productivity: "
        f"{day['actual_productivity']} | "
        f"Distance: "
        f"{day['distance']:.2f}"
    )


summary = summarize_similar_days(
    similar_days
)

print("\nSummary:")
print(summary)

print("\n=======================================\n")