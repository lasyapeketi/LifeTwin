import sqlite3
import os
from datetime import date


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "lifetwin.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# ============================================================
# CREATE DATABASE TABLE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id TEXT NOT NULL,

            record_date TEXT NOT NULL,

            sleep_hours REAL,
            sleep_quality REAL,
            wake_time_consistency REAL,
            study_work_hours REAL,
            focus_sessions INTEGER,
            screen_time_hours REAL,
            social_media_hours REAL,
            notification_interruptions INTEGER,
            exercise_minutes REAL,
            step_count INTEGER,
            stress_level REAL,
            mood_score REAL,
            energy_level REAL,
            caffeine_intake REAL,
            break_minutes REAL,

            age_group TEXT,
            day_of_week TEXT,

            productivity_score REAL,
            productive_class INTEGER,
            actual_productivity REAL

        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# SAVE DAILY RECORD
# ============================================================

def save_daily_record(user_id, user_input, prediction):

    connection = get_connection()
    cursor = connection.cursor()

    # Check whether this user already has a record for today
    cursor.execute("""
        SELECT id
        FROM daily_records
        WHERE user_id = ?
        AND record_date = ?
    """, (
        user_id,
        str(date.today())
    ))

    existing_record = cursor.fetchone()

    if existing_record:

        # Update today's existing record
        cursor.execute("""
            UPDATE daily_records
            SET
                sleep_hours = ?,
                sleep_quality = ?,
                wake_time_consistency = ?,
                study_work_hours = ?,
                focus_sessions = ?,
                screen_time_hours = ?,
                social_media_hours = ?,
                notification_interruptions = ?,
                exercise_minutes = ?,
                step_count = ?,
                stress_level = ?,
                mood_score = ?,
                energy_level = ?,
                caffeine_intake = ?,
                break_minutes = ?,
                age_group = ?,
                day_of_week = ?,
                productivity_score = ?,
                productive_class = ?
            WHERE id = ?
        """, (

            user_input["sleep_hours"],
            user_input["sleep_quality"],
            user_input["wake_time_consistency"],
            user_input["study_work_hours"],
            user_input["focus_sessions"],
            user_input["screen_time_hours"],
            user_input["social_media_hours"],
            user_input["notification_interruptions"],
            user_input["exercise_minutes"],
            user_input["step_count"],
            user_input["stress_level"],
            user_input["mood_score"],
            user_input["energy_level"],
            user_input["caffeine_intake"],
            user_input["break_minutes"],
            user_input["age_group"],
            user_input["day_of_week"],

            prediction["productivity_score"],
            prediction["productive_class"],

            existing_record[0]
        ))

    else:

        # Create a new record for a new day
        cursor.execute("""
            INSERT INTO daily_records (

                user_id,
                record_date,

                sleep_hours,
                sleep_quality,
                wake_time_consistency,
                study_work_hours,
                focus_sessions,
                screen_time_hours,
                social_media_hours,
                notification_interruptions,
                exercise_minutes,
                step_count,
                stress_level,
                mood_score,
                energy_level,
                caffeine_intake,
                break_minutes,

                age_group,
                day_of_week,

                productivity_score,
                productive_class

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            user_id,
            str(date.today()),

            user_input["sleep_hours"],
            user_input["sleep_quality"],
            user_input["wake_time_consistency"],
            user_input["study_work_hours"],
            user_input["focus_sessions"],
            user_input["screen_time_hours"],
            user_input["social_media_hours"],
            user_input["notification_interruptions"],
            user_input["exercise_minutes"],
            user_input["step_count"],
            user_input["stress_level"],
            user_input["mood_score"],
            user_input["energy_level"],
            user_input["caffeine_intake"],
            user_input["break_minutes"],

            user_input["age_group"],
            user_input["day_of_week"],

            prediction["productivity_score"],
            prediction["productive_class"]
        ))

    connection.commit()
    connection.close()


# ============================================================
# GET USER HISTORY
# ============================================================

def get_user_history(user_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM daily_records
        WHERE user_id = ?
        ORDER BY record_date ASC
    """, (user_id,))

    records = cursor.fetchall()

    connection.close()

    return records


# ============================================================
# COUNT USER HISTORY
# ============================================================

def get_history_count(user_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM daily_records
        WHERE user_id = ?
    """, (user_id,))

    count = cursor.fetchone()[0]

    connection.close()

    return count

# ============================================================
# UPDATE ACTUAL PRODUCTIVITY
# ============================================================

def update_actual_productivity(user_id, record_date, actual_productivity):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        UPDATE daily_records
        SET actual_productivity = ?
        WHERE user_id = ?
        AND record_date = ?
    """, (
        actual_productivity,
        user_id,
        record_date
    ))

    connection.commit()

    rows_updated = cursor.rowcount

    connection.close()

    return rows_updated

def get_completed_history(user_id, before_date=None):

    connection = get_connection()
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    query = """
        SELECT
            record_date,

            sleep_hours,
            sleep_quality,
            wake_time_consistency,
            study_work_hours,
            focus_sessions,
            screen_time_hours,
            social_media_hours,
            notification_interruptions,
            exercise_minutes,
            step_count,
            stress_level,
            mood_score,
            energy_level,
            caffeine_intake,
            break_minutes,

            productivity_score,
            actual_productivity

        FROM daily_records

        WHERE user_id = ?
        AND actual_productivity IS NOT NULL
    """

    params = [user_id]

    if before_date:

        query += """
            AND record_date < ?
        """

        params.append(before_date)

    query += """
        ORDER BY record_date ASC
    """

    cursor.execute(
        query,
        params
    )

    records = [
        dict(row)
        for row in cursor.fetchall()
    ]

    connection.close()

    return records