"""
db.py

MySQL connection + CRUD helpers for the `students` and `predictions` tables.

Security notes:
  - Credentials are read from environment variables (.env), never hardcoded.
  - Every query uses parameterized placeholders (%s) - never string
    formatting/concatenation - to prevent SQL injection.
"""

import os
try:
    import mysql.connector
except ModuleNotFoundError:
    mysql = None
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "student_performance_db"),
    "port": int(os.getenv("DB_PORT", "3306")),
}


def get_connection():
    """Open a new MySQL connection using env-configured credentials."""
    if mysql is None:
        raise RuntimeError("MySQL connector is not installed.")
    return mysql.connector.connect(**DB_CONFIG)


# ------------------------------------------------------------------
# Users and authentication
# ------------------------------------------------------------------

def get_user_by_email(email: str):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email = %s", (email.lower(),))
        return cur.fetchone()
    finally:
        conn.close()


def get_user_by_google_id(google_id: str):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE google_id = %s", (google_id,))
        return cur.fetchone()
    finally:
        conn.close()


def create_user(name: str, email: str, password_hash=None, google_id=None,
                profile_picture=None, auth_provider="local") -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users
               (name, email, password_hash, google_id, profile_picture, auth_provider)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (name, email.lower(), password_hash, google_id, profile_picture, auth_provider),
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_user_auth(user_id: int, google_id=None, profile_picture=None,
                     auth_provider=None, password_hash=None) -> None:
    fields = []
    values = []
    for field, value in (("google_id", google_id), ("profile_picture", profile_picture),
                         ("auth_provider", auth_provider), ("password_hash", password_hash)):
        if value is not None:
            fields.append(f"{field} = %s")
            values.append(value)
    if not fields:
        return
    values.append(user_id)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = %s", tuple(values))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------
# Students CRUD
# ------------------------------------------------------------------

def add_student(data: dict) -> int:
    query = """
        INSERT INTO students
        (name, age, gender, study_time, attendance, previous_score,
         assignment_score, parental_education, extracurricular_activity)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        data["name"], data["age"], data["gender"], data["study_time"],
        data["attendance"], data["previous_score"], data["assignment_score"],
        data["parental_education"], data["extracurricular_activity"],
    )
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_students() -> list:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM students ORDER BY id ASC")
        return cur.fetchall()
    finally:
        conn.close()


def get_student(student_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM students WHERE id = %s", (student_id,))
        return cur.fetchone()
    finally:
        conn.close()


def update_student(student_id: int, data: dict) -> None:
    query = """
        UPDATE students SET
            name = %s, age = %s, gender = %s, study_time = %s,
            attendance = %s, previous_score = %s, assignment_score = %s,
            parental_education = %s, extracurricular_activity = %s
        WHERE id = %s
    """
    params = (
        data["name"], data["age"], data["gender"], data["study_time"],
        data["attendance"], data["previous_score"], data["assignment_score"],
        data["parental_education"], data["extracurricular_activity"],
        student_id,
    )
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def delete_student(student_id: int) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM students WHERE id = %s", (student_id,))
        conn.commit()
    finally:
        conn.close()


def search_students(keyword: str) -> list:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        like_pattern = f"%{keyword}%"
        cur.execute(
            "SELECT * FROM students WHERE name LIKE %s ORDER BY created_at DESC",
            (like_pattern,),
        )
        return cur.fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------

def save_prediction(student_id, predicted_score, performance_category, model_name) -> int:
    query = """
        INSERT INTO predictions (student_id, predicted_score, performance_category, model_name)
        VALUES (%s, %s, %s, %s)
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, (student_id, predicted_score, performance_category, model_name))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recent_predictions(limit: int = 8) -> list:
    """Return recent predictions with an optional linked student name."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT p.created_at, p.predicted_score, p.performance_category,
                   p.model_name, COALESCE(s.name, 'Ad hoc prediction') AS student_name
            FROM predictions p
            LEFT JOIN students s ON s.id = p.student_id
            ORDER BY p.created_at DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()
    finally:
        conn.close()


def get_dashboard_stats() -> dict:
    """Aggregate stats for the dashboard cards."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT
                COUNT(*) AS total_students,
                AVG(previous_score) AS avg_previous_score,
                AVG(attendance) AS avg_attendance,
                SUM(CASE WHEN previous_score >= 50 THEN 1 ELSE 0 END) AS passing_count
            FROM students
        """)
        row = cur.fetchone() or {}
        total = row.get("total_students") or 0
        pass_pct = (row["passing_count"] / total * 100) if total else 0
        return {
            "total_students": total,
            "avg_score": round(float(row["avg_previous_score"]), 1) if row["avg_previous_score"] else 0,
            "avg_attendance": round(float(row["avg_attendance"]), 1) if row["avg_attendance"] else 0,
            "pass_percentage": round(pass_pct, 1),
        }
    finally:
        conn.close()


def test_connection() -> bool:
    try:
        conn = get_connection()
        conn.close()
        return True
    except Error as e:
        print(f"DB connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing DB connection...")
    print("OK" if test_connection() else "FAILED")
