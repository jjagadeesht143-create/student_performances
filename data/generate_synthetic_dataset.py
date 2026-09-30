"""
generate_synthetic_dataset.py

IMPORTANT / DISCLOSURE
-----------------------
No real-world student dataset was provided for this project. This script
GENERATES a SYNTHETIC dataset for development, learning, and portfolio
purposes only. It is NOT real student data and must never be presented
as such.

The generation logic deliberately builds in realistic, noisy relationships
(e.g. more study time and better attendance tend to raise scores) so that
the downstream EDA and ML pipeline have something meaningful to find,
without ever leaking the target into the features.

Run:
    python data/generate_synthetic_dataset.py
Output:
    data/raw/students.csv
"""

import numpy as np
import pandas as pd
import os

# Fixed seed -> reproducible dataset every time this script is run.
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_STUDENTS = 500

# ---- Base / demographic features -------------------------------------------------
student_id = np.arange(1, N_STUDENTS + 1)
age = rng.integers(15, 19, size=N_STUDENTS)  # typical secondary-school age range
gender = rng.choice(["Male", "Female"], size=N_STUDENTS, p=[0.5, 0.5])

# ---- Pre-outcome, "known before final exam" features ------------------------------
# Hours studied per day (0.5 - 10), roughly bell shaped
study_time = np.clip(rng.normal(4.5, 2.0, N_STUDENTS), 0.5, 10).round(1)

# Attendance percentage (40 - 100)
attendance = np.clip(rng.normal(80, 12, N_STUDENTS), 40, 100).round(1)

# Previous exam score (e.g. last term / midterm) (30 - 100)
previous_score = np.clip(rng.normal(65, 15, N_STUDENTS), 30, 100).round(1)

# Assignment score - average of assignments turned in during the term (0-100)
assignment_score = np.clip(rng.normal(70, 14, N_STUDENTS), 0, 100).round(1)

parental_education = rng.choice(
    ["High School", "Bachelors", "Graduate"],
    size=N_STUDENTS,
    p=[0.4, 0.4, 0.2],
)
extracurricular_activity = rng.choice(["Yes", "No"], size=N_STUDENTS, p=[0.45, 0.55])

# ---- Introduce a small number of missing values (to be handled in cleaning) --------
def add_missing(arr, frac=0.03):
    arr = arr.astype(float)
    n_missing = int(len(arr) * frac)
    idx = rng.choice(len(arr), size=n_missing, replace=False)
    arr[idx] = np.nan
    return arr

attendance = add_missing(attendance, frac=0.02)
assignment_score = add_missing(assignment_score, frac=0.02)

# ---- Target: final_score -----------------------------------------------------------
# Built as a function of the features ABOVE only (no leakage), plus random noise.
# Missing attendance/assignment values are filled with column means only for the
# purpose of generating a plausible target signal here; the actual ML pipeline
# will do its own, leakage-safe imputation later on the train split.
_att_fill = np.nanmean(attendance)
_asg_fill = np.nanmean(assignment_score)
att_for_target = np.where(np.isnan(attendance), _att_fill, attendance)
asg_for_target = np.where(np.isnan(assignment_score), _asg_fill, assignment_score)

parent_bonus = pd.Series(parental_education).map(
    {"High School": 0, "Bachelors": 3, "Graduate": 6}
).to_numpy()
extra_bonus = pd.Series(extracurricular_activity).map({"Yes": 2, "No": 0}).to_numpy()

final_score = (
    0.28 * study_time * 10        # studying matters
    + 0.25 * att_for_target        # attendance matters
    + 0.25 * previous_score        # past performance is a strong predictor
    + 0.20 * asg_for_target        # assignments matter
    + parent_bonus
    + extra_bonus
    + rng.normal(0, 6, N_STUDENTS)  # irreducible noise
)
final_score = np.clip(final_score, 0, 100).round(1)

# A handful of duplicate rows on purpose, to be caught in the cleaning stage.
df = pd.DataFrame({
    "student_id": student_id,
    "name": [f"Student_{i}" for i in student_id],
    "age": age,
    "gender": gender,
    "study_time": study_time,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignment_score": assignment_score,
    "parental_education": parental_education,
    "extracurricular_activity": extracurricular_activity,
    "final_score": final_score,
})

duplicate_rows = df.sample(5, random_state=RANDOM_STATE)
df = pd.concat([df, duplicate_rows], ignore_index=True)

os.makedirs("data/raw", exist_ok=True)
out_path = "data/raw/students.csv"
df.to_csv(out_path, index=False)
print(f"Synthetic dataset written to {out_path} — shape: {df.shape}")
print("NOTE: This is SYNTHETIC data for development/testing only.")
