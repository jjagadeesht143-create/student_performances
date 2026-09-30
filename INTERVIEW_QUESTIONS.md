# Interview Questions & Answers

Covers Python, Pandas, NumPy, SQL, Machine Learning, EDA, Feature
Engineering, Train/Test Split, Overfitting/Underfitting, Regression,
Classification, Random Forest, Linear Regression, Evaluation Metrics,
Target Leakage, Flask, REST APIs, MySQL, and Git/GitHub — plus prompts to
practice explaining this specific project.

---

## Python

**1. What is the difference between a list and a tuple in Python?**
A list is mutable (can be changed after creation) and uses `[]`. A tuple is
immutable and uses `()`. Tuples are slightly faster and are used when data
shouldn't change (e.g., a coordinate pair). In this project, `FEATURE_COLUMNS`
is a list because it's iterated and indexed like an ordinary sequence.

**2. What is a Python decorator, and where does this project use one?**
A decorator wraps a function to add behavior without changing its code.
Flask's `@app.route("/api/predict", methods=["POST"])` is a decorator that
registers a function as the handler for that URL and HTTP method.

**3. What's the difference between `==` and `is`?**
`==` compares values; `is` compares object identity (same memory location).
Use `is` for `None` checks (`if x is None`), and `==` for value comparison.

**4. What is a virtual environment and why use one?**
An isolated Python environment with its own installed packages, so one
project's dependencies (e.g., a specific scikit-learn version) don't
conflict with another's. Created with `python -m venv venv`.

**5. Explain `try/except/finally` and how it's used in `db.py`.**
`try` runs code that might fail; `except` catches specific errors; `finally`
always runs (e.g., closing resources). In `db.py`, every function opens a
connection in `try` and closes it in `finally`, guaranteeing the connection
is released even if the query raises an exception.

---

## Pandas & NumPy

**6. What's the difference between `.loc` and `.iloc`?**
`.loc` selects by label (column/index name); `.iloc` selects by integer
position. `df.loc[0, "final_score"]` vs `df.iloc[0, 10]`.

**7. How do you detect and handle missing values in Pandas?**
`df.isnull().sum()` counts missing values per column. Handling depends on
context: drop rows (`dropna`), impute (mean/median/mode), or flag them.
This project checks for missing values explicitly in
`data_quality_report()` before deciding what to do, rather than silently
dropping/filling.

**8. What does `df.describe()` show, and why is `include='all'` useful?**
By default, `describe()` shows count/mean/std/min/max/quartiles for numeric
columns. `include='all'` adds categorical columns too, showing count,
unique, top, and frequency — useful for a fuller pre-cleaning picture.

**9. What's the difference between NumPy arrays and Python lists?**
NumPy arrays are homogeneous (single dtype), stored contiguously in memory,
and support vectorized operations (fast, no explicit loops). Python lists
are heterogeneous and slower for numeric computation.

**10. How would you find and remove duplicate rows in a DataFrame?**
`df.duplicated()` returns a boolean mask; `df.duplicated().sum()` counts
them; `df.drop_duplicates()` removes them. Used in `clean_data()`.

---

## SQL / MySQL

**11. What is a parameterized query and why does this project use them?**
A query where values are passed separately from the SQL string (using `%s`
placeholders), rather than being concatenated into it. This prevents SQL
injection, because user input is never interpreted as SQL syntax. Every
query in `src/db.py` uses this pattern.

**12. What is a foreign key, and where is one used here?**
A column that references the primary key of another table, enforcing
referential integrity. `predictions.student_id` is a foreign key referencing
`students.id`, with `ON DELETE SET NULL` so deleting a student doesn't
orphan-fail a prediction record.

**13. What's the difference between `INNER JOIN` and `LEFT JOIN`?**
`INNER JOIN` returns only rows with matches in both tables. `LEFT JOIN`
returns all rows from the left table, with NULLs for unmatched right-table
columns.

**14. Why use environment variables for database credentials?**
So secrets never live in source code or version control. `.env` (git-
ignored) holds real values locally; `.env.example` documents the required
keys with placeholders.

**15. What's an index, and when would you add one to the `students` table?**
An index speeds up lookups on a column at the cost of extra storage and
slower writes. Adding an index on `students.name` would speed up the
`search_students()` LIKE-query if the table grew large.

---

## Machine Learning / EDA / Feature Engineering

**16. What is target leakage, and how did you prevent it in this project?**
Target leakage happens when information that wouldn't be available at
prediction time — or that's derived from the target itself — ends up as an
input feature, making the model look artificially accurate. In this
project: `final_score` and anything derived from it are never features;
`train_test_split` happens before any preprocessing is fit; imputation/
scaling/encoding live inside a `Pipeline` fit only on the training split;
and the classification labels are derived independently for train/test
*after* the split.

**17. What is the difference between overfitting and underfitting?**
Overfitting: the model learns noise/specifics of the training data and
performs much worse on new data (low train error, high test error).
Underfitting: the model is too simple to capture the real pattern (high
error on both train and test). Comparing CV score to test score for each
model here shows neither is dramatically overfitting.

**18. Why use cross-validation instead of a single train/test split alone?**
A single split's result depends partly on which rows happened to land in
which split. K-fold cross-validation averages performance over several
different splits of the training data, giving a more reliable estimate of
how a model generalizes, before ever touching the final test set.

**19. What is a `ColumnTransformer` and why use one?**
It applies different preprocessing to different columns (e.g., scaling
numeric columns, one-hot encoding categorical columns) within a single,
fit-able object. Used here to build one preprocessor for both feature
types, fit once on the training data.

**20. Why impute missing values with the median instead of the mean for numeric features?**
The median is robust to outliers/skew; a few extreme values can pull the
mean far from the "typical" value, whereas the median is less affected.

**21. What is one-hot encoding, and why not just use label encoding for categorical features here?**
One-hot encoding creates a binary column per category, avoiding an implied
order/magnitude. Label encoding assigns an integer per category, which
would incorrectly imply `Graduate > Bachelors > High School` as a numeric
relationship to a linear/logistic model. `parental_education` isn't
ordinal in a way that's safe to encode as a single number for these models.

**22. How do you decide which features to include as model inputs?**
Only include information genuinely available *before* the prediction is
needed, that has a plausible real-world relationship with the target, and
that isn't a proxy for the target itself. Here: study_time, attendance,
previous_score, assignment_score, parental_education, and
extracurricular_activity qualify; final_score-derived fields do not.

---

## Regression / Classification / Random Forest

**23. Explain Linear Regression in simple terms.**
It fits a straight-line (or hyperplane, with multiple features) relationship
between inputs and a continuous target, minimizing the sum of squared
errors between predictions and actual values.

**24. Explain Random Forest in simple terms.**
It trains many decision trees on random subsets of the data and features,
then averages their predictions (regression) or takes a majority vote
(classification). This reduces the overfitting risk of any single tree.

**25. Why might Linear Regression outperform Random Forest on a small,
noisy, mostly-linear dataset (as happened in this project)?**
If the true relationship between features and target is close to linear
and the dataset is fairly small, a Random Forest can overfit to noise in
the training data and generalize slightly worse, while Linear Regression's
simplicity acts as a form of regularization. That's consistent with the
actual result here (Linear Regression: test R²=0.548 vs Random Forest:
0.495).

**26. What's the difference between regression and classification?**
Regression predicts a continuous numeric value (e.g., `final_score`).
Classification predicts a discrete category (e.g., `performance_category`).
This project does both, from the same input features.

**27. Why use `StratifiedKFold` for classification but plain `KFold` for regression?**
`StratifiedKFold` preserves the proportion of each class in every fold,
which matters when classes are imbalanced (here, "Excellent" and "Needs
Improvement" are rare). Plain `KFold` has no concept of class balance,
which is fine for a continuous regression target.

---

## Evaluation Metrics

**28. Explain MAE, MSE, RMSE, and R² for regression.**
- MAE (Mean Absolute Error): average absolute difference between predicted
  and actual values — easy to interpret in the target's own units.
- MSE (Mean Squared Error): average squared difference — penalizes larger
  errors more heavily.
- RMSE: square root of MSE — back in the target's original units.
- R²: proportion of variance in the target explained by the model (1.0 =
  perfect, 0 = no better than predicting the mean).

**29. Explain Accuracy, Precision, Recall, and F1 Score for classification.**
- Accuracy: fraction of predictions that were correct overall.
- Precision: of predicted-positive cases, how many were actually positive.
- Recall: of actual-positive cases, how many were correctly identified.
- F1: harmonic mean of precision and recall — useful when classes are
  imbalanced, since accuracy alone can be misleading.

**30. What does a confusion matrix show, and why check it beyond accuracy?**
A confusion matrix shows, for each true class, how predictions were
distributed across predicted classes. It reveals *which* classes are
being confused — in this project, "Excellent" and "Needs Improvement" (the
rarest classes) have weak recall, which overall accuracy alone would hide.

---

## Flask / REST API

**31. What is a REST API, and what does `/api/predict` return?**
An architectural style for web services using HTTP methods (GET, POST,
etc.) on resources, typically exchanging JSON. `/api/predict` accepts a
POST request with student data as JSON and returns a JSON object with
`predicted_score` and `performance_category`.

**32. How does this project validate API input, and what happens on invalid input?**
`src/predict.py::validate_input()` checks required fields, numeric ranges,
and allowed categorical values, returning a list of error messages. The
Flask route catches `ValueError` and returns a 400 response with a JSON
`{"error": "..."}` body, rather than crashing or silently guessing.

**33. Why keep the ML model loaded once (module-level) instead of loading it on every request?**
Loading a joblib artifact from disk is relatively slow. Loading it once
when the module first imports (and caching it in a global) means each
request only pays the (fast) prediction cost, not the load cost.

---

## Git / GitHub

**34. What's the difference between `git merge` and `git rebase`?**
`merge` creates a new commit joining two branch histories, preserving both
histories as-is. `rebase` replays one branch's commits on top of another,
producing a linear history — useful for a cleaner log, but rewrites commit
hashes, so it's riskier on shared/public branches.

**35. Why is `.env` in `.gitignore` but `.env.example` is committed?**
`.env` holds real secrets (DB password, Flask secret key) that must never
be pushed to a public repo. `.env.example` documents which variables are
needed, with placeholder values, so anyone cloning the repo knows what to
configure without exposing real credentials.

---

## Project-Specific Prompts

**"Explain this project in 60 seconds."**
This is a student performance analysis and prediction system. I built a
synthetic-but-realistic dataset, cleaned it without leaking test
information into training, ran an EDA to see what actually correlates with
final exam score, then trained and compared two regression models and two
classification models inside leakage-safe scikit-learn pipelines. The best
regression model predicts a numeric score; the best classifier predicts a
performance category. Both are wrapped in a Flask app with a MySQL-backed
student database, a REST API, and a Bootstrap frontend, so a user can add
students, view analysis charts, and get a live prediction with input
validation.

**"Explain this project in 2 minutes."**
Start with the problem: schools collect data but don't always use it
predictively. I generated a synthetic dataset (disclosed clearly as
synthetic) with realistic relationships between study time, attendance,
prior scores, assignment scores, and a final score. Before touching
modeling, I ran a full data-quality check — shape, dtypes, missing values,
duplicates — and cleaned only what was safe to clean pre-split: duplicates
and rows with a missing label. I explicitly avoided imputing or scaling
features at the whole-dataset level, because that would leak test-set
statistics into training; instead, all of that lives inside a
`ColumnTransformer` fit only on the training split. For EDA, I generated
eight charts showing distributions and relationships, and a correlation
heatmap that confirmed study time and previous score are the strongest
predictors. For modeling, I compared Linear Regression against Random
Forest for score prediction, and Logistic Regression against Random Forest
for a four-class performance category, using 5-fold cross-validation on
the training split and a single honest evaluation on a held-out test set.
Linear Regression won on R² (0.548 vs 0.495) and Logistic Regression won
on F1 (0.759 vs 0.724) — both saved together in one joblib artifact so
inference always uses identical preprocessing. Finally, I built a Flask
app around it: CRUD for student records against a MySQL database with
parameterized queries, an analysis page rendering the saved charts, a
prediction form, and a JSON `/api/predict` endpoint with input validation,
all styled with Bootstrap.

**"Explain the project architecture."**
See README section 13. In short: a Bootstrap frontend talks to a Flask
backend; Flask uses `src/predict.py` to load a single joblib artifact
containing both the trained regression and classification pipelines for
live predictions, and uses `src/db.py` (parameterized MySQL queries) for
all student CRUD and prediction logging. Offline, a separate pipeline
(`data_preprocessing.py` → `eda.py` → `feature_engineering.py` →
`train_model.py` → `evaluate_model.py`) turns raw CSV data into that saved
model artifact and the analysis charts the web app displays.

**"What problems did you face?"**
The main challenge was making sure leakage couldn't creep in anywhere —
it's easy to accidentally fit a scaler on the full dataset, or derive a
classification label before splitting. I addressed this by pushing all
data-dependent transformations into a `Pipeline`/`ColumnTransformer` that's
only ever fit on `X_train`, and by deriving the classification labels
separately for the train and test splits after splitting, not before.

**"Why did you choose this model?"**
Not arbitrarily — by comparing documented cross-validation and held-out
test metrics for two candidate models per task, and selecting whichever
scored higher on the test set (R² for regression, weighted F1 for
classification), rather than assuming a more complex model (Random Forest)
would automatically win.

**"How did you prevent target leakage?"**
Four concrete measures: (1) only pre-exam-available columns are used as
features; (2) `train_test_split` happens before any preprocessing is fit;
(3) imputation/scaling/encoding live inside a pipeline fit solely on the
training split; (4) the classification target is derived from
`final_score` independently for train and test, after the split, so no
label information crosses between them.
