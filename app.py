"""
app.py - Flask application entry point for the Student Performance
Analysis & Prediction System.
"""

import os
import re
import sys
from functools import wraps
from datetime import timedelta
from collections import Counter, defaultdict
from statistics import median
from math import isfinite
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash
from authlib.integrations.flask_client import OAuth
from mysql.connector.errors import IntegrityError
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import db  # noqa: E402
from predict import get_model_info, predict_single  # noqa: E402
from feature_engineering import categorize_performance  # noqa: E402

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.getenv("SECRET_KEY", "dev-key-change-me"))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_ENV") == "production"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
_legacy_admin_password = os.getenv("ADMIN_PASSWORD")
ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    generate_password_hash(_legacy_admin_password) if _legacy_admin_password else
    "scrypt:32768:8:1$Gu8VDDmvRhao70AL$e87fc9c4b63c85acaefbf193802c7ce7c3a233094f71424871bf9fb0c37e52400aa3eb742dab48524bb8e03ebd39f082d79579d41011d139c7788a919b792948",
)

VALID_GENDERS = {"Male", "Female", "Other"}
VALID_PARENTAL_EDUCATION = {"High School", "Bachelors", "Graduate"}
VALID_EXTRACURRICULAR = {"Yes", "No"}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def _log_user_in(user, remember=False):
    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_email"] = user["email"]
    session["profile_picture"] = user.get("profile_picture")
    session.permanent = remember


def _valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))


def _number(value):
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_analysis_context(students, filters):
    """Build filtered KPI and chart data from the columns in the students table."""
    def matches(student):
        if filters["gender"] and student.get("gender") != filters["gender"]:
            return False
        if filters["parental_education"] and student.get("parental_education") != filters["parental_education"]:
            return False
        if filters["extracurricular"] and student.get("extracurricular_activity") != filters["extracurricular"]:
            return False
        attendance = _number(student.get("attendance"))
        score = _number(student.get("previous_score"))
        if filters["attendance_min"] is not None and (attendance is None or attendance < filters["attendance_min"]):
            return False
        if filters["attendance_max"] is not None and (attendance is None or attendance > filters["attendance_max"]):
            return False
        if filters["score_min"] is not None and (score is None or score < filters["score_min"]):
            return False
        if filters["score_max"] is not None and (score is None or score > filters["score_max"]):
            return False
        return True

    filtered = [student for student in students if matches(student)]
    scores = [_number(student.get("previous_score")) for student in filtered]
    scores = [score for score in scores if score is not None]
    attendance = [_number(student.get("attendance")) for student in filtered]
    attendance = [value for value in attendance if value is not None]
    study_time = [_number(student.get("study_time")) for student in filtered]
    study_time = [value for value in study_time if value is not None]
    total = len(filtered)

    def average(values):
        return round(sum(values) / len(values), 1) if values else 0

    categories = [categorize_performance(score) for score in scores]
    category_counts = Counter(categories)
    score_ranges = [("0-39", 0, 39), ("40-49", 40, 49), ("50-59", 50, 59),
                    ("60-69", 60, 69), ("70-79", 70, 79), ("80-89", 80, 89), ("90-100", 90, 100)]
    score_distribution = [sum(1 for score in scores if low <= score <= high) for _, low, high in score_ranges]
    attendance_ranges = [("0-59", 0, 59), ("60-69", 60, 69), ("70-79", 70, 79),
                         ("80-89", 80, 89), ("90-100", 90, 100)]
    attendance_distribution = [sum(1 for value in attendance if low <= value <= high) for _, low, high in attendance_ranges]

    def grouped_average(field):
        groups = defaultdict(list)
        for student in filtered:
            score = _number(student.get("previous_score"))
            group = student.get(field)
            if score is not None and group:
                groups[str(group)].append(score)
        labels = sorted(groups)
        return labels, [average(groups[label]) for label in labels]

    gender_labels, gender_averages = grouped_average("gender")
    parent_labels, parent_averages = grouped_average("parental_education")
    scatter = [{"x": _number(student.get("attendance")), "y": _number(student.get("previous_score"))} for student in filtered]
    scatter = [point for point in scatter if point["x"] is not None and point["y"] is not None]
    insights = []
    if scores:
        passed = sum(1 for score in scores if score >= 50)
        insights.append(f"The average previous score is {average(scores):.1f}, with a median of {median(scores):.1f}.")
        insights.append(f"{passed} of {len(scores)} scored at least 50 ({passed / len(scores) * 100:.1f}%).")
    if attendance:
        insights.append(f"Average attendance is {average(attendance):.1f}%, and {sum(1 for value in attendance if value < 75)} students are below the 75% attendance risk threshold.")
    if gender_labels:
        highest_gender = gender_labels[gender_averages.index(max(gender_averages))]
        insights.append(f"{highest_gender} has the highest average previous score among the filtered groups ({max(gender_averages):.1f}).")
    if not insights:
        insights.append("No records match the selected filters yet.")

    return {
        "kpis": {
            "total_students": total,
            "avg_score": average(scores),
            "avg_attendance": average(attendance),
            "pass_percentage": round(sum(1 for score in scores if score >= 50) / len(scores) * 100, 1) if scores else 0,
            "highest_score": round(max(scores), 1) if scores else 0,
            "lowest_score": round(min(scores), 1) if scores else 0,
            "median_score": round(median(scores), 1) if scores else 0,
            "at_risk": sum(1 for student in filtered if (_number(student.get("previous_score")) is not None and _number(student.get("previous_score")) < 50) or (_number(student.get("attendance")) is not None and _number(student.get("attendance")) < 75)),
            "avg_study_time": average(study_time),
        },
        "charts": {
            "score_distribution": {"labels": [label for label, _, _ in score_ranges], "values": score_distribution},
            "attendance_distribution": {"labels": [label for label, _, _ in attendance_ranges], "values": attendance_distribution},
            "categories": {"labels": ["Needs Improvement", "Average", "Good", "Excellent"], "values": [category_counts.get(label, 0) for label in ["Needs Improvement", "Average", "Good", "Excellent"]]},
            "gender": {"labels": gender_labels, "values": gender_averages},
            "parental_education": {"labels": parent_labels, "values": parent_averages},
            "scatter": scatter,
        },
        "insights": insights,
        "has_data": bool(filtered),
    }


def validate_student_form(form) -> list:
    """Validate the Add/Edit Student form. Returns a list of error strings."""
    errors = []
    name = form.get("name", "").strip()
    if not name:
        errors.append("Name is required.")

    try:
        age = int(form.get("age", ""))
        if not (5 <= age <= 100):
            errors.append("Age must be between 5 and 100.")
    except ValueError:
        errors.append("Age must be a whole number.")

    if form.get("gender") not in VALID_GENDERS:
        errors.append(f"Gender must be one of {sorted(VALID_GENDERS)}.")

    for field, low, high in [
        ("study_time", 0, 24), ("attendance", 0, 100),
        ("previous_score", 0, 100), ("assignment_score", 0, 100),
    ]:
        try:
            val = float(form.get(field, ""))
            if not (low <= val <= high):
                errors.append(f"{field.replace('_', ' ').title()} must be between {low} and {high}.")
        except ValueError:
            errors.append(f"{field.replace('_', ' ').title()} must be a number.")

    if form.get("parental_education") not in VALID_PARENTAL_EDUCATION:
        errors.append(f"Parental education must be one of {sorted(VALID_PARENTAL_EDUCATION)}.")

    if form.get("extracurricular_activity") not in VALID_EXTRACURRICULAR:
        errors.append("Extracurricular activity must be Yes or No.")

    return errors


def form_to_student_dict(form) -> dict:
    return {
        "name": form.get("name", "").strip(),
        "age": int(form.get("age")),
        "gender": form.get("gender"),
        "study_time": float(form.get("study_time")),
        "attendance": float(form.get("attendance")),
        "previous_score": float(form.get("previous_score")),
        "assignment_score": float(form.get("assignment_score")),
        "parental_education": form.get("parental_education"),
        "extracurricular_activity": form.get("extracurricular_activity"),
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("username", request.form.get("email", "")).strip().lower()
        password = request.form.get("password", "")
        try:
            user = db.get_user_by_email(email)
        except Exception:
            user = None
        valid_password = user and user.get("password_hash") and check_password_hash(user["password_hash"], password)
        legacy_login = email == ADMIN_USERNAME.lower() and check_password_hash(ADMIN_PASSWORD_HASH, password)
        if valid_password:
            _log_user_in(user, request.form.get("remember") == "on")
            flash("Login successful.", "success")
            return redirect(request.args.get("next") or url_for("index"))
        if legacy_login:
            _log_user_in({"id": "legacy-admin", "name": "Administrator", "email": email}, request.form.get("remember") == "on")
            flash("Login successful.", "success")
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        errors = []
        if not name:
            errors.append("Please enter your name.")
        if not _valid_email(email):
            errors.append("Please enter a valid email address.")
        if len(password) < 8:
            errors.append("Password must contain at least 8 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if not errors:
            try:
                if db.get_user_by_email(email):
                    errors.append("An account with this email already exists.")
                else:
                    user_id = db.create_user(name, email, generate_password_hash(password))
                    _log_user_in({"id": user_id, "name": name, "email": email}, True)
                    flash("Your account has been created.", "success")
                    return redirect(url_for("index"))
            except IntegrityError:
                app.logger.exception("Registration failed because the email already exists: %s", email)
                errors.append("An account with this email already exists. Please sign in.")
            except Exception:
                app.logger.exception("Registration failed for email: %s", email)
                errors.append("We could not create your account right now. Please try again.")
        for error in errors:
            flash(error, "error")
    return render_template("register.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        flash("Password reset requires email service configuration. Please contact the administrator.", "info")
    return render_template("forgot_password.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    try:
        user = db.get_user_by_email(session.get("user_email", ""))
    except Exception:
        user = None
    if not user:
        flash("Your profile is temporarily unavailable.", "error")
        return redirect(url_for("index"))
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            provider = "both" if user.get("google_id") else "local"
            db.update_user_auth(user["id"], auth_provider=provider,
                                password_hash=generate_password_hash(password))
            flash("Password updated successfully.", "success")
            return redirect(url_for("profile"))
    return render_template("profile.html", user=user)


@app.route("/auth/google")
def google_login():
    if not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        flash("Google sign-in is not configured yet.", "error")
        return redirect(url_for("login"))
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    if not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv("GOOGLE_CLIENT_SECRET"):
        flash("Google sign-in is not configured yet.", "error")
        return redirect(url_for("login"))
    if request.args.get("error"):
        if request.args.get("error") == "access_denied":
            flash("Google sign-in was cancelled.", "error")
        else:
            app.logger.warning("Google OAuth callback returned error: %s", request.args.get("error"))
            flash("Unable to sign in with Google. Please try again.", "error")
        return redirect(url_for("login"))
    try:
        token = google.authorize_access_token()
        profile = token.get("userinfo") or google.userinfo()
        google_id = str(profile["sub"])
        email = profile.get("email", "").lower()
        if not email or not profile.get("email_verified", False):
            raise ValueError("Google did not return a verified email address.")
        google_user = db.get_user_by_google_id(google_id)
        email_user = db.get_user_by_email(email)
        if google_user and email_user and google_user["id"] != email_user["id"]:
            app.logger.error("Google identity collision for verified email: %s", email)
            flash("This Google account is linked to a different account. Please contact support.", "error")
            return redirect(url_for("login"))
        user = google_user or email_user
        if user:
            provider = "both" if user.get("password_hash") else "google"
            db.update_user_auth(user["id"], google_id, profile.get("picture"), provider)
            user.update({"google_id": google_id, "profile_picture": profile.get("picture"), "auth_provider": provider})
        else:
            user_id = db.create_user(profile.get("name") or email.split("@", 1)[0], email,
                                     google_id=google_id, profile_picture=profile.get("picture"),
                                     auth_provider="google")
            user = {"id": user_id, "name": profile.get("name") or email, "email": email,
                    "profile_picture": profile.get("picture")}
        _log_user_in(user, True)
        flash("Google sign-in successful.", "success")
        return redirect(url_for("index"))
    except Exception:
        app.logger.exception("Google OAuth callback failed")
        flash("Google sign-in could not be completed. Please try again.", "error")
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def index():
    try:
        stats = db.get_dashboard_stats()
        db_error = None
    except Exception as e:
        stats = {"total_students": 0, "avg_score": 0, "avg_attendance": 0, "pass_percentage": 0}
        db_error = str(e)
    return render_template("index.html", stats=stats, db_error=db_error)


@app.route("/")
def root():
    if session.get("user_id"):
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/students")
@login_required
def students():
    search = request.args.get("q", "").strip()
    try:
        rows = db.search_students(search) if search else db.get_all_students()
        db_error = None
    except Exception as e:
        rows = []
        db_error = str(e)
    return render_template("students.html", students=rows, search=search, db_error=db_error)


@app.route("/add-student", methods=["GET", "POST"])
@login_required
def add_student():
    if request.method == "POST":
        errors = validate_student_form(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("add_student.html", form=request.form)
        try:
            db.add_student(form_to_student_dict(request.form))
            flash("Student added successfully.", "success")
            return redirect(url_for("students"))
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return render_template("add_student.html", form=request.form)
    return render_template("add_student.html", form={})


@app.route("/edit-student/<int:student_id>", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    if request.method == "POST":
        errors = validate_student_form(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("edit_student.html", student={"id": student_id, **request.form})
        try:
            db.update_student(student_id, form_to_student_dict(request.form))
            flash("Student updated successfully.", "success")
            return redirect(url_for("students"))
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return render_template("edit_student.html", student={"id": student_id, **request.form})

    try:
        student = db.get_student(student_id)
    except Exception as e:
        flash(f"Database error: {e}", "error")
        student = None
    if student is None:
        flash("Student not found.", "error")
        return redirect(url_for("students"))
    return render_template("edit_student.html", student=student)


@app.route("/delete-student/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    try:
        db.delete_student(student_id)
        flash("Student deleted.", "success")
    except Exception as e:
        flash(f"Database error: {e}", "error")
    return redirect(url_for("students"))


@app.route("/analysis")
@login_required
def analysis():
    filters = {
        "gender": request.args.get("gender", "").strip(),
        "parental_education": request.args.get("parental_education", "").strip(),
        "extracurricular": request.args.get("extracurricular", "").strip(),
    }
    for field in ("attendance_min", "attendance_max", "score_min", "score_max"):
        filters[field] = _number(request.args.get(field))
    try:
        students = db.get_all_students()
        analysis_data = build_analysis_context(students, filters)
        db_error = None
    except Exception:
        students = []
        analysis_data = build_analysis_context([], filters)
        db_error = "Live student data is currently unavailable."
    return render_template("analysis.html", analysis=analysis_data, filters=filters, filter_options={
        "genders": sorted({str(student.get("gender")) for student in students if student.get("gender")}),
        "parental_education": sorted({str(student.get("parental_education")) for student in students if student.get("parental_education")}),
    }, db_error=db_error)


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None
    form = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        data = {
            "study_time": request.form.get("study_time"),
            "attendance": request.form.get("attendance"),
            "previous_score": request.form.get("previous_score"),
            "assignment_score": request.form.get("assignment_score"),
            "parental_education": request.form.get("parental_education"),
            "extracurricular_activity": request.form.get("extracurricular_activity"),
        }
        try:
            result = predict_single(data)
            try:
                db.save_prediction(
                    None, result["predicted_score"],
                    result["performance_category"], result["regressor_used"],
                )
            except Exception:
                pass  # prediction still shown even if logging to DB fails
        except ValueError as e:
            flash(str(e), "error")
        except FileNotFoundError:
            flash("The trained prediction model is unavailable. Please contact the administrator.", "error")
        except Exception:
            flash("Prediction could not be completed. Please check the input values and try again.", "error")
    try:
        model_info = get_model_info()
    except (FileNotFoundError, KeyError, AttributeError):
        model_info = None
    try:
        recent_predictions = db.get_recent_predictions()
    except Exception:
        recent_predictions = []
    return render_template("prediction.html", result=result, form=form, model_info=model_info, recent_predictions=recent_predictions)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        result = predict_single(data)
        return jsonify({
            "predicted_score": result["predicted_score"],
            "performance_category": result["performance_category"],
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
