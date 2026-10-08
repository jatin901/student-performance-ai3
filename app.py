from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

from database import configure_database, init_db
from models import Prediction, Student, User, db

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
ARTIFACTS = BASE / "artifacts"
MODEL_PATH = ARTIFACTS / "student_performance_pipeline.joblib"
METADATA_PATH = ARTIFACTS / "metadata.json"
app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
)
configure_database(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to view this page."
csrf = CSRFProtect(app)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
FEATURES = metadata.get("features", ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "reading_score", "writing_score"])
CATEGORIES = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"],
}


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def admin_required(view):
    from functools import wraps
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user.role not in ("admin", "teacher"):
            flash("Teacher access is required.", "error")
            return redirect(url_for("student_dashboard"))
        return view(*args, **kwargs)
    return wrapped


def valid_email(value):
    value = (value or "").strip().lower()
    if len(value) > 254 or "@" not in value or value.startswith("@") or value.endswith("@"):
        raise ValueError("Enter a valid email address.")
    return value


def validate(payload):
    row = {}
    for key, choices in CATEGORIES.items():
        value = str(payload.get(key, "")).strip()
        if value not in choices:
            raise ValueError(f"Choose a valid value for {key.replace('_', ' ')}.")
        row[key] = value
    for key in ("reading_score", "writing_score"):
        try:
            value = float(payload.get(key, ""))
        except (TypeError, ValueError):
            raise ValueError(f"{key.replace('_', ' ').title()} must be a number from 0 to 100.")
        if not np.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{key.replace('_', ' ').title()} must be between 0 and 100.")
        row[key] = value
    return row


def prediction_details(row):
    if model is None:
        raise RuntimeError("Model artifact is missing. Run python train.py to create it.")
    frame = pd.DataFrame([{key: row[key] for key in FEATURES}])
    score = float(np.clip(model.predict(frame)[0], 0, 100))
    spread = float(metadata.get("residual_std", 5.4))
    low, high = np.clip([score - 1.96 * spread, score + 1.96 * spread], 0, 100)
    if score >= 90: label = "Outstanding"
    elif score >= 80: label = "Excellent"
    elif score >= 70: label = "Very good"
    elif score >= 60: label = "Good"
    elif score >= 50: label = "Developing"
    else: label = "Needs support"
    drivers = [
        {"name": "Reading", "score": row["reading_score"], "note": "Higher reading results are associated with stronger math predictions in this dataset."},
        {"name": "Writing", "score": row["writing_score"], "note": "Writing performance is one of the strongest academic signals in this model."},
    ]
    return {"score": round(score, 1), "label": label, "low": round(float(low), 1), "high": round(float(high), 1), "drivers": drivers}


def save_prediction(row, result):
    pred = Prediction(predicted_score=result["score"], reading_score=int(row["reading_score"]), writing_score=int(row["writing_score"]), profile={k: row[k] for k in CATEGORIES})
    db.session.add(pred)
    db.session.commit()
    return pred.id


def page_context(**extra):
    count = db.session.query(db.func.count(Prediction.id)).scalar() or 0
    avg = db.session.query(db.func.avg(Prediction.predicted_score)).scalar() or 0
    history = Prediction.query.order_by(Prediction.id.desc()).limit(8).all()
    results_path = ARTIFACTS / "model_results.csv"
    comparison = pd.read_csv(results_path).to_dict(orient="records") if results_path.exists() else []
    return {"history": history, "total_predictions": count, "average_prediction": round(avg, 1),
            "metadata": metadata, "comparison": comparison, **extra}


def parse_optional_number(payload, key, low, high, integer=False):
    raw = (payload.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key.replace('_', ' ').title()} must be numeric.")
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key.replace('_', ' ').title()} must be between {low} and {high}.")
    return int(value) if integer else value


def student_fields(payload, student):
    name = (payload.get("name") or "").strip()
    if not name or len(name) > 120:
        raise ValueError("Name is required and must be 120 characters or fewer.")
    student.name = name
    student.email = valid_email(payload.get("email"))
    student.course = (payload.get("course") or "").strip()[:120] or None
    student.branch = (payload.get("branch") or "").strip()[:120] or None
    student.semester = parse_optional_number(payload, "semester", 1, 20, integer=True)
    student.roll_number = (payload.get("roll_number") or "").strip()[:40] or None
    student.attendance = parse_optional_number(payload, "attendance", 0, 100)
    for key in ("assignment_marks", "test_marks", "midterm_marks", "final_marks"):
        setattr(student, key, parse_optional_number(payload, key, 0, 100))
    student.refresh_grade()


def duplicate_error(exc):
    db.session.rollback()
    return "That email or roll number is already in use."


@app.get("/health")
def health():
    try:
        db.session.execute(db.select(User.id).limit(1))
        status = "ok" if model is not None else "model_missing"
        return jsonify({"status": status, "model": metadata.get("best_model", "unknown")}), (200 if model is not None else 503)
    except Exception:
        return jsonify({"status": "database_unavailable"}), 503


@app.get("/api/metrics")
def metrics():
    results_path = ARTIFACTS / "model_results.csv"
    comparison = pd.read_csv(results_path).to_dict(orient="records") if results_path.exists() else []
    return jsonify({"metadata": metadata, "model_comparison": comparison})


@app.post("/api/predict")
@csrf.exempt
def api_predict():
    try:
        row = validate(request.get_json(silent=True) or {})
        result = prediction_details(row)
        result["id"] = save_prediction(row, result)
        return jsonify(result)
    except (ValueError, RuntimeError) as exc:
        db.session.rollback()
        return jsonify({"error": str(exc)}), 400


@app.get("/api/history")
def api_history():
    rows = Prediction.query.order_by(Prediction.id.desc()).limit(50).all()
    return jsonify([{"id": r.id, "created_at": r.created_at.isoformat(), "predicted_score": r.predicted_score, "reading_score": r.reading_score, "writing_score": r.writing_score} for r in rows])


@app.post("/history/clear")
@admin_required
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    return render_template("index.html", **page_context(message="Prediction history cleared."))


@app.route("/", methods=["GET", "POST"])
def home():
    result = error = None
    form = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        try:
            row = validate(request.form)
            result = prediction_details(row)
            result["id"] = save_prediction(row, result)
        except (ValueError, RuntimeError) as exc:
            db.session.rollback()
            error = str(exc)
    return render_template("index.html", **page_context(result=result, error=error, form=form))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("student_dashboard"))
    if request.method == "POST":
        try:
            name = (request.form.get("name") or "").strip()
            email = valid_email(request.form.get("email"))
            password = request.form.get("password", "")
            if not name or len(name) > 120: raise ValueError("Enter your name (up to 120 characters).")
            if len(password) < 10: raise ValueError("Password must be at least 10 characters.")
            user = User(email=email, password_hash=generate_password_hash(password), role="student")
            student = Student(user=user, name=name, email=email)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Your student account is ready.", "success")
            return redirect(url_for("student_dashboard"))
        except ValueError as exc:
            db.session.rollback(); flash(str(exc), "error")
        except Exception:
            db.session.rollback(); flash("An account with that email may already exist.", "error")
    return render_template("auth.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin_dashboard" if current_user.role in ("admin", "teacher") else "student_dashboard"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, request.form.get("password", "")):
            login_user(user)
            return redirect(url_for("admin_dashboard" if user.role in ("admin", "teacher") else "student_dashboard"))
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", mode="login")


@app.post("/logout")
@login_required
def logout():
    logout_user()
    flash("You have signed out.", "success")
    return redirect(url_for("home"))


@app.get("/student")
@login_required
def student_dashboard():
    if current_user.role in ("admin", "teacher"):
        return redirect(url_for("admin_dashboard"))
    student = current_user.student
    if student is None:
        return "Student profile unavailable.", 404
    return render_template("student.html", student=student)


@app.get("/teacher")
@admin_required
def admin_dashboard():
    query = Student.query.join(User)
    search = (request.args.get("q") or "").strip()
    if search:
        like = f"%{search}%"
        query = query.filter(db.or_(Student.name.ilike(like), Student.email.ilike(like), Student.roll_number.ilike(like), Student.branch.ilike(like)))
    branch = request.args.get("branch", "")
    semester = request.args.get("semester", "")
    if branch: query = query.filter(Student.branch == branch)
    if semester.isdigit(): query = query.filter(Student.semester == int(semester))
    sort = request.args.get("sort", "name")
    sort_map = {"name": Student.name, "marks": Student.final_marks, "attendance": Student.attendance, "semester": Student.semester}
    order = sort_map.get(sort, Student.name)
    students = query.order_by(order.desc().nullslast() if sort != "name" else order.asc()).all()
    all_students = Student.query.all()
    marks = [s.overall_performance for s in all_students if s.overall_performance is not None]
    attendances = [s.attendance for s in all_students if s.attendance is not None]
    grades = {g: sum(s.grade == g for s in all_students) for g in ("A", "B", "C", "D", "F")}
    avg_marks = round(sum(marks) / len(marks), 1) if marks else 0
    needing_help = sum((s.overall_performance is not None and s.overall_performance < 60) or (s.attendance is not None and s.attendance < 75) for s in all_students)
    top = max((s for s in all_students if s.overall_performance is not None), key=lambda s: s.overall_performance, default=None)
    return render_template("teacher.html", students=students, branches=sorted({s.branch for s in all_students if s.branch}), semesters=sorted({s.semester for s in all_students if s.semester}), search=search, selected_branch=branch, selected_semester=semester, sort=sort, total=len(all_students), avg_marks=avg_marks, avg_attendance=round(sum(attendances)/len(attendances), 1) if attendances else 0, needing_help=needing_help, top=top, grades=grades)


@app.route("/teacher/students/new", methods=["GET", "POST"])
@admin_required
def create_student():
    student = Student()
    if request.method == "POST":
        try:
            password = request.form.get("password", "")
            if len(password) < 10: raise ValueError("Initial password must be at least 10 characters.")
            student_fields(request.form, student)
            email = student.email
            user = User(email=email, password_hash=generate_password_hash(password), role="student")
            user.student = student
            db.session.add(user); db.session.commit()
            flash("Student account created.", "success")
            return redirect(url_for("admin_dashboard"))
        except ValueError as exc: db.session.rollback(); flash(str(exc), "error")
        except Exception: flash(duplicate_error(None), "error")
    return render_template("student_form.html", student=student, editing=False)


@app.route("/teacher/students/<int:student_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_student(student_id):
    student = db.get_or_404(Student, student_id)
    if request.method == "POST":
        try:
            student_fields(request.form, student)
            student.user.email = student.email
            db.session.commit()
            flash("Student record updated.", "success")
            return redirect(url_for("admin_dashboard"))
        except ValueError as exc: db.session.rollback(); flash(str(exc), "error")
        except Exception: flash(duplicate_error(None), "error")
    return render_template("student_form.html", student=student, editing=True)


@app.post("/teacher/students/<int:student_id>/delete")
@admin_required
def delete_student(student_id):
    student = db.get_or_404(Student, student_id)
    db.session.delete(student.user)
    db.session.commit()
    flash("Student and their login account were deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.get("/teacher/students/<int:student_id>")
@admin_required
def view_student(student_id):
    student = db.get_or_404(Student, student_id)
    return render_template("student.html", student=student, teacher_view=True)


@app.cli.command("init-db")
def init_db_command():
    """Create database tables."""
    with app.app_context(): init_db()
    print("Database tables are ready.")


@app.cli.command("create-admin")
def create_admin_command():
    """Create the first (or another) teacher account interactively."""
    import click
    email = valid_email(click.prompt("Admin email"))
    password = click.prompt("Admin password (10+ characters)", hide_input=True, confirmation_prompt=True)
    if len(password) < 10: raise click.ClickException("Password must be at least 10 characters.")
    with app.app_context():
        if User.query.filter_by(email=email).first(): raise click.ClickException("That email already exists.")
        user = User(email=email, password_hash=generate_password_hash(password), role="admin")
        db.session.add(user); db.session.commit()
    click.echo("Admin account created.")


app.jinja_env.globals["getattr"] = getattr

@app.cli.command("migrate-legacy")
def migrate_legacy_command():
    """Import old raw-SQLite prediction history once, if present."""
    source = ARTIFACTS / "predictions.db"
    if not source.exists():
        print("No legacy prediction database found.")
        return
    with sqlite3.connect(source) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            print("No legacy prediction table found.")
            return
    if Prediction.query.count():
        raise RuntimeError("The new predictions table is not empty; refusing a duplicate import.")
    for row in rows:
        try:
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            created = None
        profile = json.loads(row["profile"])
        db.session.add(Prediction(id=row["id"], created_at=created, predicted_score=row["predicted_score"],
            reading_score=row["reading_score"], writing_score=row["writing_score"], profile=profile))
    db.session.commit()
    print(f"Imported {len(rows)} legacy predictions.")


with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=False)
