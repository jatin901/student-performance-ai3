import json


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student performance profile" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert "metadata" in response.get_json()


def test_prediction_api_and_history(client):
    payload = {
        "gender": "female", "race_ethnicity": "group C",
        "parental_level_of_education": "some college", "lunch": "standard",
        "test_preparation_course": "completed", "reading_score": 78,
        "writing_score": 81,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert 0 <= result["score"] <= 100
    assert result["low"] <= result["high"]
    assert result["id"] > 0
    assert len(client.get("/api/history").get_json()) == 1
    history_page = client.get("/")
    assert history_page.status_code == 200
    assert b"Prediction history" in history_page.data


def test_invalid_prediction_is_rejected(client):
    payload = {
        "gender": "female", "race_ethnicity": "group C",
        "parental_level_of_education": "some college", "lunch": "standard",
        "test_preparation_course": "completed", "reading_score": 120,
        "writing_score": 81,
    }
    response = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_form_rejects_out_of_range_scores(client):
    response = client.post("/", data={
        "gender": "female", "race_ethnicity": "group C",
        "parental_level_of_education": "some college", "lunch": "standard",
        "test_preparation_course": "none", "reading_score": "-1", "writing_score": "70",
    })
    assert response.status_code == 200
    assert b"must be between 0 and 100" in response.data
from models import User, db
from werkzeug.security import generate_password_hash


def test_student_registration_and_staff_route_protection(client):
    response = client.post("/register", data={"name": "Asha Student", "email": "asha@example.com", "password": "long-secure-pass"})
    assert response.status_code == 302
    assert client.get("/student").status_code == 200
    assert client.get("/teacher").status_code == 302
    client.post("/logout")
    assert client.get("/student").status_code == 302


def test_teacher_can_create_and_update_student(client):
    with __import__("app").app.app_context():
        admin = User(email="teacher@example.com", password_hash=generate_password_hash("long-secure-pass"), role="admin")
        db.session.add(admin)
        db.session.commit()
        client.post("/login", data={"email": "teacher@example.com", "password": "long-secure-pass"})
    response = client.get("/teacher")
    assert response.status_code == 200
    assert b"Teacher dashboard" in response.data
    assert client.get("/teacher/students/new").status_code == 200
    response = client.post("/teacher/students/new", data={
        "name": "Ravi Learner", "email": "ravi@example.com", "password": "student-secure-pass",
        "course": "BSc", "branch": "Computer Science", "semester": "2", "roll_number": "CS-12",
        "attendance": "91", "assignment_marks": "88", "test_marks": "90", "midterm_marks": "84", "final_marks": "92",
    })
    assert response.status_code == 302
    with __import__("app").app.app_context():
        student = __import__("models").Student.query.filter_by(email="ravi@example.com").one()
        assert student.overall_performance == 88.5
        assert student.grade == "B"
        student_id = student.id
    assert client.get(f"/teacher/students/{student_id}").status_code == 200
    assert client.get(f"/teacher/students/{student_id}/edit").status_code == 200
