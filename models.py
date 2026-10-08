from __future__ import annotations

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin


def utcnow():
    return datetime.now(timezone.utc)


db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="student", index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    student = db.relationship("Student", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Student(db.Model):
    __tablename__ = "students"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    course = db.Column(db.String(120))
    branch = db.Column(db.String(120), index=True)
    semester = db.Column(db.Integer, index=True)
    roll_number = db.Column(db.String(40), unique=True, index=True)
    attendance = db.Column(db.Float)
    assignment_marks = db.Column(db.Float)
    test_marks = db.Column(db.Float)
    midterm_marks = db.Column(db.Float)
    final_marks = db.Column(db.Float)
    reading_score = db.Column(db.Float)
    writing_score = db.Column(db.Float)
    predicted_score = db.Column(db.Float)
    grade = db.Column(db.String(2), index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    user = db.relationship("User", back_populates="student")

    @property
    def overall_performance(self):
        scores = [x for x in (self.assignment_marks, self.test_marks, self.midterm_marks, self.final_marks) if x is not None]
        return round(sum(scores) / len(scores), 1) if scores else self.predicted_score

    def refresh_grade(self):
        score = self.overall_performance
        if score is None:
            self.grade = None
        elif score >= 90:
            self.grade = "A"
        elif score >= 80:
            self.grade = "B"
        elif score >= 70:
            self.grade = "C"
        elif score >= 60:
            self.grade = "D"
        else:
            self.grade = "F"


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    predicted_score = db.Column(db.Float, nullable=False)
    reading_score = db.Column(db.Integer, nullable=False)
    writing_score = db.Column(db.Integer, nullable=False)
    profile = db.Column(db.JSON, nullable=False)


