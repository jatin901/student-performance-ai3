import os
from pathlib import Path
from models import db


def configure_database(app):
    default_path = Path(app.instance_path) / "studentiq.db"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get("DATABASE_URL", f"sqlite:///{default_path.as_posix()}")
    # Render and some providers still expose postgres:// URLs.
    if uri.startswith("postgres://"):
        uri = "postgresql+psycopg://" + uri[len("postgres://"):]
    elif uri.startswith("postgresql://"):
        uri = "postgresql+psycopg://" + uri[len("postgresql://"):]
    app.config["SQLALCHEMY_DATABASE_URI"] = uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)


def init_db():
    db.create_all()
