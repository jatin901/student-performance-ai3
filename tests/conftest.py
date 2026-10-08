import pytest
import app as student_app


@pytest.fixture
def client(tmp_path):
    db = student_app.db
    with student_app.app.app_context():
        db.session.remove()
        db.engine.dispose()
        db.engines.pop(student_app.app, None)
        student_app.app.config.update(
            TESTING=True,
            WTF_CSRF_ENABLED=False,
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{(tmp_path / 'studentiq-test.db').as_posix()}",
        )
        db.create_all()
    with student_app.app.test_client() as test_client:
        yield test_client
    with student_app.app.app_context():
        db.session.remove()
        db.drop_all()
        db.engine.dispose()
        db.engines.pop(student_app.app, None)
