# StudentIQ — Student Performance Management and Analytics

StudentIQ keeps the original Flask/scikit-learn mathematics estimator and model transparency pages, and adds a database-backed student management system. It supports individual student accounts, teacher/admin CRUD, class summaries, role checks, and SQLAlchemy persistence.

## Project files

- `app.py` — existing prediction and model metrics API plus authentication, role-gated student management, validation, dashboards, and CLI commands.
- `models.py` — SQLAlchemy `User`, `Student`, and `Prediction` models and their relationships.
- `database.py` — SQLAlchemy setup and `create_all` initialization, configurable by `DATABASE_URL`.
- `templates/index.html` — original prediction dashboard, now with account navigation and CSRF protection.
- `templates/auth.html` — student registration and sign-in.
- `templates/student.html` — private student dashboard / staff student view.
- `templates/teacher.html` — searchable, sortable, filterable class dashboard with CRUD actions and class metrics.
- `templates/student_form.html` — staff student and assessment entry form.
- `static/style.css` — existing StudentIQ styling, extended for the new pages.
- `artifacts/`, `data/`, `train.py` — existing model, training data, and training pipeline.
- `render.yaml` — Render web service blueprint with secret generation and a production database URL setting.
- `.env.example` — environment variable template. Copy to `.env` for local overrides; do not commit secrets.
- `tests/` — tests for the retained predictor/API behavior.

## Install and run locally (Windows PowerShell)

Use Python 3.11 or newer.

From the folder containing `app.py`:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Replace SECRET_KEY in .env with output from: python -c "import secrets; print(secrets.token_hex(32))"
python train.py
flask --app app init-db
python app.py
```

Open <http://127.0.0.1:5000>. The database is created at `instance/studentiq.db` by default. You can also use `flask --app app run`.

Run `python -m pytest -q` to check the retained predictor routes and account/authorization behavior. The app creates tables on startup so the existing `python app.py` and Gunicorn workflows continue to work. `flask --app app init-db` is an explicit setup command. `create_all` creates missing tables; it does not alter existing tables. For future schema changes, use Alembic/Flask-Migrate migrations rather than relying on `create_all` to upgrade columns.

### First admin/teacher account

Create the first staff account from the project root:

```powershell
flask --app app create-admin
```

Enter an email and a password of at least 10 characters at the prompts. The password is hashed with Werkzeug and is never stored as plaintext. Staff accounts are created only through this CLI (student self-registration always assigns the `student` role).

### Students and privacy

- Students can register at `/register`, or a teacher can create their account in the teacher dashboard.
- Students sign in at `/login` and see their own `/student` record. They cannot choose a role or access another student by changing a URL.
- Teachers/admins sign in at `/login`; successful staff sign-in opens `/teacher`.
- Staff can add, search, filter, sort, inspect, update, and delete student records, including marks and attendance.
- Deleting a student also deletes that student’s login. Role checks guard staff routes, and CSRF tokens protect browser form submissions. Passwords use secure hashes; session cookies are HTTP-only and SameSite Lax.
- Public student registration creates only the user’s own basic record. Academic marks and attendance are entered by staff.

Each student record includes name, email, course, branch, semester, roll number, attendance, assignment/test/midterm/final marks, reading/writing scores, and timestamps. Fields absent from the original estimator (branch, semester, roll, attendance, and assessments) are stored as management data. Overall performance is the average of recorded assignment, test, midterm, and final marks; where none are recorded, the dashboard may show the estimator score if available. Grade bands are A (90+), B (80–89), C (70–79), D (60–69), F (below 60). Missing assessments are not treated as zero.

The original predictor at `/` and `/api/predict`, history endpoint `/api/history`, metrics endpoint `/api/metrics`, and `/health` remain available. The predictor is an educational estimate and is not an assessment grade. Existing anonymous prediction history remains shared as it was before; it contains prediction input context without account identity. `POST /history/clear` is limited to authenticated staff because production history is centralized.

### Legacy prediction history

The new SQLAlchemy database stores predictions. If the project has an older `artifacts/predictions.db`, import its rows once after initialization:

```powershell
flask --app app migrate-legacy
```

The command refuses to import when the new predictions table already contains records, preventing accidental duplicate imports.

## Centralized production database

SQLite is suitable for local development. A local SQLite file is not suitable for multiple application instances or durable production data on ephemeral serverless filesystems. Vercel’s serverless filesystem is not a shared durable database; this project does not include Vercel deployment configuration. Use a hosted PostgreSQL database such as Neon, Supabase, or Render PostgreSQL, and set `DATABASE_URL` to its connection URL in the hosting provider’s environment settings. The included `psycopg[binary]` driver supports PostgreSQL through SQLAlchemy. The URL is read from the environment and must never be committed to source control.

For Render, create a PostgreSQL database in Render, copy its **internal** connection URL into the web service’s `DATABASE_URL` environment variable, and deploy the web service from this project. `render.yaml` generates `SECRET_KEY` and enables secure cookies; `DATABASE_URL` is marked for provider-side configuration. Render’s build command installs requirements and trains the existing model; Gunicorn serves the app. Ensure the PostgreSQL database and web service are in the same region when possible, and back up the database through the provider.

For another provider, define a random `SECRET_KEY`, `COOKIE_SECURE=true`, and a PostgreSQL SQLAlchemy URL, for example `postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME`. Restrict database network access and keep credentials in the provider’s secret manager. Initialize tables by running `flask --app app init-db` as a release/setup command before serving traffic. Do not put production credentials in `.env.example` or source code.

## Existing ML model

`python train.py` trains and compares five regression pipelines on `data/stud.csv` and writes the selected model and metrics to `artifacts/`. The model predicts mathematics scores from the original student-context and reading/writing features. It is kept separate from teacher-entered marks so prediction estimates are not silently treated as official grades.

