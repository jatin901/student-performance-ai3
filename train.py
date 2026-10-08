"""Train and compare regression pipelines for the StudentIQ dataset."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
DATA = BASE / "data" / "stud.csv"
TARGET = "math_score"
ARTIFACTS.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
if TARGET not in df or df.empty:
    raise ValueError(f"Dataset must contain a non-empty '{TARGET}' column.")
df = df.dropna(subset=[TARGET]).copy()
X, y = df.drop(columns=[TARGET]), df[TARGET].astype(float)
cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
num = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
models = {
    "Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=400, random_state=42, max_depth=10, min_samples_leaf=2, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=.04, max_depth=3, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=400, random_state=42, max_depth=12, min_samples_leaf=2, n_jobs=-1),
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
results, pipelines = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({"model": name, "mae": mean_absolute_error(y_test, pred), "rmse": np.sqrt(mean_squared_error(y_test, pred)), "r2": r2_score(y_test, pred)})
    pipelines[name] = pipe
result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
best_name = result_df.iloc[0]["model"]
best = pipelines[best_name]
test_pred = best.predict(X_test)
residual_std = float(np.std(y_test.to_numpy() - test_pred, ddof=1))
# Refit winning pipeline on all available records after honest held-out evaluation.
best.fit(X, y)
joblib.dump(best, ARTIFACTS / "student_performance_pipeline.joblib")
result_df.to_csv(ARTIFACTS / "model_results.csv", index=False, float_format="%.4f")
metadata = {"target": TARGET, "best_model": best_name, "features": X.columns.tolist(), "categorical_features": cat,
            "numeric_features": num, "dataset_rows": int(len(df)), "test_size": .2, "random_state": 42,
            "evaluation": {"mae": round(float(result_df.iloc[0]["mae"]), 4), "rmse": round(float(result_df.iloc[0]["rmse"]), 4), "r2": round(float(result_df.iloc[0]["r2"]), 4)},
            "residual_std": round(residual_std, 4), "validation": "80/20 held-out split; seed 42"}
(ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Selected {best_name}; MAE={metadata['evaluation']['mae']:.3f}, RMSE={metadata['evaluation']['rmse']:.3f}, R2={metadata['evaluation']['r2']:.3f}")
