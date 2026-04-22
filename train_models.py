import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from project_config import DATASETS_DIR, MODEL_FILES, MODELS_DIR


def save_model(model, filename: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / filename).open("wb") as file:
        pickle.dump(model, file)


def train_diabetes() -> None:
    df = pd.read_csv(DATASETS_DIR / "diabetes.csv")
    df["gender"] = df["gender"].astype("string")
    df["smoking_history"] = df["smoking_history"].astype("string")
    df = df[df["gender"] != "Other"].drop_duplicates()

    for feature in ["bmi", "HbA1c_level", "blood_glucose_level"]:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]

    def recategorize_smoking(value: str) -> str:
        if value in ["never", "No Info"]:
            return "non-smoker"
        if value == "current":
            return "current"
        return "past_smoker"

    df["smoking_history"] = df["smoking_history"].apply(recategorize_smoking)
    df_encoded = pd.get_dummies(
        df,
        columns=["gender", "smoking_history"],
        dtype=int,
        drop_first=True,
    )

    X = df_encoded.drop("diabetes", axis=1).values
    y = df_encoded["diabetes"].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [0, 1, 2, 3, 4, 5]),
            ("cat", OneHotEncoder(drop="first"), [6, 7, 8]),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    eval_metric="error",
                    objective="binary:logistic",
                    subsample=0.8,
                    n_estimators=200,
                    min_child_weight=3,
                    max_depth=3,
                    learning_rate=0.1,
                    colsample_bytree=1.0,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    save_model(pipeline, MODEL_FILES["diabetes"])


def train_heart() -> None:
    df = pd.read_csv(DATASETS_DIR / "heart (2).csv")
    X = df.drop(columns="target")
    y = df["target"]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    save_model(model, MODEL_FILES["heart"])


def train_parkinsons() -> None:
    df = pd.read_csv(DATASETS_DIR / "parkinsons.csv")
    df = df.select_dtypes(include=["int64", "float64"])
    X = df.drop(columns=["status"])
    y = df["status"]

    model = AdaBoostClassifier(random_state=42)
    model.fit(X, y)
    save_model(model, MODEL_FILES["parkinsons"])


def train_kidney() -> None:
    df = pd.read_csv(DATASETS_DIR / "kidney.csv")
    columns = ["sg", "al", "sc", "hemo", "pcv", "htn", "classification"]
    df = df[columns].dropna(axis=0).copy()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

    X = df.drop(columns=["classification"])
    y = df["classification"]

    model = SVC(kernel="linear", random_state=42)
    model.fit(X, y)
    save_model(model, MODEL_FILES["kidney"])


if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_parkinsons()
    train_kidney()
    print(f"Saved trained models to {MODELS_DIR}")
