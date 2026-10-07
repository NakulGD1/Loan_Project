from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "Loan_default.csv"
TARGET = "Default"

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]
IMPORTANT_FEATURES = ["DTIRatio", "LoanAmount", "Income", "Age", "InterestRate"]
CATEGORICAL_FEATURES: list[str] = []
FEATURES = IMPORTANT_FEATURES
FEATURE_SELECTION_COUNT = 5
MODEL_BUNDLE_VERSION = 4
DEFAULT_THRESHOLD = 0.9
DEFAULT_MODEL = "decision_tree"
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
}
MODEL_PATHS = {
    "logistic_regression": ROOT / "loan_logistic_regression.joblib",
    "decision_tree": ROOT / "loan_decision_tree.joblib",
}
MODEL_PATH = MODEL_PATHS[DEFAULT_MODEL]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_pipeline(
    max_depth: int = 8,
    model_name: str = DEFAULT_MODEL,
    features: list[str] | None = None,
) -> Pipeline:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported model: {model_name}")

    selected_features = features if features is not None else FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler() if model_name == "logistic_regression" else "passthrough",
                selected_features,
            ),
        ]
    )
    classifier = (
        LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")
        if model_name == "logistic_regression"
        else DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=20,
            random_state=42,
        )
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def pca_analysis(
    data: pd.DataFrame,
    target: pd.Series | None = None,
    components: int = 3,
) -> dict[str, Any]:
    component_count = min(len(NUMERIC_FEATURES), len(data))
    if component_count < 1:
        raise ValueError("PCA requires at least one row and one component")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[NUMERIC_FEATURES])
    pca = PCA(n_components=component_count, random_state=42)
    component_scores = pca.fit_transform(scaled)
    if target is None:
        contribution = (
            pca.components_[:components] ** 2
            * pca.explained_variance_ratio_[:components, None]
        ).sum(axis=0)
    else:
        component_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="liblinear",
        ).fit(component_scores, target)
        projected_coefficients = (
            component_model.coef_[0] @ pca.components_
        )
        contribution = projected_coefficients**2

    ranking = pd.DataFrame(
        {"feature": NUMERIC_FEATURES, "contribution": contribution}
    )
    ranking = ranking.sort_values("contribution", ascending=False).reset_index(drop=True)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_[:components].tolist(),
        "ranking": ranking,
    }


def train_model(
    model_name: str = DEFAULT_MODEL,
    max_depth: int = 8,
    save: bool = True,
) -> dict[str, Any]:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported model: {model_name}")

    data = load_data()
    candidate_features = (
        NUMERIC_FEATURES if model_name == "logistic_regression" else FEATURES
    )
    X = data[candidate_features]
    # The dataset labels default risk, while the app reports approval likelihood.
    y = (data[TARGET] == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if model_name == "logistic_regression":
        pca = pca_analysis(
            X_train,
            target=y_train,
        )
        validation_features = (
            pca["ranking"]
            .head(FEATURE_SELECTION_COUNT)["feature"]
            .tolist()
        )

        validation_model = build_pipeline(
            max_depth=max_depth,
            model_name=model_name,
            features=validation_features,
        )
        validation_model.fit(X_train[validation_features], y_train)
        predictions = validation_model.predict(X_test[validation_features])
        probabilities = validation_model.predict_proba(X_test[validation_features])[:, 1]

        full_data = X.assign(**{TARGET: y})
        pca = pca_analysis(full_data, target=y)
        features = (
            pca["ranking"]
            .head(FEATURE_SELECTION_COUNT)["feature"]
            .tolist()
        )
        pipeline = build_pipeline(
            max_depth=max_depth,
            model_name=model_name,
            features=features,
        )
        pipeline.fit(X[features], y)
        fit_rows = len(X)
    else:
        features = FEATURES
        pca = pca_analysis(data)
        pipeline = build_pipeline(max_depth=max_depth, model_name=model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        fit_rows = len(X_train)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_name": model_name,
        "fit_rows": fit_rows,
        "features": features,
        "validation_features": (
            validation_features
            if model_name == "logistic_regression"
            else features
        ),
    }
    if model_name == "decision_tree":
        metrics["max_depth"] = max_depth
    model = {
        "pipeline": pipeline,
        "metrics": metrics,
        "features": features,
        "model_name": model_name,
        "pca": pca,
        "bundle_version": MODEL_BUNDLE_VERSION,
    }
    if save:
        joblib.dump(model, MODEL_PATHS[model_name])
        load_model.cache_clear()
    return model


@lru_cache(maxsize=3)
def load_model(model_name: str | None = None) -> dict[str, Any]:
    if model_name is not None:
        if model_name not in MODEL_LABELS:
            raise ValueError(f"Unsupported model: {model_name}")
        path = MODEL_PATHS[model_name]
        if not path.exists():
            train_model(model_name=model_name)
        model = joblib.load(path)
        if model.get("bundle_version") != MODEL_BUNDLE_VERSION:
            model = train_model(model_name=model_name)
        return model

    models = {name: load_model(name) for name in MODEL_LABELS}
    default_bundle = models[DEFAULT_MODEL]
    return {
        "models": models,
        "default_model": DEFAULT_MODEL,
        "pipeline": default_bundle["pipeline"],
        "metrics": default_bundle["metrics"],
        "features": default_bundle["features"],
        "pca": default_bundle["pca"],
        "bundle_version": MODEL_BUNDLE_VERSION,
    }


def model_bundle(bundle: dict[str, Any], model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported model: {model_name}")
    return bundle["models"][model_name]


def predict_one(
    values: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported model: {model_name}")
    bundle = load_model()
    selected_model = model_bundle(bundle, model_name)
    missing_features = [
        feature
        for feature in selected_model["features"]
        if values.get(feature) is None
    ]
    if missing_features:
        raise ValueError(
            "Missing values for model features: " + ", ".join(missing_features)
        )
    frame = pd.DataFrame([values])[selected_model["features"]]
    pipeline = selected_model["pipeline"]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(probability >= threshold)
    result = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "loan_status": prediction,
        "approval_probability": probability,
        "threshold": threshold,
        "decision": "Likely approved" if prediction == 1 else "Likely declined",
    }
    if model_name == "decision_tree":
        tree = pipeline.named_steps["classifier"]
        transformed = pipeline.named_steps["preprocessor"].transform(frame)
        leaf = int(tree.apply(transformed)[0])
        leaf_counts = tree.tree_.value[leaf, 0]
        leaf_samples = int(tree.tree_.n_node_samples[leaf])
        leaf_rate = float(leaf_counts[1] / leaf_counts.sum()) if leaf_counts.sum() else probability
        result.update(
            {
                "tree_leaf": leaf,
                "leaf_samples": leaf_samples,
                "leaf_approval_rate": leaf_rate,
            }
        )
    if (
        values.get("Income") is not None
        and values.get("DTIRatio") is not None
        and values["Income"] >= 150000
        and values["DTIRatio"] <= 0.3
        and prediction == 0
    ):
        result["data_warning"] = "This result reflects patterns in the supplied training data and is not a universal credit rule."
    return result
