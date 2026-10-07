from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loan_model import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    load_model,
    model_bundle,
    predict_one,
)


app = FastAPI(
    title="LoanLens API",
    version="1.0.0",
    description="Loan approval scoring service with decision-tree and logistic-regression models",
)


class LoanApplication(BaseModel):
    Age: float | None = Field(default=None, ge=18, le=100)
    Income: float | None = Field(default=None, ge=0)
    LoanAmount: float | None = Field(default=None, gt=0)
    CreditScore: float | None = Field(default=None, ge=0)
    MonthsEmployed: float | None = Field(default=None, ge=0)
    NumCreditLines: float | None = Field(default=None, ge=0)
    InterestRate: float | None = Field(default=None, ge=0)
    LoanTerm: float | None = Field(default=None, gt=0)
    DTIRatio: float | None = Field(default=None, ge=0)
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0, lt=1)
    model: Literal["logistic_regression", "decision_tree"] = DEFAULT_MODEL


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    bundle = load_model()
    return {
        "default_model": bundle["default_model"],
        "metrics": bundle["metrics"],
        "features": bundle["features"],
        "models": {
            name: {"metrics": model["metrics"], "features": model["features"]}
            for name, model in bundle["models"].items()
        },
    }


@app.post("/predict")
def predict(application: LoanApplication) -> dict[str, Any]:
    values = application.model_dump()
    threshold = values.pop("threshold")
    model_name = values.pop("model")
    values = {feature: value for feature, value in values.items() if value is not None}
    selected_model = model_bundle(load_model(), model_name)
    missing_features = [
        feature
        for feature in selected_model["features"]
        if feature not in values
    ]
    if missing_features:
        raise HTTPException(
            status_code=422,
            detail="Provide values for the selected model features: "
            + ", ".join(missing_features),
        )
    return predict_one(values, threshold=threshold, model_name=model_name)