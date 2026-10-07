import pandas as pd
import streamlit as st

from loan_model import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    MODEL_LABELS,
    NUMERIC_FEATURES,
    load_data,
    load_model,
    predict_one,
)


st.title("Prediction lab")
st.caption("Choose a trained model and enter the features it uses for an approval estimate.")

presets = {
    "Balanced applicant": {"age": 30, "income": 60000.0, "amount": 10000.0, "rate": 11.0, "dti": 0.25},
    "Dataset-approved example": {"age": 29, "income": 33164.0, "amount": 12000.0, "rate": 13.23, "dti": 0.30},
    "Higher-risk applicant": {"age": 23, "income": 22000.0, "amount": 18000.0, "rate": 18.0, "dti": 0.75},
}

feature_inputs = {
    "Age": ("Applicant age", 18.0, 100.0, 1.0, True),
    "Income": ("Annual income", 0.0, 1_000_000.0, 1000.0, False),
    "LoanAmount": ("Loan amount", 500.0, 1_000_000.0, 500.0, False),
    "CreditScore": ("Credit score", 300.0, 850.0, 1.0, True),
    "MonthsEmployed": ("Months employed", 0.0, 600.0, 1.0, True),
    "NumCreditLines": ("Number of credit lines", 0.0, 20.0, 1.0, True),
    "InterestRate": ("Interest rate (%)", 0.0, 40.0, 0.1, False),
    "LoanTerm": ("Loan term (months)", 1.0, 120.0, 1.0, True),
    "DTIRatio": ("Debt-to-income ratio", 0.0, 2.0, 0.01, False),
}


@st.cache_data
def get_feature_defaults():
    return load_data()[NUMERIC_FEATURES].median().to_dict()


with st.sidebar:
    st.header("Scenario controls")
    selected_label = st.segmented_control(
        "Scoring model",
        options=list(MODEL_LABELS.values()),
        default=MODEL_LABELS[DEFAULT_MODEL],
        key="prediction_model",
    )
    model_name = next(
        (
            name
            for name, label in MODEL_LABELS.items()
            if label == selected_label
        ),
        DEFAULT_MODEL,
    )
    preset = st.selectbox("Load a profile", list(presets))
    threshold = st.slider(
        "Approval threshold",
        0.10,
        0.99,
        DEFAULT_THRESHOLD,
        0.05,
        help="The probability required to label an application likely approved.",
    )

selected_model = load_model(model_name)
profile = get_feature_defaults() | {
    "Age": presets[preset]["age"],
    "Income": presets[preset]["income"],
    "LoanAmount": presets[preset]["amount"],
    "InterestRate": presets[preset]["rate"],
    "DTIRatio": presets[preset]["dti"],
}
values = {}
input_columns = st.columns(2)
for index, feature in enumerate(selected_model["features"]):
    label, minimum, maximum, step, integer = feature_inputs[feature]
    initial_value = min(max(float(profile[feature]), minimum), maximum)
    if integer:
        initial_value = int(round(initial_value))
    with input_columns[index % 2]:
        values[feature] = st.number_input(
            label,
            min_value=int(minimum) if integer else minimum,
            max_value=int(maximum) if integer else maximum,
            value=initial_value,
            step=int(step) if integer else step,
            key=f"{preset}_{model_name}_{feature}",
        )

result = predict_one(values, threshold=threshold, model_name=model_name)

st.divider()
st.subheader("Current assessment")
result_cols = st.columns([1, 1, 1.4])
result_cols[0].metric("Approval probability", f"{result['approval_probability']:.1%}")
result_cols[1].metric("Decision threshold", f"{threshold:.0%}")
if result["loan_status"]:
    result_cols[2].success(result["decision"])
else:
    result_cols[2].error(result["decision"])
st.progress(result["approval_probability"], text=f"Model score: {result['approval_probability']:.1%}")

evidence_cols = st.columns(3)
if model_name == "decision_tree":
    evidence_cols[0].metric("Tree leaf approval rate", f"{result['leaf_approval_rate']:.1%}")
    evidence_cols[1].metric("Comparable training rows", f"{result['leaf_samples']:,}")
else:
    evidence_cols[0].metric("Model fit rows", f"{selected_model['metrics']['fit_rows']:,}")
    evidence_cols[1].metric("Validation ROC-AUC", f"{selected_model['metrics']['roc_auc']:.3f}")
evidence_cols[2].metric("Selected features", f"{len(selected_model['features'])}")

with st.expander("Inspect submitted values"):
    submitted_values = pd.DataFrame({"Field": list(values), "Value": [str(value) for value in values.values()]})
    st.dataframe(submitted_values, width="stretch", hide_index=True)

if "data_warning" in result:
    st.warning(result["data_warning"])

if result["approval_probability"] < threshold:
    st.info("Below threshold: adjust one or more of the selected model features and reassess the estimate.")
else:
    st.info("Above threshold: increase the threshold to apply a more conservative approval policy.")

st.subheader("One-variable sensitivity")
st.caption("Each comparison changes one factor while keeping the rest of the current profile fixed.")
sensitivity_rows = []
adjustments = {
    "Age": ("Applicant age +1", lambda value: min(value + 1, 100)),
    "Income": ("Annual income +10%", lambda value: min(value * 1.1, 1_000_000)),
    "LoanAmount": ("Loan amount -20%", lambda value: max(value * 0.8, 500)),
    "CreditScore": ("Credit score +25", lambda value: min(value + 25, 850)),
    "MonthsEmployed": ("Months employed +12", lambda value: min(value + 12, 600)),
    "NumCreditLines": ("Credit lines +1", lambda value: min(value + 1, 20)),
    "InterestRate": ("Interest rate -2%", lambda value: max(value - 2, 0)),
    "LoanTerm": ("Loan term +12 months", lambda value: min(value + 12, 120)),
    "DTIRatio": ("Debt-to-income -0.10", lambda value: max(value - 0.10, 0)),
}
for feature in selected_model["features"][:3]:
    label, adjust = adjustments[feature]
    adjusted_value = adjust(values[feature])
    if adjusted_value == values[feature]:
        continue
    changes = {feature: adjusted_value}
    scenario_result = predict_one(values | changes, threshold=threshold, model_name=model_name)
    sensitivity_rows.append({"Change": label, "Approval probability": scenario_result["approval_probability"], "Decision": scenario_result["decision"]})
if sensitivity_rows:
    st.dataframe(pd.DataFrame(sensitivity_rows).style.format({"Approval probability": "{:.1%}"}), width="stretch", hide_index=True)
else:
    st.caption("No in-range sensitivity adjustments are available for these values.")

if model_name == "logistic_regression":
    st.caption(
        f"Logistic regression uses the five PCA-and-logistic-selected features: "
        f"{', '.join(selected_model['features'])}."
    )
else:
    st.caption(
        f"The decision tree uses: {', '.join(selected_model['features'])}."
    )