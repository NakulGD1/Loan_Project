import pandas as pd
import streamlit as st

from loan_model import (
    DEFAULT_MODEL,
    MODEL_LABELS,
    load_model,
)

st.title("Model insights")
st.caption("Select a trained model to inspect its performance and feature interpretation.")
selected_label = st.segmented_control(
    "Model",
    options=list(MODEL_LABELS.values()),
    default=MODEL_LABELS[DEFAULT_MODEL],
    key="insights_model",
)
model_name = next(
    (
        name
        for name, label in MODEL_LABELS.items()
        if label == selected_label
    ),
    DEFAULT_MODEL,
)
selected_model = load_model(model_name)
pipeline = selected_model["pipeline"]
metrics = selected_model["metrics"]
st.subheader(MODEL_LABELS[model_name])

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
metric_cols[1].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
metric_cols[2].metric("Model fit rows", f"{metrics['fit_rows']:,}")
metric_cols[3].metric("Validation rows", f"{metrics['test_rows']:,}")

st.subheader("Confusion matrix")
matrix = pd.DataFrame(metrics["confusion_matrix"], index=["Actual declined", "Actual approved"], columns=["Predicted declined", "Predicted approved"])
st.dataframe(matrix, width="stretch")

st.subheader(
    "Feature influence"
    if model_name == "logistic_regression"
    else "Feature importance"
)
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
encoded_names = list(preprocessor.get_feature_names_out())
if model_name == "logistic_regression":
    importance = pd.DataFrame(
        {"feature": encoded_names, "coefficient": classifier.coef_[0]}
    )
    importance["absolute_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("absolute_coefficient", ascending=False)
    st.caption(
        "Positive standardized coefficients increase estimated approval odds; "
        "negative coefficients decrease them."
    )
    st.bar_chart(
        importance.set_index("feature"),
        y="coefficient",
        x_label="Selected feature",
        y_label="Coefficient (approval)",
    )
else:
    importance = pd.DataFrame(
        {"feature": encoded_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)
    st.bar_chart(
        importance.set_index("feature"),
        y="importance",
        x_label="Feature",
        y_label="Importance",
    )

st.subheader(
    "PCA feature selection"
    if model_name == "logistic_regression"
    else "PCA context"
)
pca = selected_model["pca"]
pca_cols = st.columns(len(pca["explained_variance_ratio"]), gap="small")
for index, column in enumerate(pca_cols, start=1):
    column.metric(
        f"PC{index} variance",
        f"{pca['explained_variance_ratio'][index - 1]:.1%}",
    )
st.dataframe(
    pca["ranking"].assign(
        selected=pca["ranking"]["feature"].isin(selected_model["features"])
    ),
    width="stretch",
    hide_index=True,
)

if model_name == "logistic_regression":
    st.info(
        "PCA components are fit on the training split, logistic regression is "
        "fit to those components, and its coefficients are projected back to "
        "rank the original inputs. The validation model uses its top "
        f"{len(metrics['validation_features'])} features: "
        f"{', '.join(metrics['validation_features'])}. The final deployed "
        f"model is ranked from the full CSV and fit on all "
        f"{metrics['fit_rows']:,} rows with: "
        f"{', '.join(selected_model['features'])}."
    )
else:
    st.info(
        f"The decision tree uses: {', '.join(selected_model['features'])}. "
        "PCA is shown for context. This is a screening aid, not an automatic "
        "credit decision."
    )