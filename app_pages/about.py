import streamlit as st


st.title("Project guide")
st.markdown("""
### LoanLens
LoanLens is a compact end-to-end machine learning project built around the loan application dataset.

**Workflow**
1. Explore the source records and portfolio patterns.
2. Compare decision-tree feature importance with PCA- and logistic-coefficient-ranked features.
3. Train the decision tree on a stratified split; select logistic-regression features using PCA and a training-split logistic model, test on held-out rows, and refit the selected-feature model on the full CSV.
4. Score applications interactively in Streamlit or through the FastAPI service.

The prediction lab and model insights page let you select either model. Logistic regression is saved and loaded separately after PCA feature selection. The default approval threshold is 90%.

**API quick start**
```text
GET  /health
GET  /model-info
POST /predict (optional `model`: `decision_tree` or `logistic_regression`)
```

Run the API from this folder with `uvicorn api:app --reload --port 8000`.
""")