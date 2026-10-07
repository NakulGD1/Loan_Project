import streamlit as st

from loan_model import load_data, load_model


data = load_data()
bundle = load_model()
metrics = bundle["metrics"]
historical_non_default_rate = (data["Default"] == 0).mean()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">LoanLens / credit intelligence</div>
      <h1>See the shape of lending risk.</h1>
      <p>Explore borrower default outcomes and compare models that estimate non-default likelihood.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
cards = [
    ("Applications", f"{len(data):,}"),
    ("Historical non-default rate", f"{historical_non_default_rate:.2%}"),
    ("Holdout accuracy", f"{metrics['accuracy']:.2%}"),
    ("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
]
for col, (label, value) in zip(cols, cards):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

rate_context = (
    "The dataset records default outcomes, not approval decisions: Default = 0 "
    "means the borrower did not default. The historical rate uses all "
    f"{len(data):,} records; holdout accuracy uses "
    f"{metrics['test_rows']:,} validation records. Accuracy alone can obscure "
    "how well a model detects the less common default outcome."
)
if f"{historical_non_default_rate:.1%}" == f"{metrics['accuracy']:.1%}":
    rate_context += " These different measures happen to round to the same value at one decimal."
st.caption(rate_context)

st.write("")
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Historical non-default profile")
    non_default_by_purpose = data.groupby("LoanPurpose", as_index=False)["Default"].mean()
    non_default_by_purpose["Non-default rate"] = (
        1 - non_default_by_purpose.pop("Default")
    ).mul(100).round(1)
    non_default_by_purpose = non_default_by_purpose.sort_values("Non-default rate", ascending=False)
    st.bar_chart(
        non_default_by_purpose.set_index("LoanPurpose"),
        y="Non-default rate",
        x_label="Loan purpose",
        y_label="Non-default rate (%)",
    )
with right:
    st.subheader("Portfolio snapshot")
    st.dataframe(
        data[["LoanAmount", "InterestRate", "CreditScore", "DTIRatio", "Default"]].describe().T,
        width="stretch",
    )

st.subheader("What is in the data?")
st.write("Borrower demographics, income and employment history, loan purpose, pricing, debt burden, credit history, and default behavior.")