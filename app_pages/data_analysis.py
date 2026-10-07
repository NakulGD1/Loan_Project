import streamlit as st

from loan_model import load_data


@st.cache_data(show_spinner="Loading loan data...")
def get_data():
    return load_data()


data = get_data()
st.title("Data analysis")
st.caption("Filter the portfolio and inspect distributions before making a modeling decision.")

with st.sidebar:
    st.header("Analysis filters")
    purposes = st.multiselect("Loan purposes", sorted(data["LoanPurpose"].unique()), default=sorted(data["LoanPurpose"].unique()))
    employment_types = st.multiselect("Employment types", sorted(data["EmploymentType"].unique()), default=sorted(data["EmploymentType"].unique()))
    score_range = st.slider("Credit score", int(data.CreditScore.min()), int(data.CreditScore.max()), (int(data.CreditScore.min()), int(data.CreditScore.max())))

filtered = data[data.LoanPurpose.isin(purposes) & data.EmploymentType.isin(employment_types) & data.CreditScore.between(*score_range)]
st.write(f"Showing **{len(filtered):,}** of **{len(data):,}** applications")

metric_cols = st.columns(3)
metric_cols[0].metric("Non-default rate", f"{(filtered.Default == 0).mean():.1%}" if len(filtered) else "n/a")
metric_cols[1].metric("Median income", f"${filtered.Income.median():,.0f}" if len(filtered) else "n/a")
metric_cols[2].metric("Median loan", f"${filtered.LoanAmount.median():,.0f}" if len(filtered) else "n/a")
st.caption("Non-default rate is based on the dataset's Default label (0 = no observed default), not on recorded loan-approval decisions.")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Credit score distribution")
    st.bar_chart(filtered.assign(score_band=(filtered.CreditScore // 25) * 25).groupby("score_band").size(), x_label="Score band", y_label="Applications")
with chart_right:
    st.subheader("Income versus loan amount")
    scatter_data = filtered.sample(n=min(len(filtered), 5_000), random_state=42)
    st.scatter_chart(scatter_data.assign(**{"Non-default": scatter_data.Default == 0}), x="Income", y="LoanAmount", color="Non-default", x_label="Annual income", y_label="Loan amount")

st.subheader("Filtered records")
st.dataframe(filtered.head(500), width="stretch", hide_index=True)