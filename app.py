from pathlib import Path

import streamlit as st


st.set_page_config(
	page_title="LoanLens | Credit intelligence",
	page_icon=":material/account_balance:",
	layout="wide",
	initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
	:root { --ink: #17211f; --muted: #66736d; --mint: #d8f3dc; --coral: #b8442b; }
	html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
	h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
	.hero { padding: 2.5rem 0 1.4rem; }
	.hero-kicker { color: var(--coral); font-weight: 700; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; }
	.hero h1 { font-size: clamp(2.2rem, 5vw, 4.4rem); line-height: .98; margin: .45rem 0 .8rem; max-width: 780px; }
	.hero p { color: var(--muted); max-width: 650px; font-size: 1.05rem; }
	.metric-card { border-left: 4px solid var(--coral); padding: 1rem 1.1rem; background: rgba(255,255,255,.72); border-radius: 8px; min-height: 112px; }
	.metric-label { color: var(--muted); font-size: .82rem; }
	.metric-value { color: var(--ink); font: 700 1.8rem 'Space Grotesk', sans-serif; margin-top: .35rem; }
	</style>
	""",
	unsafe_allow_html=True,
)

pages = {
	"Explore": [
		st.Page("app_pages/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
		st.Page("app_pages/data_analysis.py", title="Data analysis", icon=":material/analytics:"),
		st.Page("app_pages/model_insights.py", title="Model insights", icon=":material/account_tree:"),
		st.Page("app_pages/predict.py", title="Predict a loan", icon=":material/online_prediction:"),
	],
	"Resources": [
		st.Page("app_pages/about.py", title="Project guide", icon=":material/menu_book:"),
	],
}

pg = st.navigation(pages)
pg.run()
