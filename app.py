import streamlit as st
import pandas as pd
from html import escape as escape_html
from datetime import date
from src.behavioral_profile import predict_behavioral_profile
from src.inference import predict_productivity
from src.similar_days import (
    find_similar_days,
    summarize_similar_days
)
from src.database import (
    initialize_database,
    save_daily_record,
    get_history_count,
    get_user_history,
    update_actual_productivity,
    get_completed_history
)

from src.personalization import personalize_prediction
from src.pca_visualization import transform_to_pca

st.set_page_config(
    page_title="LifeTwin | Behavioral Intelligence",
    layout="wide"
)

initialize_database()

SESSION_DEFAULTS = {
    "prediction_result": None,
    "population_prediction": None,
    "personalization_result": None,
    "behavioral_result": None,
    "pca_result": None,
    "what_if_result": None,
    "prediction_user_id": None,
    "today_input": None,
}
for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --bg: #F7FAF7;
        --surface: #FFFFFF;
        --surface-strong: #E5F1E8;
        --border: #DCE8DF;
        --text: #385B4B;
        --muted: #557064;
        --forest: #174C36;
        --action: #24734D;
        --sage: #E5F1E8;
        --accent: #24734D;
        --accent-soft: #D5EBDD;
        --shadow: 0 4px 14px rgba(23, 76, 54, 0.05);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: var(--bg); }
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div { padding: 1.35rem 1rem 1rem; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        min-width: 0;
    }
    .sidebar-brand {
        margin: 0 0 1.35rem;
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        line-height: 1.2;
    }
    .sidebar-brand span {
        display: block;
        margin-top: 0.42rem;
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.105em;
        line-height: 1.5;
    }
    .sidebar-section-label {
        margin: 0 0 0.5rem;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        width: 100%;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.3rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        position: relative;
        display: flex;
        width: 100%;
        min-width: 0;
        min-height: 2.75rem;
        align-items: center;
        gap: 0.75rem;
        padding: 0.58rem 0.7rem 0.58rem 0.8rem;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #385B4B;
        cursor: pointer;
        transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div[aria-hidden="true"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
        min-width: 0;
        overflow-wrap: anywhere;
        white-space: normal;
        line-height: 1.35;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]::before {
        content: "";
        flex: 0 0 1.2rem;
        width: 1.2rem;
        height: 1.2rem;
        background-position: center;
        background-repeat: no-repeat;
        background-size: 1.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(1)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23174C36' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3.5' y='3.5' width='7' height='7' rx='1'/%3E%3Crect x='13.5' y='3.5' width='7' height='7' rx='1'/%3E%3Crect x='3.5' y='13.5' width='7' height='7' rx='1'/%3E%3Crect x='13.5' y='13.5' width='7' height='7' rx='1'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(2)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23174C36' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12h4l3-7 4 14 3-7h4'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(3)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23174C36' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 4.5a3 3 0 0 0-5.5 1.7 3.5 3.5 0 0 0-.1 6.6A3.5 3.5 0 0 0 7 18.5a3 3 0 0 0 5-2.2V7.5a3 3 0 0 0-3-3Z'/%3E%3Cpath d='M15 4.5a3 3 0 0 1 5.5 1.7 3.5 3.5 0 0 1 .1 6.6 3.5 3.5 0 0 1-3.6 5.7 3 3 0 0 1-5-2.2V7.5a3 3 0 0 1 3-3Z'/%3E%3Cpath d='M8 9h2m4 0h2m-8 5h2m4 0h2'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(4)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23174C36' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 6h16M4 12h16M4 18h16'/%3E%3Ccircle cx='9' cy='6' r='2' fill='%23fff'/%3E%3Ccircle cx='15' cy='12' r='2' fill='%23fff'/%3E%3Ccircle cx='11' cy='18' r='2' fill='%23fff'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(5)::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23174C36' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 2.6-6.4L3 8'/%3E%3Cpath d='M3 3v5h5m4-1v5l3 2'/%3E%3C/svg%3E");
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: #F3F7F4;
        color: var(--forest);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        border-color: var(--border);
        background: var(--sage);
        color: var(--forest);
        font-weight: 700;
        box-shadow: inset 3px 0 0 var(--forest);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:focus-within {
        outline: 3px solid rgba(36, 115, 77, 0.32);
        outline-offset: 2px;
    }
    .sidebar-footer {
        margin-top: 1rem;
        padding: 0.9rem 0.85rem;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #F7FAF7;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        overflow-wrap: anywhere;
    }
    .sidebar-footer strong {
        color: var(--forest);
        font-weight: 700;
    }
    h1, h2, h3, h4 {
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
    }
    h1 { font-weight: 800; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMarkdownContainer"] p { color: var(--text); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--forest); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: var(--shadow);
    }
    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: var(--shadow);
    }
    .stButton > button, .stFormSubmitButton > button {
        border: 1px solid var(--action);
        border-radius: 11px;
        color: #FFFFFF !important;
        background: var(--action) !important;
        transition: border-color 140ms ease, background-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
    }
    .stButton > button *, .stFormSubmitButton > button * {
        color: #FFFFFF !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        border-color: var(--forest);
        color: #FFFFFF !important;
        background: var(--forest) !important;
        box-shadow: 0 5px 14px rgba(23, 76, 54, 0.16);
        transform: translateY(-1px);
    }
    .stButton > button:active, .stFormSubmitButton > button:active {
        border-color: #103B29;
        color: #FFFFFF !important;
        background: #103B29 !important;
        transform: translateY(0);
    }
    .stButton > button:disabled, .stFormSubmitButton > button:disabled {
        border-color: var(--border);
        color: var(--muted) !important;
        background: #E5F1E8 !important;
        box-shadow: none;
        cursor: not-allowed;
        transform: none;
    }
    .stButton > button:disabled *, .stFormSubmitButton > button:disabled * {
        color: var(--muted) !important;
    }
    .stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible,
    [data-testid="stRadio"] [role="radio"]:focus-visible,
    [data-testid="stSelectbox"] input:focus-visible,
    [data-testid="stTextInput"] input:focus-visible,
    [data-testid="stNumberInput"] input:focus-visible,
    [data-testid="stSlider"] [role="slider"]:focus-visible {
        outline: 3px solid rgba(36, 115, 77, 0.45) !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 3px rgba(229, 241, 232, 0.9) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input,
    [data-testid="stTextArea"] textarea {
        color: var(--forest) !important;
        background: #FFFFFF !important;
        border-color: rgba(23, 76, 54, 0.24) !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder { color: #71877B !important; }
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
    [data-testid="stSelectbox"] label, [data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label, [data-testid="stTextInput"] label {
        color: var(--forest);
    }
    [data-testid="stRadio"] input[type="radio"],
    [data-testid="stCheckbox"] input[type="checkbox"] { accent-color: var(--action); }
    [data-baseweb="popover"], [data-baseweb="menu"] {
        color: var(--forest);
        background: #FFFFFF;
    }
    [data-testid="stAlert"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(23, 76, 54, 0.04);
    }
    [data-testid="stSlider"] [role="slider"] { accent-color: var(--action); }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
        background: #D5EBDD;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * {
        color: var(--forest);
    }
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"],
    [data-testid="stPlotlyChart"], [data-testid="stAltairChart"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.45rem;
        box-shadow: var(--shadow);
    }
    hr { border-color: rgba(23, 76, 54, 0.14); }
    .eyebrow {
        color: var(--action);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .hero-card, .overview-primary-kpi, .productivity-result-score, .behavioral-profile-card {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(36, 115, 77, 0.24);
        border-radius: 20px;
        background: #FFFFFF;
        box-shadow: var(--shadow);
    }
    .hero-card p { margin-bottom: 0; }
    .quiet-card, .overview-nav-card, .overview-support-card, .productivity-result-card {
        padding: 1.1rem 1.25rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    .quiet-card { min-height: 100%; }
    .quiet-card h4, .overview-nav-card h4 { margin: 0 0 0.55rem; }
    .quiet-card p, .overview-nav-card p { margin: 0; line-height: 1.55; }
    .overview-primary-kpi, .productivity-result-score, .behavioral-profile-card {
        padding: 1.35rem 1.5rem;
    }
    .overview-kpi-label, .productivity-result-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 600;
    }
    .overview-kpi-value, .productivity-result-value {
        margin-top: 0.4rem;
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 6vw, 4.25rem);
        font-weight: 800;
        line-height: 1.1;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .overview-kpi-note, .overview-insight-meta, .behavioral-chart-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .overview-kpi-note { margin-top: 0.4rem; }
    .overview-support-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.85rem 0 1.25rem;
    }
    .overview-support-card { min-width: 0; }
    .overview-support-label, .productivity-result-card-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        line-height: 1.4;
        text-transform: uppercase;
    }
    .overview-support-value, .productivity-result-card-value {
        margin-top: 0.45rem;
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1rem, 2vw, 1.3rem);
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .overview-insight, .productivity-personalization {
        padding: 1.15rem 1.3rem;
        border: 1px solid rgba(36, 115, 77, 0.18);
        border-radius: 17px;
        background: #FFFFFF;
        box-shadow: 0 6px 22px rgba(23, 76, 54, 0.06);
    }
    .overview-insight h4 { margin: 0 0 0.55rem; }
    .overview-insight p { margin: 0; line-height: 1.6; overflow-wrap: anywhere; }
    .overview-insight-meta { margin-top: 0.65rem; font-size: 0.82rem; overflow-wrap: anywhere; }
    .overview-nav-card {
        display: flex;
        min-height: 9.6rem;
        flex-direction: column;
        padding: 1.05rem 1.1rem;
        margin-bottom: 0.55rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    .overview-nav-card:hover {
        border-color: #B8D0BF;
        box-shadow: 0 6px 18px rgba(23, 76, 54, 0.08);
    }
    .overview-nav-card-heading {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.5rem;
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 700;
    }
    .overview-nav-card-heading > span:last-child {
        min-width: 0;
        overflow-wrap: anywhere;
    }
    .overview-nav-card-icon {
        display: inline-flex;
        width: 1.35rem;
        height: 1.35rem;
        flex: 0 0 1.35rem;
        color: var(--action);
    }
    .overview-nav-card-icon svg {
        width: 100%;
        height: 100%;
        fill: none;
        stroke: currentColor;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 1.7;
    }
    .overview-nav-card p { min-height: 2.8rem; }
    .overview-nav-action { margin-bottom: 0.9rem; }
    .productivity-section {
        margin: 0 0 1rem;
        padding: 1.15rem 1.25rem 0.35rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: #FFFFFF;
        box-shadow: var(--shadow);
    }
    .productivity-section-heading {
        margin: 0 0 0.9rem;
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .productivity-section-heading::before {
        content: '';
        display: inline-block;
        width: 0.42rem;
        height: 0.95rem;
        margin-right: 0.65rem;
        border-radius: 1rem;
        background: var(--action);
        vertical-align: -0.1rem;
    }
    .productivity-result-score { padding: 1.25rem 1.45rem; }
    .productivity-result-value { font-size: clamp(2.5rem, 7vw, 4.5rem); }
    .productivity-result-card { height: 100%; padding: 1.05rem 1.15rem; }
    .productivity-result-card-value.is-productive { color: #176344; }
    .productivity-result-card-value.is-non-productive { color: #7A3C4A; }
    .productivity-personalization { margin-top: 0.85rem; padding: 1rem 1.15rem; }
    .productivity-personalization p { margin: 0.4rem 0 0; line-height: 1.6; overflow-wrap: anywhere; }
    .productivity-personalization-stage {
        color: var(--action);
        font-family: 'Manrope', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .behavioral-profile-card { padding: 1.4rem 1.55rem; }
    .behavioral-profile-label, .behavioral-profile-name { color: var(--forest); }
    .behavioral-profile-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .behavioral-profile-name {
        margin-top: 0.45rem;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.35rem, 3vw, 2rem);
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .behavioral-profile-note, .behavioral-history-empty {
        color: var(--muted);
        line-height: 1.6;
    }
    .behavioral-profile-note { margin: 0.65rem 0 0; }
    .behavioral-chart-note { margin: 0.35rem 0 0.8rem; }
    .behavioral-history-empty {
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 15px;
        background: var(--surface);
    }
    @media (max-width: 720px) {
        .productivity-section { padding: 1rem 0.85rem 0.2rem; }
        .productivity-result-score, .overview-primary-kpi { padding: 1.1rem 1.15rem; }
        .overview-support-grid { grid-template-columns: 1fr; gap: 0.65rem; }
        .overview-nav-card { min-height: auto; }
        .overview-nav-card p { min-height: 0; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover { transform: none; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODULES = [
    "Overview",
    "Productivity",
    "Behavioral Intelligence",
    "What-If Lab",
    "My Journey",
]

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">LifeTwin'
        '<span>PERSONAL BEHAVIORAL INTELLIGENCE</span></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-section-label">WORKSPACE</div>', unsafe_allow_html=True)
    selected_module = st.radio(
        "Navigation",
        MODULES,
        label_visibility="collapsed",
        key="active_module"
    )
    st.divider()
    user_id = st.text_input(
        "User ID",
        value="USER001",
        help="Use the same ID each day to build your personal history.",
        key="user_id"
    )
    history_count = get_history_count(user_id.strip()) if user_id.strip() else 0
    st.markdown(
        f'<div class="sidebar-footer">'
        f'<strong>{escape_html(user_id.strip() or "No user selected")}</strong><br>'
        f'{history_count} recorded day(s)<br>'
        'Your patterns. Your personal baseline.'
        '</div>',
        unsafe_allow_html=True
    )

history = get_user_history(user_id.strip()) if user_id.strip() else []
today_record = next(
    (record for record in reversed(history) if str(record[2]) == str(date.today())),
    None
)
saved_actual_today = (
    None
    if today_record is None or today_record[22] is None
    else float(today_record[22])
)

st.markdown('<div class="eyebrow">PERSONAL INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)
st.title(selected_module)
st.caption(f"Your personal behavioral signal | {date.today().strftime('%A, %B %d, %Y')}")


def render_behavioral_inputs():
    previous_input = st.session_state.today_input or {}
    st.subheader("Today's behavioral inputs")
    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Sleep &amp; Recovery</div>',
            unsafe_allow_html=True
        )
        input_col1, input_col2, input_col3 = st.columns(3)
        with input_col1:
            sleep_hours = st.number_input(
                "Sleep Hours", min_value=0.0, max_value=24.0,
                value=float(previous_input.get("sleep_hours", 7.0)), step=0.5, key="sleep_hours"
            )
        with input_col2:
            sleep_quality = st.slider(
                "Sleep Quality", min_value=1.0, max_value=5.0,
                value=float(previous_input.get("sleep_quality", 4.0)), step=0.1, key="sleep_quality"
            )
        with input_col3:
            wake_time_consistency = st.slider(
                "Wake Time Consistency", min_value=0.0, max_value=1.0,
                value=float(previous_input.get("wake_time_consistency", 0.8)), step=0.05, key="wake_time_consistency"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Work &amp; Focus</div>',
            unsafe_allow_html=True
        )
        input_col1, input_col2, input_col3 = st.columns(3)
        with input_col1:
            study_work_hours = st.number_input(
                "Study / Work Hours", min_value=0.0, max_value=24.0,
                value=float(previous_input.get("study_work_hours", 6.0)), step=0.5, key="study_work_hours"
            )
        with input_col2:
            focus_sessions = st.number_input(
                "Focus Sessions", min_value=0, max_value=30,
                value=int(previous_input.get("focus_sessions", 4)), step=1, key="focus_sessions"
            )
        with input_col3:
            break_minutes = st.number_input(
                "Break Minutes", min_value=0, max_value=600,
                value=int(previous_input.get("break_minutes", 60)), step=5, key="break_minutes"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Digital Habits</div>',
            unsafe_allow_html=True
        )
        input_col1, input_col2, input_col3 = st.columns(3)
        with input_col1:
            screen_time_hours = st.number_input(
                "Screen Time (Hours)", min_value=0.0, max_value=24.0,
                value=float(previous_input.get("screen_time_hours", 5.0)), step=0.5, key="screen_time_hours"
            )
        with input_col2:
            social_media_hours = st.number_input(
                "Social Media (Hours)", min_value=0.0, max_value=24.0,
                value=float(previous_input.get("social_media_hours", 2.0)), step=0.5, key="social_media_hours"
            )
        with input_col3:
            notification_interruptions = st.number_input(
                "Notification Interruptions", min_value=0, max_value=100,
                value=int(previous_input.get("notification_interruptions", 10)), step=1, key="notification_interruptions"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Physical Activity</div>',
            unsafe_allow_html=True
        )
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            exercise_minutes = st.number_input(
                "Exercise (Minutes)", min_value=0, max_value=300,
                value=int(previous_input.get("exercise_minutes", 30)), step=5, key="exercise_minutes"
            )
        with input_col2:
            step_count = st.number_input(
                "Step Count", min_value=0, max_value=50000,
                value=int(previous_input.get("step_count", 7000)), step=500, key="step_count"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Mood &amp; Wellbeing</div>',
            unsafe_allow_html=True
        )
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            stress_level = st.slider(
                "Stress Level", min_value=1.0, max_value=5.0,
                value=float(previous_input.get("stress_level", 3.0)), step=0.1, key="stress_level"
            )
            mood_score = st.slider(
                "Mood Score", min_value=1.0, max_value=5.0,
                value=float(previous_input.get("mood_score", 4.0)), step=0.1, key="mood_score"
            )
        with input_col2:
            energy_level = st.slider(
                "Energy Level", min_value=1.0, max_value=5.0,
                value=float(previous_input.get("energy_level", 4.0)), step=0.1, key="energy_level"
            )
            caffeine_intake = st.number_input(
                "Caffeine Intake", min_value=0, max_value=20,
                value=int(previous_input.get("caffeine_intake", 2)), step=1, key="caffeine_intake"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="productivity-section-heading">Profile &amp; Schedule</div>',
            unsafe_allow_html=True
        )
        context_col1, context_col2 = st.columns(2)
        with context_col1:
            age_group = st.selectbox(
                "Age Group", ["18-24", "25-34", "35-44", "45+"],
                index=["18-24", "25-34", "35-44", "45+"].index(
                    previous_input.get("age_group", "18-24")
                ),
                key="age_group"
            )
        with context_col2:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_of_week = st.selectbox(
                "Day of Week",
                days,
                index=days.index(previous_input.get("day_of_week", "Monday")),
                key="day_of_week"
            )

    return {
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "wake_time_consistency": wake_time_consistency,
        "study_work_hours": study_work_hours,
        "focus_sessions": focus_sessions,
        "screen_time_hours": screen_time_hours,
        "social_media_hours": social_media_hours,
        "notification_interruptions": notification_interruptions,
        "exercise_minutes": exercise_minutes,
        "step_count": step_count,
        "stress_level": stress_level,
        "mood_score": mood_score,
        "energy_level": energy_level,
        "caffeine_intake": caffeine_intake,
        "break_minutes": break_minutes,
        "age_group": age_group,
        "day_of_week": day_of_week
    }


def render_similar_days(current_day, completed_history):
    if len(completed_history) < 3:
        st.info("Similar-Day intelligence becomes available after at least 3 completed personal days.")
        return

    similar_days = find_similar_days(
        current_day=current_day,
        history=completed_history,
        n_neighbors=3
    )
    if not similar_days:
        st.info("No completed similar days are available for this behavior yet.")
        return

    summary = summarize_similar_days(similar_days)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Similar Days", summary["count"])
    metric2.metric("Average Productivity", f"{summary['average_productivity']:.2f}")
    metric3.metric("Best Similar Day", f"{summary['best_productivity']:.2f}")
    st.caption("Closest completed days from your personal history.")
    st.dataframe(
        pd.DataFrame([
            {
                "Date": day["record_date"],
                "Actual Productivity": day["actual_productivity"],
                "Similarity Distance": round(day["distance"], 2)
            }
            for day in similar_days
        ]),
        use_container_width=True,
        hide_index=True
    )


if selected_module == "Overview":
    prediction = st.session_state.prediction_result
    personalization = st.session_state.personalization_result
    behavioral = st.session_state.behavioral_result

    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">YOUR DAY, IN FOCUS</div>
            <h2 style="margin: 0.45rem 0 0.35rem 0;">A clearer view of your behavioral rhythm.</h2>
            <p>LifeTwin turns your daily signals into a personal, evolving view of productivity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    if prediction is None:
        st.info("Run today's prediction in Productivity to populate your personal overview.")
        score = None
        classification = "Awaiting prediction"
        score_source = "Run a prediction to see today's estimate."
    else:
        score = (
            personalization["prediction"]
            if personalization is not None
            else prediction["productivity_score"]
        )
        classification = (
            "Productive day"
            if prediction["productive_class"] == 1
            else "Non-productive day"
        )
        score_source = (
            "Personalized estimate"
            if personalization is not None
            else "Model estimate"
        )

    profile = "Not available" if behavioral is None else behavioral["profile"]
    personalization_stage = (
        "Building baseline"
        if personalization is None
        else personalization["stage"]
    )
    shown_score = "—" if score is None else f"{score:.2f}"
    st.markdown(
        f"""
        <section class="overview-primary-kpi" aria-label="Today's productivity">
            <div class="overview-kpi-label">Today's productivity</div>
            <div class="overview-kpi-value">{escape_html(shown_score)}</div>
            <div class="overview-kpi-note">{escape_html(score_source)}</div>
        </section>
        <div class="overview-support-grid">
            <div class="overview-support-card">
                <div class="overview-support-label">Classification</div>
                <div class="overview-support-value">{escape_html(classification)}</div>
            </div>
            <div class="overview-support-card">
                <div class="overview-support-label">Behavioral profile</div>
                <div class="overview-support-value">{escape_html(profile)}</div>
            </div>
            <div class="overview-support-card">
                <div class="overview-support-label">Personalization stage</div>
                <div class="overview-support-value">{escape_html(personalization_stage)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if personalization is not None:
        insight_text = personalization["description"]
    elif prediction is not None:
        insight_text = (
            "A personalized explanation is not available for this prediction. "
            "The displayed score is the model estimate."
        )
    else:
        insight_text = (
            "Run today's prediction in Productivity to see the estimate and any "
            "personalized insight available from your saved history."
        )
    st.markdown(
        f"""
        <div class="overview-insight">
            <h4>LifeTwin insight</h4>
            <p>{escape_html(insight_text)}</p>
            <div class="overview-insight-meta">
                {history_count} recorded day(s) in this user's history.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Explore your intelligence dashboard")
    nav_columns = st.columns(2)
    for index, (title, description, module, icon) in enumerate([
        (
            "Productivity",
            "Behavioral inputs and today's prediction",
            "Productivity",
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 3.5h7v7h-7z"/><path d="M13.5 3.5h7v7h-7z"/><path d="M3.5 13.5h7v7h-7z"/><path d="M13.5 13.5h7v7h-7z"/></svg>'
        ),
        (
            "Behavioral Intelligence",
            "Your detected profile and PCA position",
            "Behavioral Intelligence",
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4.5a3 3 0 0 0-5.5 1.7 3.5 3.5 0 0 0-.1 6.6A3.5 3.5 0 0 0 7 18.5a3 3 0 0 0 5-2.2V7.5a3 3 0 0 0-3-3Z"/><path d="M15 4.5a3 3 0 0 1 5.5 1.7 3.5 3.5 0 0 1 .1 6.6 3.5 3.5 0 0 1-3.6 5.7 3 3 0 0 1-5-2.2V7.5a3 3 0 0 1 3-3Z"/><path d="M8 9h2m4 0h2m-8 5h2m4 0h2"/></svg>'
        ),
        (
            "What-If Lab",
            "Explore an alternate input scenario",
            "What-If Lab",
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="11" cy="18" r="2"/></svg>'
        ),
        (
            "My Journey",
            "Review history and personal progress",
            "My Journey",
            '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5m4-1v5l3 2"/></svg>'
        ),
    ]):
        column = nav_columns[index % 2]
        with column:
            st.markdown(
                f'<div class="overview-nav-card">'
                f'<div class="overview-nav-card-heading">'
                f'<span class="overview-nav-card-icon">{icon}</span>'
                f'<span>{escape_html(title)}</span></div>'
                f'<p>{escape_html(description)}</p></div>',
                unsafe_allow_html=True
            )
            st.button(
                f"Open {title}",
                key=f"overview_nav_{index}",
                use_container_width=True,
                type="secondary",
                on_click=lambda destination=module: st.session_state.__setitem__(
                    "active_module", destination
                )
            )

elif selected_module == "Productivity":
    user_input = render_behavioral_inputs()
    st.session_state.today_input = user_input.copy()

    if st.button("Predict My Productivity", type="primary", use_container_width=True):
        if not user_id.strip():
            st.error("Please enter a User ID.")
            st.stop()
        try:
            result = predict_productivity(user_input)
            behavioral_result = predict_behavioral_profile(user_input)
            pca_result = transform_to_pca(user_input)
            completed_history = get_completed_history(
                user_id.strip(),
                before_date=str(date.today())
            )
            population_prediction = result["productivity_score"]
            personalization_result = personalize_prediction(
                population_prediction=population_prediction,
                history=completed_history
            )

            st.session_state.prediction_result = result
            st.session_state.population_prediction = population_prediction
            st.session_state.personalization_result = personalization_result
            st.session_state.behavioral_result = behavioral_result
            st.session_state.pca_result = pca_result
            st.session_state.prediction_user_id = user_id.strip()
            st.session_state.today_input = user_input.copy()

            save_daily_record(
                user_id=user_id.strip(),
                user_input=user_input,
                prediction=result
            )
            st.success("Today's prediction has been saved to your personal history.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    result = st.session_state.prediction_result
    personalization = st.session_state.personalization_result
    if result is not None:
        st.divider()
        st.subheader("Today's prediction")
        shown_score = (
            personalization["prediction"]
            if personalization is not None
            else result["productivity_score"]
        )
        class_name = "PRODUCTIVE DAY" if result["productive_class"] == 1 else "NON-PRODUCTIVE DAY"
        classification_class = (
            "is-productive" if result["productive_class"] == 1 else "is-non-productive"
        )
        actual_value = (
            "No feedback recorded"
            if saved_actual_today is None
            else f"{saved_actual_today:.0f}"
        )
        result_col1, result_col2 = st.columns([1.5, 1])
        with result_col1:
            st.markdown(
                f"""
                <div class="productivity-result-score">
                    <div class="productivity-result-label">Predicted productivity</div>
                    <div class="productivity-result-value">{shown_score:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with result_col2:
            st.markdown(
                f"""
                <div class="productivity-result-card">
                    <div class="productivity-result-card-label">Classification</div>
                    <div class="productivity-result-card-value {classification_class}">
                        {escape_html(class_name)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        actual_col, _ = st.columns([1, 1.5])
        with actual_col:
            st.markdown(
                f"""
                <div class="productivity-result-card">
                    <div class="productivity-result-card-label">Actual productivity</div>
                    <div class="productivity-result-card-value">{escape_html(actual_value)}</div>
                    <div class="overview-insight-meta">
                        {"Recorded feedback" if saved_actual_today is not None else "Feedback has not been recorded for today."}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        if personalization is not None:
            stage = escape_html(str(personalization["stage"]))
            description = escape_html(str(personalization["description"]))
            st.markdown(
                f"""
                <div class="productivity-personalization">
                    <div class="productivity-result-card-label">Personalization</div>
                    <div class="productivity-personalization-stage">{stage}</div>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if personalization["personal_bias"] != 0:
                st.caption(f"Personal adjustment: {personalization['personal_bias']:+.2f} points")
        else:
            st.markdown(
                """
                <div class="productivity-personalization">
                    <div class="productivity-result-card-label">Personalization</div>
                    <div class="productivity-personalization-stage">Not available</div>
                    <p>This prediction does not include a personalized stage or description.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        if saved_actual_today is not None:
            st.metric(
                "Prediction vs actual",
                f"{shown_score - saved_actual_today:+.2f}",
                help="Difference between the current personalized prediction and your recorded actual score."
            )
        else:
            st.info("Record an actual score in My Journey to enable a prediction-versus-actual comparison.")

elif selected_module == "Behavioral Intelligence":
    prediction_matches_user = (
        st.session_state.prediction_user_id == user_id.strip()
    )
    behavioral = (
        st.session_state.behavioral_result
        if prediction_matches_user
        else None
    )
    pca = st.session_state.pca_result if prediction_matches_user else None
    if behavioral is None:
        st.info(
            "Run a prediction in Productivity for this User ID to generate the "
            "matching behavioral profile and PCA position."
        )
    else:
        st.subheader("K-Means behavioral profile")
        profile_name = escape_html(str(behavioral["profile"]))
        profile_descriptions = {
            "High Work / High Stress Pattern":
                "This project-defined profile label describes a pattern combining substantial work or study activity with stronger stress signals.",
            "Balanced / Active Pattern":
                "This project-defined profile label describes a pattern reflecting a more balanced mix of activity and daily behavioral signals.",
            "High Screen / Social Pattern":
                "This project-defined profile label describes a pattern with relatively stronger screen-time and social-media signals."
        }
        profile_description = escape_html(
            profile_descriptions.get(
                behavioral["profile"],
                "This project-defined label summarizes the behavioral pattern assigned by the model."
            )
        )
        st.markdown(
            f"""
            <div class="behavioral-profile-card">
                <div class="behavioral-profile-label">Current K-Means profile</div>
                <div class="behavioral-profile-name">{profile_name}</div>
                <p class="behavioral-profile-note">
                    {profile_description} Cluster identifiers are model-local and do
                    not represent universal categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.divider()
    st.subheader("Behavioral position")
    st.markdown(
        '<p class="behavioral-chart-note">PCA components compress the existing behavioral features into two dimensions. They are not productivity scores; the plotted marker is your current position.</p>',
        unsafe_allow_html=True
    )
    if pca is not None:
        pca_1 = pca["pca_1"]
        pca_2 = pca["pca_2"]
        profile_label = (
            "Not available"
            if behavioral is None
            else str(behavioral["profile"])
        )
        pca_col1, pca_col2 = st.columns(2)
        pca_col1.metric("PCA component 1", f"{pca_1:.2f}")
        pca_col2.metric("PCA component 2", f"{pca_2:.2f}")
        pca_position = pd.DataFrame([{
            "PCA component 1": pca_1,
            "PCA component 2": pca_2,
            "Point": "Current user",
            "Behavioral profile": profile_label
        }])
        st.scatter_chart(
            pca_position,
            x="PCA component 1",
            y="PCA component 2",
            color="#24734D",
            x_label="PCA component 1",
            y_label="PCA component 2",
            height=420,
            use_container_width=True
        )
        st.caption(
            f"Current user | {profile_label} | Hover over the point for its coordinates and profile."
        )
    else:
        st.info("A current-user PCA position is not available yet. Run a prediction in Productivity.")

    st.divider()
    st.subheader("Historical behavioral patterns")
    if not user_id.strip():
        st.markdown(
            '<div class="behavioral-history-empty">Enter a User ID in the sidebar to view that user’s saved behavioral history.</div>',
            unsafe_allow_html=True
        )
    elif len(history) < 2:
        st.markdown(
            '<div class="behavioral-history-empty">At least two saved daily records for this user are needed to show a behavioral trend.</div>',
            unsafe_allow_html=True
        )
    else:
        behavioral_features = {
            "Sleep (hours)": 3,
            "Study / work (hours)": 6,
            "Stress level": 13,
            "Mood score": 14,
            "Energy level": 15
        }
        selected_feature = st.selectbox(
            "Behavioral feature",
            list(behavioral_features),
            key="behavioral_history_feature"
        )
        feature_index = behavioral_features[selected_feature]
        trend_data = pd.DataFrame({
            "Date": [str(record[2]) for record in history],
            selected_feature: [
                pd.to_numeric(record[feature_index], errors="coerce")
                for record in history
            ]
        }).dropna(subset=[selected_feature])
        if len(trend_data) < 2:
            st.markdown(
                f'<div class="behavioral-history-empty">There are not enough saved values for {escape_html(selected_feature.lower())} to show a trend yet.</div>',
                unsafe_allow_html=True
            )
        else:
            st.line_chart(
                trend_data.set_index("Date"),
                y=selected_feature,
                color="#24734D",
                x_label="Date",
                y_label=selected_feature,
                height=320,
                use_container_width=True
            )
            st.caption(
                f"Saved {selected_feature.lower()} values for this user only."
            )

elif selected_module == "What-If Lab":
    st.markdown(
        '<div class="quiet-card"><h4>Scenario workspace</h4><p>Change selected behaviors and compare the model estimate with your current baseline. This is exploratory, not a causal guarantee.</p></div>',
        unsafe_allow_html=True
    )
    baseline_input = st.session_state.today_input
    if baseline_input is None:
        baseline_input = {
            "sleep_hours": 7.0,
            "study_work_hours": 6.0,
            "screen_time_hours": 5.0,
            "exercise_minutes": 30,
            "stress_level": 3.0,
            "break_minutes": 60,
        }
        st.info("No saved behavioral inputs yet; the scenario starts from the standard baseline. Visit Productivity to set today's inputs.")

    with st.form("what_if_simulator"):
        st.subheader("Modified scenario")
        what_if_col1, what_if_col2, what_if_col3 = st.columns(3)
        with what_if_col1:
            what_if_sleep = st.number_input(
                "What-If Sleep Hours", min_value=0.0, max_value=24.0,
                value=float(baseline_input["sleep_hours"]), step=0.5
            )
            what_if_study_work = st.number_input(
                "What-If Study / Work Hours", min_value=0.0, max_value=24.0,
                value=float(baseline_input["study_work_hours"]), step=0.5
            )
        with what_if_col2:
            what_if_screen_time = st.number_input(
                "What-If Screen Time (Hours)", min_value=0.0, max_value=24.0,
                value=float(baseline_input["screen_time_hours"]), step=0.5
            )
            what_if_exercise = st.number_input(
                "What-If Exercise (Minutes)", min_value=0, max_value=300,
                value=int(baseline_input["exercise_minutes"]), step=5
            )
        with what_if_col3:
            what_if_stress = st.slider(
                "What-If Stress Level", min_value=1.0, max_value=5.0,
                value=float(baseline_input["stress_level"]), step=0.1
            )
            what_if_breaks = st.number_input(
                "What-If Break Minutes", min_value=0, max_value=600,
                value=int(baseline_input["break_minutes"]), step=5
            )
        run_what_if = st.form_submit_button("Run What-If Prediction", use_container_width=True)

    if run_what_if:
        modified_input = baseline_input.copy()
        modified_input.update({
            "sleep_hours": what_if_sleep,
            "study_work_hours": what_if_study_work,
            "screen_time_hours": what_if_screen_time,
            "exercise_minutes": what_if_exercise,
            "stress_level": what_if_stress,
            "break_minutes": what_if_breaks
        })
        try:
            baseline_score = predict_productivity(baseline_input)["productivity_score"]
            what_if_score = predict_productivity(modified_input)["productivity_score"]
            st.session_state.what_if_result = {
                "baseline_score": baseline_score,
                "what_if_score": what_if_score,
                "baseline_input": baseline_input.copy(),
                "modified_input": modified_input
            }
        except Exception as e:
            st.session_state.what_if_result = None
            st.error(f"What-If prediction failed: {e}")

    what_if_result = st.session_state.what_if_result
    if what_if_result is not None:
        baseline_score = what_if_result["baseline_score"]
        what_if_score = what_if_result["what_if_score"]
        difference = what_if_score - baseline_score
        baseline_values = what_if_result["baseline_input"]
        modified_values = what_if_result["modified_input"]
        st.divider()
        st.subheader("Baseline vs modified scenario")
        compare1, compare2, compare3 = st.columns(3)
        compare1.metric("Baseline productivity", f"{baseline_score:.2f}")
        compare2.metric("What-If productivity", f"{what_if_score:.2f}")
        compare3.metric("Difference", f"{difference:+.2f}")
        st.dataframe(
            pd.DataFrame([
                {"Measure": "Productivity", "Baseline": baseline_score, "Modified": what_if_score},
                {"Measure": "Sleep Hours", "Baseline": baseline_values["sleep_hours"], "Modified": modified_values["sleep_hours"]},
                {"Measure": "Study / Work Hours", "Baseline": baseline_values["study_work_hours"], "Modified": modified_values["study_work_hours"]},
                {"Measure": "Screen Time (Hours)", "Baseline": baseline_values["screen_time_hours"], "Modified": modified_values["screen_time_hours"]},
                {"Measure": "Exercise (Minutes)", "Baseline": baseline_values["exercise_minutes"], "Modified": modified_values["exercise_minutes"]},
                {"Measure": "Stress Level", "Baseline": baseline_values["stress_level"], "Modified": modified_values["stress_level"]},
                {"Measure": "Break Minutes", "Baseline": baseline_values["break_minutes"], "Modified": modified_values["break_minutes"]}
            ]),
            use_container_width=True,
            hide_index=True
        )
    st.caption(
        "Model-estimated association based on the modified inputs; it does not establish causality "
        "or guarantee a productivity change."
    )

elif selected_module == "My Journey":
    st.subheader("Your personal progress")
    personalization = st.session_state.personalization_result
    completed_history = (
        get_completed_history(user_id.strip(), before_date=str(date.today()))
        if user_id.strip()
        else []
    )
    journey1, journey2, journey3 = st.columns(3)
    journey1.metric(
        "Personalization stage",
        "Building baseline" if personalization is None else personalization["stage"]
    )
    journey2.metric("Completed days", len(completed_history))
    journey3.metric(
        "Days in history",
        get_history_count(user_id.strip()) if user_id.strip() else 0
    )
    if personalization is not None:
        st.caption(personalization["description"])

    if st.session_state.today_input is not None:
        st.subheader("Similar-Day analysis")
        render_similar_days(st.session_state.today_input, completed_history)

    st.divider()
    st.subheader("History and actual productivity")
    history = get_user_history(user_id.strip()) if user_id.strip() else []
    if history:
        history_by_date = {str(record[2]): record for record in history}
        record_dates = list(history_by_date)
        selected_date = st.selectbox(
            "Select a history date to update",
            options=record_dates,
            index=len(record_dates) - 1,
            key="history_date_to_update"
        )
        selected_record = history_by_date[selected_date]
        saved_actual = selected_record[22]
        default_actual = 50.0 if saved_actual is None else float(saved_actual)

        with st.form("historical_productivity_update"):
            actual_productivity_update = st.number_input(
                "Actual Productivity (0-100)",
                min_value=0.0,
                max_value=100.0,
                value=default_actual,
                step=1.0,
                key=f"actual_productivity_{selected_date}"
            )
            save_historical_productivity = st.form_submit_button(
                "Update Actual Productivity",
                use_container_width=True
            )

        if save_historical_productivity:
            rows_updated = update_actual_productivity(
                user_id=user_id.strip(),
                record_date=selected_date,
                actual_productivity=actual_productivity_update
            )
            if rows_updated > 0:
                st.success(
                    f"Actual productivity for {selected_date} updated to "
                    f"{actual_productivity_update:.0f}."
                )
                history = get_user_history(user_id.strip())
                completed_history = get_completed_history(
                    user_id.strip(),
                    before_date=str(date.today())
                )
                if (
                    st.session_state.prediction_user_id == user_id.strip()
                    and st.session_state.population_prediction is not None
                ):
                    refreshed_personalization = personalize_prediction(
                        population_prediction=st.session_state.population_prediction,
                        history=completed_history
                    )
                    st.session_state.personalization_result = refreshed_personalization
                    st.markdown("**Refreshed personalized prediction**")
                    st.metric(
                        "Predicted Productivity",
                        f"{refreshed_personalization['prediction']:.2f}"
                    )
                if st.session_state.today_input is not None:
                    st.markdown("**Refreshed Similar-Day results**")
                    render_similar_days(st.session_state.today_input, completed_history)
            else:
                st.warning(f"No history record was found for {selected_date}.")

        history_df = pd.DataFrame(
            history,
            columns=[
                "ID", "User ID", "Date", "Sleep", "Sleep Quality", "Wake Consistency",
                "Study/Work", "Focus Sessions", "Screen Time", "Social Media",
                "Notifications", "Exercise", "Steps", "Stress", "Mood", "Energy",
                "Caffeine", "Breaks", "Age Group", "Day", "Productivity", "Class",
                "Actual Productivity"
            ]
        )
        st.subheader("Historical productivity")
        st.line_chart(
            history_df.set_index("Date")[["Productivity", "Actual Productivity"]],
            color=["#24734D", "#83A98D"],
            use_container_width=True
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No history records are available for this user yet. Make a prediction to begin your journey.")
