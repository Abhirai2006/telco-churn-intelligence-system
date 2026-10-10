import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")


@st.cache_resource
def load_model_and_meta():
    """Load once per server process. Without this, every widget change
    reloads the pickle and feels like the app is hanging on Colab."""
    model = joblib.load("churn_model.pkl")
    meta = joblib.load("model_meta.pkl")
    return model, meta


model, meta = load_model_and_meta()
threshold = meta["threshold"]

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

KNOWN_CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                       "Electronic check", "Mailed check"],
}


def normalize_categories(df):
    """Fix whitespace and casing so near-matches map back to a real category
    instead of silently becoming an unknown category. Genuinely unrecognized
    values, like a made up payment method, are left as is, handle_unknown
    on the encoder takes those."""
    df = df.copy()
    for col, known in KNOWN_CATEGORIES.items():
        if col not in df.columns:
            continue
        lookup = {k.strip().lower(): k for k in known}
        df[col] = df[col].apply(
            lambda v: lookup.get(str(v).strip().lower(), v) if pd.notna(v) else v
        )
    return df


def clean_for_scoring(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = normalize_categories(df)
    return df


def validate_batch(df):
    """Reject rows with logically impossible values instead of silently
    scoring them. Returns a boolean valid mask and a matching reason for
    every rejected row."""
    reason = pd.Series("", index=df.index)

    def flag(mask, why):
        mask = mask.fillna(False)
        reason.loc[mask & (reason == "")] = why

    non_null_count = df[REQUIRED_COLUMNS].notna().sum(axis=1)
    flag(non_null_count < len(REQUIRED_COLUMNS) * 0.5, "too many missing fields")
    flag(df["tenure"].isna(), "tenure missing or not a number")
    flag(df["tenure"] < 0, "tenure is negative")
    flag(df["tenure"] > 100, "tenure unrealistically high")
    flag(~df["SeniorCitizen"].isin([0, 1]), "SeniorCitizen must be 0 or 1")
    flag(df["MonthlyCharges"].isna(), "MonthlyCharges missing or not a number")
    flag(df["MonthlyCharges"] <= 0, "MonthlyCharges is zero or negative")
    flag(df["MonthlyCharges"] > 500, "MonthlyCharges unrealistically high")
    flag(df["TotalCharges"] < 0, "TotalCharges is negative")

    valid_mask = reason == ""
    return valid_mask, reason


def render_prediction_card(proba, threshold):
    """Animated risk card: fills a probability bar, color-codes the outcome,
    and softly pulses on high risk. Pure CSS so it works in Colab + cloudflared."""
    flagged = proba >= threshold
    pct = proba * 100
    bar_color = "#E8543E" if flagged else "#2E7D4F"
    status_label = "HIGH RISK" if flagged else "LOW RISK"
    status_msg = "Recommend a retention offer" if flagged else "Customer looks stable"
    status_bg = "rgba(232, 84, 62, 0.08)" if flagged else "rgba(46, 125, 79, 0.08)"
    status_border = "#E8543E" if flagged else "#2E7D4F"
    pulse_class = "risk-pulse" if flagged else ""

    # Threshold marker as % of the bar width
    thr_pct = threshold * 100

    html = f"""
    <div class="result-card fade-in-up {pulse_class}" style="
        margin-top: 1.2rem;
        padding: 1.35rem 1.5rem;
        border-radius: 14px;
        border: 1.5px solid {status_border};
        background: {status_bg};
        box-shadow: 0 8px 24px rgba(19, 41, 75, 0.08);
    ">
      <div style="display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap;">
        <div>
          <div style="font-size:0.8rem; color:#6B7C93; letter-spacing:0.04em; text-transform:uppercase; font-weight:600;">
            Predicted churn probability
          </div>
          <div class="count-up" style="font-size:2.4rem; font-weight:700; color:#13294B; line-height:1.15; margin-top:4px;">
            {pct:.1f}%
          </div>
        </div>
        <div style="
            background:{status_border};
            color:#fff;
            font-size:0.78rem;
            font-weight:700;
            letter-spacing:0.06em;
            padding:8px 14px;
            border-radius:999px;
        ">{status_label}</div>
      </div>

      <div style="margin-top:1.1rem; position:relative;">
        <div style="
            height:12px;
            background:#E8EDF3;
            border-radius:999px;
            overflow:hidden;
        ">
          <div class="prob-fill" style="
              height:100%;
              width:{pct:.2f}%;
              background: linear-gradient(90deg, {bar_color}CC, {bar_color});
              border-radius:999px;
          "></div>
        </div>
        <div style="
            position:absolute;
            top:-3px;
            left:{thr_pct:.2f}%;
            width:2px;
            height:18px;
            background:#13294B;
            opacity:0.55;
            transform:translateX(-1px);
        " title="Decision threshold"></div>
        <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.75rem; color:#6B7C93;">
          <span>0%</span>
          <span>threshold {threshold:.0%}</span>
          <span>100%</span>
        </div>
      </div>

      <div style="margin-top:0.95rem; font-size:0.95rem; color:#13294B; font-weight:500;">
        {status_msg}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global professional motion + polish (works in Colab / Streamlit / trycloudflare)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page entrance */
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInRight {
      from { opacity: 0; transform: translateX(18px); }
      to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    @keyframes fillBar {
      from { width: 0% !important; }
    }
    @keyframes softPulse {
      0%, 100% { box-shadow: 0 8px 24px rgba(232, 84, 62, 0.12); }
      50%      { box-shadow: 0 8px 28px rgba(232, 84, 62, 0.28); }
    }
    @keyframes badgeIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 0.92; transform: translateY(0); }
    }

    .fade-in-up    { animation: fadeInUp 0.55s ease-out both; }
    .fade-in-right { animation: fadeInRight 0.6s ease-out both; }
    .fade-in       { animation: fadeIn 0.5s ease-out both; }

    .prob-fill {
      animation: fillBar 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .risk-pulse {
      animation: softPulse 2.2s ease-in-out infinite;
    }

    /* Title polish */
    h1 {
      letter-spacing: -0.02em !important;
    }

    /* Primary button hover */
    div.stButton > button {
      transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
      border-radius: 10px !important;
      font-weight: 600 !important;
    }
    div.stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(232, 84, 62, 0.25) !important;
    }
    div.stButton > button:active {
      transform: translateY(0);
    }

    /* Tabs underline feel */
    button[data-baseweb="tab"] {
      transition: color 0.2s ease !important;
    }

    /* Builder credit badge */
    .builder-badge {
      position: fixed;
