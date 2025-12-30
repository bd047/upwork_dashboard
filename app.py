import os
import re
import glob
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- Page + Style ----------------
st.set_page_config(page_title="Upwork Dashboard", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 14px;
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ✅ Use safe relative path (works on Streamlit Cloud + locally)
DATA_DIR = Path(__file__).parent / "data"
MONTH_ORDER = ["August", "September", "October", "November", "December"]


# ---------------- Helpers ----------------
def clean_colname(x: str) -> str:
    x = str(x).replace("\n", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip().lower()


def month_from_filename(name: str) -> str:
    base = Path(str(name)).stem.strip().lower()
    if "aug" in base:
        return "August"
    if "sep" in base:
        return "September"
    if "oct" in base:
        return "October"
    if "nov" in base:
        return "November"
    if "dec" in base:
        return "December"
    return base.capitalize()


def find_col(df: pd.DataFrame, contains=None, exact=None, exclude_contains=None):
    cols = list(df.columns)
    cleaned = {c: clean_colname(c) for c in cols}

    if exact:
        for e in exact:
            e = clean_colname(e)
            for c, cl in cleaned.items():
                if cl == e:
                    return c

    if contains:
        for token in contains:
            token = clean_colname(token)
            for c, cl in cleaned.items():
                if token in cl:
                    if exclude_contains and any(clean_colname(ex) in cl for ex in exclude_contains):
                        continue
                    return c
    return None


def to_bool(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()

    if s in ["yes", "y", "true", "1", "viewed", "replied", "invite", "invited"]:
        return True
    if s in [
        "no",
        "n",
        "false",
        "0",
        "",
        "none",
        "nan",
        "-",
        "no result",
        "not result",
        "not replied",
        "not viewed",
    ]:
        return False
    try:
        return float(s) != 0
    except:
        return True


def to_num(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "-"]:
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return 0.0


def non_empty(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() != "nan"


# ✅ MAIN LOGIC: BossDecision normalization
def normalize_boss_decision(x) -> str:
    """
    Maps your 'Jobs verified' column into:
    Approved / Rejected / None / Closed / Client Hired Another
    """
    if pd.isna(x):
        return "None"

    s = str(x).strip().lower()

    # common blanks / no-result => None
    if s in ["", "nan", "none", "-", "no result", "not result", "null"]:
        return "None"

    # closed
    if "closed" in s:
        return "Closed"

    # client hired someone else
    if "hired another" in s or "client hired another" in s or "client hired someone" in s:
        return "Client Hired Another"

    # approved/rejected
    if s in ["approved", "approve", "yes", "y"]:
        return "Approved"
    if s in ["rejected", "reject", "no", "n"]:
        return "Rejected"

    # if your sheet has exactly these values:
    if s == "hired another":
        return "Client Hired Another"

    # fallback
    return "None"


@st.cache_data
def load_all_excels(uploaded_files=None):
    """
    Loads Excel either from repo folder (data/*.xlsx) OR from uploaded files.
    Works both locally and on Streamlit Cloud.
    """
    # Prefer repo files if present, otherwise use uploaded files
    repo_files = sorted(DATA_DIR.glob("*.xlsx")) if DATA_DIR.exists() else []
    files = uploaded_files if uploaded_files else repo_files

    if not files:
        return pd.DataFrame(), [], {}

    frames = []
    detected = {}

    for f in files:
        # f can be Path (repo file) OR UploadedFile (streamlit uploader)
        if isinstance(f, (str, os.PathLike, Path)):
            fname = os.path.basename(str(f))
            raw = pd.read_excel(f)
        else:
            fname = f.name
            raw = pd.read_excel(f)

        raw = raw.loc[:, ~raw.columns.astype(str).str.lower().str.startswith("unnamed")]
        raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]

        month = month_from_filename(fname)

        # detect columns
        col_jobs = find_col(raw, contains=["jobs verfied", "jobs verified"])
        col_view = find_col(raw, contains=["view proposal"], exclude_contains=["time"])
        col_reply = find_col(raw, contains=["replied"])
        col_inv = find_col(raw, contains=["inivitations", "invitations", "invitaion"])
        col_written = find_col(raw, contains=["written proposal links", "written proposal"])
        col_boost = find_col(raw, contains=["boosted connects", "boosted"])
        col_total = find_col(raw, contains=["total connects", "total connect"])

        detected[fname] = {
            "Month": month,
            "Detected": {
                "jobs_verified": col_jobs,
                "view_yesno": col_view,
                "replied": col_reply,
                "inv_bool": col_inv,
                "written": col_written,
                "boosted": col_boost,
                "total_connects": col_total,
            },
        }

        out = pd.DataFrame(
            {
                "Month": month,
                "BossDecision": raw[col_jobs].apply(normalize_boss_decision) if col_jobs else "None",
                "Viewed": raw[col_view].apply(to_bool) if col_view else False,
                "Replied": raw[col_reply].apply(to_bool) if col_reply else False,
                "Invited": raw[col_inv].apply(to_bool) if col_inv else False,
                "Written": raw[col_written].apply(non_empty) if col_written else True,
                "BoostedConnects": raw[col_boost].apply(to_num) if col_boost else 0.0,
                "TotalConnects": raw[col_total].apply(to_num) if col_total else 0.0,
            }
        )

        frames.append(out)

    all_df = pd.concat(frames, ignore_index=True)
    months_present = [m for m in MONTH_ORDER if m in all_df["Month"].unique().tolist()]
    return all_df, months_present, detected


def rate(n, d):
    return 0 if d == 0 else round((n / d) * 100, 1)


# ---------------- Load ----------------
st.title("Upwork Proposal Dashboard (Aug → Dec)")
st.caption("Filter by Month to see month-only stats, or select Overall for combined analysis.")

repo_excels = list(DATA_DIR.glob("*.xlsx")) if DATA_DIR.exists() else []
uploaded = None

if not repo_excels:
    st.warning("No Excel files found in repo. Upload august.xlsx … december.xlsx below.")
    uploaded = st.file_uploader(
        "Upload monthly Excel files",
        type=["xlsx"],
        accept_multiple_files=True,
    )

df, months, detected = load_all_excels(uploaded_files=uploaded)

if df.empty:
    st.stop()

# ---------------- Sidebar Filters ----------------
with st.sidebar:
    st.header("Filters")
    selected_month = st.selectbox("Month View", ["Overall"] + months)

    # Keep a consistent order for decisions
    decision_order = ["Approved", "Rejected", "None", "Closed", "Client Hired Another"]
    existing = df["BossDecision"].dropna().unique().tolist()
    statuses = [x for x in decision_order if x in existing] + [x for x in existing if x not in decision_order]

    selected_status = st.multiselect("Boss Decision", statuses, default=statuses)

# ---------------- Apply Filters ----------------
filtered = df.copy()
if selected_month != "Overall":
    filtered = filtered[filtered["Month"] == selected_month]
filtered = filtered[filtered["BossDecision"].isin(selected_status)]

total = len(filtered)

# ---------------- KPIs ----------------
jobs_added = total
proposals_written = int(filtered["Written"].sum())

approved = int((filtered["BossDecision"] == "Approved").sum())
rejected = int((filtered["BossDecision"] == "Rejected").sum())
none_jobs = int((filtered["BossDecision"] == "None").sum())
closed_jobs = int((filtered["BossDecision"] == "Closed").sum())
hired_another = int((filtered["BossDecision"] == "Client Hired Another").sum())

viewed = int(filtered["Viewed"].sum())
replied = int(filtered["Replied"].sum())
invites = int(filtered["Invited"].sum())

# ✅ Your requirement: Interviewed = Replied
interviewed = replied

boosted = int((filtered["BoostedConnects"] > 0).sum())
total_connects = float(filtered["TotalConnects"].sum())

# Row 1
r1 = st.columns(6)
r1[0].metric("Jobs Added", jobs_added)
r1[1].metric("Proposals Written", proposals_written, f"{rate(proposals_written, jobs_added)}%")
r1[2].metric("Viewed (Client)", viewed, f"{rate(viewed, jobs_added)}%")
r1[3].metric("Replied (Client)", replied, f"{rate(replied, jobs_added)}%")
r1[4].metric("Interviewed", interviewed, f"{rate(interviewed, viewed)}%")
r1[5].metric("Invites", invites, f"{rate(invites, jobs_added)}%")

# Row 2
r2 = st.columns(6)
r2[0].metric("Approved (Boss)", approved, f"{rate(approved, proposals_written)}%")
r2[1].metric("Rejected (Boss)", rejected, f"{rate(rejected, proposals_written)}%")
r2[2].metric("No result ", none_jobs, f"{rate(none_jobs, proposals_written)}%")
r2[3].metric("Client Closed", closed_jobs, f"{rate(closed_jobs, proposals_written)}%")
r2[4].metric("Client Hired Another", hired_another, f"{rate(hired_another, proposals_written)}%")
r2[5].metric("Total Connects", int(total_connects))

# Bonus row (optional)
r3 = st.columns(2)
r3[0].metric("Boosted", boosted, f"{rate(boosted, jobs_added)}%")
r3[1].metric("Boosted Connects Used", int(filtered["BoostedConnects"].sum()))

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "📄 Data"])

with tab1:
    c1, c2 = st.columns(2)

    funnel_df = pd.DataFrame(
        {
            "Stage": ["Jobs Added", "Proposals Written", "Viewed", "Replied", "Invites"],
            "Count": [jobs_added, proposals_written, viewed, replied, invites],
        }
    )
    fig_funnel = px.bar(
        funnel_df,
        y="Stage",
        x="Count",
        orientation="h",
        title="Funnel (Jobs → Proposals → Viewed → Replied → Invites)",
        text="Count",
    )
    fig_funnel.update_layout(yaxis={"categoryorder": "total ascending"})
    c1.plotly_chart(
        fig_funnel,
        use_container_width=True,
        key=f"funnel_{selected_month}_{'_'.join(selected_status)}",
    )

    dec_counts = filtered["BossDecision"].value_counts(dropna=False).reset_index()
    dec_counts.columns = ["BossDecision", "Count"]
    fig_dec = px.pie(dec_counts, names="BossDecision", values="Count", title="Boss Decision Breakdown")
    c2.plotly_chart(
        fig_dec,
        use_container_width=True,
        key=f"boss_pie_{selected_month}_{'_'.join(selected_status)}",
    )

with tab2:
    if selected_month == "Overall":
        trend = (
            df[df["BossDecision"].isin(selected_status)]
            .groupby("Month")
            .agg(
                Jobs=("Month", "count"),
                Written=("Written", "sum"),
                Viewed=("Viewed", "sum"),
                Replied=("Replied", "sum"),
                Invited=("Invited", "sum"),
                Approved=("BossDecision", lambda s: (s == "Approved").sum()),
                Rejected=("BossDecision", lambda s: (s == "Rejected").sum()),
                NoneJobs=("BossDecision", lambda s: (s == "None").sum()),  # ✅ fixed
                Closed=("BossDecision", lambda s: (s == "Closed").sum()),
                HiredAnother=("BossDecision", lambda s: (s == "Client Hired Another").sum()),
                Connects=("TotalConnects", "sum"),
            )
            .reset_index()
        )

        trend["Month"] = pd.Categorical(trend["Month"], categories=MONTH_ORDER, ordered=True)
        trend = trend.sort_values("Month")

        st.plotly_chart(
            px.line(
                trend,
                x="Month",
                y=["Jobs", "Written", "Viewed", "Replied", "Invited"],
                markers=True,
                title="Monthly Funnel Trend (Aug → Dec)",
            ),
            use_container_width=True,
            key="trend_line_main",
        )

        st.plotly_chart(
            px.bar(trend, x="Month", y="Connects", title="Total Connects Used by Month", text="Connects"),
            use_container_width=True,
            key="trend_connects",
        )
    else:
        st.info("Trends are only meaningful in Overall mode. Select Overall in sidebar.")

with tab3:
    st.dataframe(filtered, use_container_width=True, height=520)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"upwork_{selected_month.lower()}_filtered.csv",
        mime="text/csv",
    )

with st.expander("Show last 5 rows of the selected month"):
    st.dataframe(filtered.tail(5), use_container_width=True)
