import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="HRMS NLP — Medcross TB Clinic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');

[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"] { background: #1a2744; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,h2,h3 { color: #fff !important; }

.nlp-title {
    background: linear-gradient(135deg, #1a2744 0%, #1e4d7b 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem;
}
.nlp-title h1 { color: white; margin: 0; font-size: 1.6rem; }
.nlp-title p  { color: #94b8d8; margin: 0.3rem 0 0; font-size: 0.9rem; }

.pipeline-box {
    background: white; border-radius: 10px; padding: 1rem 1.2rem;
    border-left: 4px solid #2563eb; margin-bottom: 0.8rem;
}
.pipeline-box h4 { margin: 0 0 0.4rem; color: #1a2744; font-size: 0.85rem; }
.pipeline-box p  { margin: 0; font-size: 0.8rem; color: #475569; }

.token-chip {
    display: inline-block; background: #e0f2fe; color: #0c4a6e;
    border-radius: 20px; padding: 3px 10px; margin: 2px;
    font-size: 0.78rem; font-weight: 600;
}
.name-chip {
    display: inline-block; background: #fef3c7; color: #78350f;
    border-radius: 20px; padding: 3px 10px; margin: 2px;
    font-size: 0.78rem; font-weight: 600; border: 1px solid #f59e0b;
}
.match-chip {
    display: inline-block; background: #dcfce7; color: #14532d;
    border-radius: 20px; padding: 3px 10px; margin: 2px;
    font-size: 0.78rem; font-weight: 600; border: 1px solid #22c55e;
}
.status-on-leave { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.status-delayed  { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.status-present  { background:#dcfce7; color:#14532d; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

.result-banner {
    background: #1a2744; color: white; border-radius: 10px;
    padding: 1rem 1.4rem; margin: 1rem 0;
}
.result-banner .parsed { color: #7dd3fc; font-size:0.82rem; }

.card {
    background: white; border-radius: 12px;
    padding: 1.2rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-header { font-size:0.78rem; font-weight:700; letter-spacing:0.08em;
    text-transform:uppercase; color:#64748b; margin-bottom:0.8rem; }

.coverage-covered     { background:#dcfce7; color:#14532d; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.coverage-help        { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.coverage-understaffed{ background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.coverage-reschedule  { background:#e0f2fe; color:#0c4a6e; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }

.assign-normal      { font-weight:600; color:#1a2744; }
.assign-replacement { background:#fed7aa; color:#7c2d12; padding:2px 8px;
    border-radius:12px; font-size:0.82rem; font-weight:700; }
.assign-helper      { background:#fde68a; color:#78350f; padding:2px 8px;
    border-radius:12px; font-size:0.82rem; font-weight:700; }

.stat-card { background:white; border-radius:10px; padding:1rem;
    text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
.stat-num  { font-size:2rem; font-weight:700; }
.stat-lbl  { font-size:0.75rem; color:#64748b; text-transform:uppercase;
    letter-spacing:0.06em; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────
STAFF = [
    "Anusha","Pooja","Sachin","Sufiyaan","Mahesh","Bhawna","Kuldeep",
    "Ayush","Payal","Dr Kaushal","Praveen","Neeru","Seema","Jaya",
    "Anupam","Dr Zeeshan","Seema Dubey","Deepak","Gaurav","Seetaram",
    "Sumit","Himanshu","Vijay","Kunal"
]

TASKS = [
    dict(name="Decant and Micro",                      needed=3, primaries=["Anusha","Pooja","Sachin"],     secondaries=["Sufiyaan"],              priority="On the day"),
    dict(name="DNA",                                   needed=1, primaries=["Mahesh"],                      secondaries=["Bhawna","Kuldeep"],       priority="On the day"),
    dict(name="PCR",                                   needed=1, primaries=["Ayush"],                       secondaries=["Payal","Dr Kaushal","Praveen"], priority="On the day"),
    dict(name="LPA Blot",                              needed=1, primaries=["Neeru"],                       secondaries=["Seema","Praveen"],        priority="On the day"),
    dict(name="Culture 1st Part",                      needed=1, primaries=["Jaya"],                        secondaries=["Ayush","Bhawna"],         priority="On the day"),
    dict(name="Culture 2nd Part",                      needed=1, primaries=["Anupam"],                      secondaries=["Pooja"],                  priority="On the day"),
    dict(name="DST",                                   needed=1, primaries=["Bhawna"],                      secondaries=["Neeru","Dr Zeeshan"],     priority="On the day"),
    dict(name="NAAT",                                  needed=1, primaries=["Kuldeep"],                     secondaries=["Anupam","Seema Dubey"],   priority="On the day"),
    dict(name="Sample Collection",                     needed=1, primaries=["Bhawna"],                      secondaries=["Seema Dubey"],            priority="Next day"),
    dict(name="Reagent Prep & Microscopy Supervision", needed=2, primaries=["Praveen","Deepak"],             secondaries=["Anupam","Gaurav"],        priority="Next day"),
    dict(name="Store and Supervision",                 needed=1, primaries=["Seetaram"],                    secondaries=["Gaurav"],                 priority="Next day"),
    dict(name="Data Entry",                            needed=3, primaries=["Sumit","Seema","Himanshu"],     secondaries=["Seema","Himanshu"],       priority="Next day"),
    dict(name="Cleaning Staff",                        needed=2, primaries=["Vijay","Kunal"],                secondaries=["Seetaram"],               priority="Next day"),
]

# ── Session state ─────────────────────────────────────────────────
if "staff_status" not in st.session_state:
    st.session_state.staff_status = {s: "Present" for s in STAFF}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_parse" not in st.session_state:
    st.session_state.last_parse = None

# ═══════════════════════════════════════════════════════════════
# NLP PIPELINE (pure Python — no external models needed)
# ═══════════════════════════════════════════════════════════════

STATUS_PATTERNS = {
    "On Leave": [
        r"\bon\s+leave\b", r"\babsent\b", r"\bsick\b", r"\bnot\s+coming\b",
        r"\bwon'?t\s+come\b", r"\bwill\s+not\s+come\b", r"\bcannot\s+come\b",
        r"\bcan'?t\s+come\b", r"\bleave\b", r"\bgoing\s+away\b",
        r"\boff\s+today\b", r"\btaking\s+off\b",
    ],
    "Delayed": [
        r"\bdelayed?\b", r"\blate\b", r"\bcoming\s+late\b",
        r"\bwill\s+be\s+late\b", r"\bslightly\s+late\b",
        r"\bbit\s+late\b", r"\brunning\s+late\b",
    ],
    "Present": [
        r"\bpresent\b", r"\bback\b", r"\breturned?\b",
        r"\bcoming\b", r"\bavailable\b", r"\bjoined\b",
        r"\brecovered?\b", r"\bin\s+office\b",
    ],
}

def step1_tokenize(text: str) -> list[str]:
    """Step 1 — Tokenization: split on whitespace, strip punctuation."""
    tokens = re.findall(r'\b\w+(?:\.\s*\w+)?\b', text)
    return tokens

def step2_candidate_names(text: str) -> list[str]:
    """
    Step 2 — Name Candidate Extraction (custom NER heuristic):
    Extract capitalised words / bigrams — potential person names.
    """
    single = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    bigrams = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
    # 'Dr Kaushal' style
    titled = re.findall(r'\bDr\.?\s+[A-Z][a-z]+\b', text, re.IGNORECASE)
    candidates = list(set(titled + bigrams + single))
    # Remove common non-name capitalised words
    stopwords = {"Today","Tomorrow","Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday","On","The","Leave"}
    return [c for c in candidates if c not in stopwords and len(c) > 2]

def step3_fuzzy_match(candidates: list[str], threshold=72) -> list[tuple]:
    """
    Step 3 — Fuzzy Matching (domain-specific NER against staff roster).
    Returns list of (raw_candidate, matched_staff_name, score).
    """
    matched = []
    seen = set()
    for cand in candidates:
        result = process.extractOne(cand, STAFF, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            name = result[0]
            if name not in seen:
                seen.add(name)
                matched.append((cand, name, result[1]))
    return matched

def step4_intent(text: str) -> str | None:
    """
    Step 4 — Intent Classification via keyword pattern matching.
    Returns the detected status or None.
    """
    lower = text.lower()
    for status, patterns in STATUS_PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                return status
    return None

def step5_extract_note(text: str) -> str:
    """Step 5 — Slot Filling: extract reason/note if given."""
    m = re.search(
        r'(?:because|due to|as|since|reason:?)\s+(.+?)(?:\.|$)',
        text, re.IGNORECASE
    )
    return m.group(1).strip() if m else ""

def run_nlp_pipeline(text: str) -> dict:
    """Run full pipeline and return all intermediate steps."""
    tokens    = step1_tokenize(text)
    candidates = step2_candidate_names(text)
    matches   = step3_fuzzy_match(candidates)
    status    = step4_intent(text)
    note      = step5_extract_note(text)
    matched_names = [m[1] for m in matches]
    return dict(
        tokens=tokens,
        candidates=candidates,
        matches=matches,
        matched_names=matched_names,
        status=status,
        note=note,
        success=bool(matched_names and status),
    )

# ═══════════════════════════════════════════════════════════════
# WORK ASSIGNMENT ENGINE
# ═══════════════════════════════════════════════════════════════

def compute_assignments(staff_status: dict) -> list[dict]:
    rows = []
    for task in TASKS:
        pool = []
        for p in task["primaries"][:task["needed"]]:
            st = staff_status.get(p, "Present")
            if st != "On Leave":
                pool.append(dict(name=p, status=st, replacement=False))

        for s in task["secondaries"]:
            if len(pool) >= task["needed"]: break
            st = staff_status.get(s, "Present")
            existing_names = [x["name"] for x in pool]
            if st != "On Leave" and s not in existing_names:
                pool.append(dict(name=s, status=st, replacement=True))

        assigned = pool[:task["needed"]]
        assigned_names = [x["name"] for x in assigned]
        has_delay = any(x["status"] == "Delayed" for x in assigned)

        helper = ""
        if has_delay:
            for s in task["secondaries"]:
                if s not in assigned_names and staff_status.get(s,"Present") == "Present":
                    helper = s
                    break

        avail = len(pool)
        if avail >= task["needed"]:
            coverage = "⚠️ Help Needed" if has_delay else "✅ Covered"
        else:
            coverage = "🚨 Understaffed" if task["priority"] == "On the day" else "📅 Reschedule"

        rows.append(dict(
            task=task["name"], priority=task["priority"],
            needed=task["needed"], assigned=assigned,
            helper=helper, coverage=coverage,
        ))
    return rows

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — NLP explainer
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NLP Pipeline")
    st.markdown("---")
    st.markdown("""
**Step 1 — Tokenization**
Splits input into individual word tokens using regex.

**Step 2 — Name Candidate Extraction**
Finds capitalised words (heuristic NER). Works like a lightweight Named Entity Recogniser.

**Step 3 — Fuzzy Matching**
Matches candidates to the staff roster using RapidFuzz `token_sort_ratio`. Handles typos and partial names.

**Step 4 — Intent Classification**
Regex pattern matching detects status intent: On Leave / Delayed / Present.

**Step 5 — Slot Filling**
Extracts reason/notes if provided ("because…", "due to…").
""")
    st.markdown("---")
    st.markdown("**Try these examples:**")
    examples = [
        "Mahesh is on leave today",
        "Anusha and Pooja are absent",
        "Bhawna will be late due to traffic",
        "Dr Kaushal is sick today",
        "Kuldeep and Sachin won't come",
        "Seema Dubey is back",
        "Mark Ayush as delayed",
    ]
    for ex in examples:
        if st.button(f"💬 {ex}", key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🔄 Reset all to Present", use_container_width=True):
        st.session_state.staff_status = {s: "Present" for s in STAFF}
        st.session_state.history = []
        st.session_state.last_parse = None
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="nlp-title">
  <h1>🏥 HRMS NLP Command Centre</h1>
  <p>Medcross TB Clinic · Natural Language Staff Management · Powered by Custom NLP Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────
ss = st.session_state.staff_status
n_present  = sum(1 for v in ss.values() if v == "Present")
n_leave    = sum(1 for v in ss.values() if v == "On Leave")
n_delayed  = sum(1 for v in ss.values() if v == "Delayed")
assignments = compute_assignments(ss)
n_covered  = sum(1 for a in assignments if "Covered" in a["coverage"])
n_help     = sum(1 for a in assignments if "Help" in a["coverage"])
n_under    = sum(1 for a in assignments if "Understaffed" in a["coverage"])
n_resched  = sum(1 for a in assignments if "Reschedule" in a["coverage"])

col1,col2,col3,col4,col5,col6 = st.columns(6)
stats = [
    (col1, str(len(STAFF)), "Total Staff",   "#1a2744","white"),
    (col2, str(n_present),  "Present",        "#166534","white"),
    (col3, str(n_leave),    "On Leave",       "#991b1b","white"),
    (col4, str(n_delayed),  "Delayed",        "#92400e","white"),
    (col5, str(n_covered),  "Tasks Covered",  "#14532d","white"),
    (col6, str(n_under+n_resched), "At Risk", "#7c3aed","white"),
]
for col,num,lbl,bg,fg in stats:
    col.markdown(f"""
    <div class="stat-card" style="border-top:4px solid {bg};">
      <div class="stat-num" style="color:{bg};">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── NLP Input ─────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.text_input(
    "🗣️  Type a natural language command:",
    value=prefill,
    placeholder='e.g. "Mahesh is on leave today" or "Anusha and Pooja are absent due to training"',
    key="nlp_input"
)
run_btn = st.button("▶  Parse & Apply", type="primary", use_container_width=False)

# ── Run pipeline ──────────────────────────────────────────────
if run_btn and user_input.strip():
    result = run_nlp_pipeline(user_input.strip())
    st.session_state.last_parse = {"input": user_input.strip(), "result": result}

    if result["success"]:
        for name in result["matched_names"]:
            st.session_state.staff_status[name] = result["status"]
        st.session_state.history.insert(0, {
            "cmd": user_input.strip(),
            "names": result["matched_names"],
            "status": result["status"],
        })

# ── Show NLP Pipeline Breakdown ──────────────────────────────
if st.session_state.last_parse:
    p = st.session_state.last_parse
    r = p["result"]
    st.markdown("### 🔬 NLP Pipeline Breakdown")

    c1,c2 = st.columns(2)

    with c1:
        # Step 1 - tokens
        token_html = " ".join(f'<span class="token-chip">{t}</span>' for t in r["tokens"])
        st.markdown(f"""<div class="pipeline-box">
            <h4>Step 1 — Tokenization &nbsp; <code style="font-size:0.75rem;color:#6366f1;">regex.findall()</code></h4>
            {token_html}
        </div>""", unsafe_allow_html=True)

        # Step 2 - candidates
        cand_html = " ".join(f'<span class="name-chip">{c}</span>' for c in r["candidates"]) or "<i style='color:#94a3b8;font-size:0.8rem;'>none found</i>"
        st.markdown(f"""<div class="pipeline-box" style="border-left-color:#f59e0b;">
            <h4>Step 2 — Name Candidate Extraction &nbsp; <code style="font-size:0.75rem;color:#6366f1;">Capitalised heuristic</code></h4>
            {cand_html}
        </div>""", unsafe_allow_html=True)

        # Step 3 - fuzzy
        if r["matches"]:
            match_html = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<span class="name-chip">{m[0]}</span>'
                f'<span style="color:#64748b;font-size:0.8rem;">→ fuzzy score {m[2]}% →</span>'
                f'<span class="match-chip">✓ {m[1]}</span></div>'
                for m in r["matches"]
            )
        else:
            match_html = "<i style='color:#94a3b8;font-size:0.8rem;'>No match found — try different name</i>"
        st.markdown(f"""<div class="pipeline-box" style="border-left-color:#22c55e;">
            <h4>Step 3 — Fuzzy Name Matching &nbsp; <code style="font-size:0.75rem;color:#6366f1;">RapidFuzz token_sort_ratio</code></h4>
            {match_html}
        </div>""", unsafe_allow_html=True)

    with c2:
        # Step 4 - intent
        intent_color = {"On Leave":"#dc2626","Delayed":"#d97706","Present":"#16a34a"}.get(r["status"],"#64748b")
        intent_badge = f'<span style="background:{intent_color};color:white;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9rem;">{r["status"] or "Not detected"}</span>' if r["status"] else '<span style="color:#94a3b8;font-size:0.8rem;"><i>status not detected</i></span>'
        st.markdown(f"""<div class="pipeline-box" style="border-left-color:#8b5cf6;">
            <h4>Step 4 — Intent Classification &nbsp; <code style="font-size:0.75rem;color:#6366f1;">Keyword pattern matching</code></h4>
            <p style="margin:0.4rem 0;">Detected status:&nbsp; {intent_badge}</p>
            <p style="color:#94a3b8;font-size:0.75rem;margin-top:6px;">Matched pattern from {len([pp for v in STATUS_PATTERNS.values() for pp in v])} regex rules</p>
        </div>""", unsafe_allow_html=True)

        # Step 5 - slot filling
        note_text = f'<span style="color:#1e293b;font-weight:600;">"{r["note"]}"</span>' if r["note"] else '<i style="color:#94a3b8;">No reason/note detected</i>'
        st.markdown(f"""<div class="pipeline-box" style="border-left-color:#ec4899;">
            <h4>Step 5 — Slot Filling &nbsp; <code style="font-size:0.75rem;color:#6366f1;">Reason extraction</code></h4>
            <p>Note: {note_text}</p>
        </div>""", unsafe_allow_html=True)

        # Final result
        if r["success"]:
            names_str = ", ".join(r["matched_names"])
            st.markdown(f"""<div style="background:#1a2744;border-radius:10px;padding:1rem 1.2rem;">
                <p style="color:#7dd3fc;font-size:0.75rem;margin:0 0 4px;text-transform:uppercase;letter-spacing:0.08em;">✅ Action Applied</p>
                <p style="color:white;font-size:0.95rem;font-weight:600;margin:0;">{names_str}</p>
                <p style="color:#94b8d8;font-size:0.85rem;margin:4px 0 0;">Status set to <b style="color:white;">{r["status"]}</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:#7f1d1d;border-radius:10px;padding:1rem 1.2rem;">
                <p style="color:#fca5a5;font-size:0.85rem;margin:0;">❌ Could not parse. Try: <i>"[Name] is on leave"</i></p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

# ── Staff Status + Work Assignment ────────────────────────────
left, right = st.columns([1, 1.5])

with left:
    st.markdown("### 👥 Staff Status")
    rows_html = ""
    for name in STAFF:
        status = st.session_state.staff_status[name]
        badge_cls = {"Present":"status-present","On Leave":"status-on-leave","Delayed":"status-delayed"}.get(status,"status-present")
        rows_html += f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:6px 0;border-bottom:0.5px solid #f1f5f9;">
            <span style="font-size:0.88rem;color:#1e293b;font-weight:500;">{name}</span>
            <span class="{badge_cls}">{status}</span>
        </div>"""
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

with right:
    st.markdown("### ⚙️ Work Assignment (Live)")
    assignments = compute_assignments(st.session_state.staff_status)
    for a in assignments:
        cov = a["coverage"]
        cov_cls = "coverage-covered" if "Covered" in cov else \
                  "coverage-help" if "Help" in cov else \
                  "coverage-understaffed" if "Understaffed" in cov else \
                  "coverage-reschedule"

        # Build assigned names with replacement highlight
        names_html = ""
        for p in a["assigned"]:
            cls = "assign-replacement" if p["replacement"] else "assign-normal"
            names_html += f'<span class="{cls}">{p["name"]}</span> '
        if a["helper"]:
            names_html += f'<span class="assign-helper">+ {a["helper"]} (helper)</span>'
        if not names_html:
            names_html = '<span style="color:#94a3b8;font-size:0.8rem;font-style:italic;">Unassigned</span>'

        pri_color = "#dc2626" if a["priority"] == "On the day" else "#0369a1"
        st.markdown(f"""<div style="background:white;border-radius:8px;padding:10px 14px;
            margin-bottom:6px;border-left:3px solid {pri_color};">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px;">
                <span style="font-size:0.88rem;font-weight:600;color:#1e293b;">{a['task']}</span>
                <span class="{cov_cls}">{cov}</span>
            </div>
            <div style="font-size:0.8rem;color:#64748b;margin-bottom:4px;">
                {a['priority']} · Needs {a['needed']} staff
            </div>
            <div style="font-size:0.85rem;">{names_html}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px;padding:8px;background:#f8fafc;border-radius:6px;">
        🟠 <b>Orange</b> = Replacement worker (covering for absent primary) &nbsp;|&nbsp;
        🟡 <b>Yellow</b> = Extra helper (supporting delayed worker)
    </div>""", unsafe_allow_html=True)

# ── Command History ───────────────────────────────────────────
if st.session_state.history:
    st.markdown("### 📜 Command History")
    for i, h in enumerate(st.session_state.history[:5]):
        status_color = {"On Leave":"#dc2626","Delayed":"#d97706","Present":"#16a34a"}.get(h["status"],"#64748b")
        st.markdown(f"""<div style="background:white;border-radius:8px;padding:8px 14px;
            margin-bottom:5px;display:flex;justify-content:space-between;align-items:center;
            border:0.5px solid #e2e8f0;">
            <span style="color:#64748b;font-size:0.8rem;">💬</span>
            <span style="font-style:italic;font-size:0.85rem;color:#334155;flex:1;margin:0 12px;">"{h['cmd']}"</span>
            <span style="font-size:0.82rem;font-weight:600;color:#1e293b;">{", ".join(h["names"])}</span>
            <span style="background:{status_color};color:white;padding:2px 10px;border-radius:12px;
                font-size:0.78rem;font-weight:600;margin-left:10px;">{h["status"]}</span>
        </div>""", unsafe_allow_html=True)
