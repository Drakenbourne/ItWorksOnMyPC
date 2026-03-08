import time
from pathlib import Path
from typing import List

import streamlit as st

from device_utils import get_device
from llm_utils import (
    analyze_candidate_profile,
    evaluate_answer,
    generate_reference_answer,
    get_evaluator_backend_label,
    warmup_evaluator_model,
)
from rag_engine import get_rag_engine, warmup_rag_system
from resume_parser import extract_text_from_upload


st.set_page_config(page_title="AI Interview Preparation Bot", page_icon=":dart:", layout="wide")

st.markdown(
    """
    <style>
        .hero { padding: 1rem 1.25rem; border-radius: 14px; background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; border: 1px solid #334155; margin-bottom: 1rem; }
        .question-card { padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid #334155; background: #0b1220; margin: 0.5rem 0 1rem 0; }
        .question-source { font-size: 0.85rem; color: #93c5fd; margin-bottom: 0.4rem; }
        .summary-card { padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid #1f2937; background: #0f172a; margin-top: 1rem; }
        .step-chip { display: inline-block; padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid #334155; margin-right: 0.4rem; font-size: 0.8rem; color: #cbd5e1; background: #111827; }
        .loader-shell { margin-top: 2rem; padding: 1.2rem; border-radius: 16px; border: 1px solid #334155; background: linear-gradient(120deg, #020617 0%, #0f172a 100%); }
        .loader-bar { position: relative; height: 10px; border-radius: 999px; background: #1e293b; overflow: hidden; margin: 0.6rem 0 0.2rem 0; }
        .loader-bar::before { content: ""; position: absolute; inset: 0; width: 45%; background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8); animation: glide 1.2s ease-in-out infinite; }
        .runtime-chip { display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.78rem; border: 1px solid #334155; color:#cbd5e1; background:#111827; margin-left:0.5rem; }
        @keyframes glide { 0% { transform: translateX(-120%); } 50% { transform: translateX(70%); } 100% { transform: translateX(220%); } }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = Path(__file__).parent / "data" / "interview_questions.txt"
ROLES = ["Backend", "AI Engineer", "Data Scientist", "Fullstack"]
ROLE_MODES = ["Select Role", "Infer From Job Description"]
INTERVIEW_TONES = ["Easy", "Medium", "Strict"]

ROLE_CATEGORY_HINTS = {
    "Backend": ["Backend Development", "System Design", "Software Engineering"],
    "AI Engineer": ["Machine Learning", "Data Science", "System Design"],
    "Data Scientist": ["Data Science", "Machine Learning"],
    "Fullstack": ["Software Engineering", "Backend Development", "System Design"],
}

ROLE_SKILL_HINTS = {
    "Backend": ["api", "backend", "database", "sql", "microservice", "docker", "kubernetes", "redis", "system design"],
    "AI Engineer": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "mlops", "llm", "vector"],
    "Data Scientist": ["data", "statistics", "python", "pandas", "numpy", "scikit", "experiment", "analytics"],
    "Fullstack": ["frontend", "backend", "react", "node", "javascript", "typescript", "api", "database", "system design"],
}

FULLSTACK_BLOCKLIST = {"machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "mlops", "hadoop", "spark"}


for key, default in {
    "runtime_ready": False,
    "resume_text": "",
    "jd_text": "",
    "skills": [],
    "analysis_done": False,
    "analysis_role": "",
    "analysis_focus": "",
    "interview_role": "",
    "questions": [],
    "question_sources": [],
    "current_index": 0,
    "active_question_index": -1,
    "current_answer_input": "",
    "answers": [],
    "evaluations": [],
    "interview_started": False,
    "pending_retake": False,
    "pending_retake_role": "",
    "pending_retake_count": 6,
    "typed_done": {},
    "reference_cache": {},
    "interview_tone": "Medium",
    "role_mode": ROLE_MODES[0],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

evaluator_backend = get_evaluator_backend_label()


def _render_typing(
    container,
    text: str,
    anim_key: str,
    words_per_minute: int = 40,
    max_duration_sec: float = 12.0,
) -> None:
    text = (text or "").strip()
    if not text:
        return

    typed_done = st.session_state.get("typed_done", {})
    if typed_done.get(anim_key):
        container.write(text)
        return

    chars_per_second = max(1.0, (words_per_minute * 5) / 60.0)
    total_chars = len(text)
    chunk_size = 1
    estimated_duration = total_chars / chars_per_second
    if estimated_duration > max_duration_sec:
        chunk_size = max(1, int(total_chars / (chars_per_second * max_duration_sec)))
    delay = max(0.01, chunk_size / chars_per_second)

    for i in range(0, total_chars, chunk_size):
        container.write(text[: i + chunk_size])
        time.sleep(delay)

    typed_done[anim_key] = True
    st.session_state.typed_done = typed_done


def _filter_skills_for_role(skills: List[str], role: str) -> List[str]:
    role_hints = ROLE_SKILL_HINTS.get(role, [])
    filtered = []
    for skill in skills:
        s = skill.lower()
        if role == "Fullstack" and any(token in s for token in FULLSTACK_BLOCKLIST):
            continue
        if any(h in s for h in role_hints):
            filtered.append(skill)
    return filtered


def _start_interview(role: str, count: int, jd_context: str = "") -> None:
    rag_engine = get_rag_engine(str(DATA_FILE))
    filtered_skills = _filter_skills_for_role(st.session_state.skills, role)
    skill_query = ", ".join(filtered_skills[:8])
    jd_query = (jd_context or "").strip()
    if len(jd_query) > 800:
        jd_query = jd_query[:800]

    query_parts = [f"Role: {role}"]
    if skill_query:
        query_parts.append(f"Skills: {skill_query}")
    if jd_query:
        query_parts.append(f"Job Description Context: {jd_query}")
    retrieval_query = ". ".join(query_parts)

    questions = rag_engine.retrieve_questions(
        retrieval_query,
        top_k=count,
        preferred_categories=ROLE_CATEGORY_HINTS.get(role, []),
    )

    st.session_state.questions = questions
    st.session_state.question_sources = ["RAG"] * len(questions)
    st.session_state.interview_role = role
    st.session_state.current_index = 0
    st.session_state.active_question_index = 0
    st.session_state.current_answer_input = ""
    st.session_state.answers = [""] * len(questions)
    st.session_state.evaluations = [None] * len(questions)
    st.session_state.interview_started = True
    st.session_state.typed_done = {}
    st.session_state.reference_cache = {}


if not st.session_state.runtime_ready:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Preparing AI Interview Bot</h2>
            <p style="margin:0.35rem 0 0 0;">Loading models and vector index. This happens once at startup.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="loader-shell">
            <div style="font-size:0.95rem;color:#cbd5e1;">Initializing runtime components...</div>
            <div class="loader-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress = st.progress(0)
    status = st.empty()
    steps = [
        ("Loading embedding model and building FAISS index...", lambda: warmup_rag_system(str(DATA_FILE))),
        (f"Loading evaluator backend: {evaluator_backend}...", warmup_evaluator_model),
    ]

    try:
        for i, (msg, fn) in enumerate(steps, start=1):
            status.info(msg)
            fn()
            progress.progress(int((i / len(steps)) * 100))
            time.sleep(0.15)
    except Exception as exc:
        st.error(f"Startup initialization failed: {exc}")
        st.stop()

    status.success("All models loaded. Launching interface...")
    time.sleep(0.2)
    st.session_state.runtime_ready = True
    st.rerun()

runtime_device = get_device().upper()
st.markdown(
    f"""
    <div class="hero">
        <h2 style="margin:0;">AI Interview Preparation Bot
            <span class="runtime-chip">Runtime: {runtime_device}</span>
            <span class="runtime-chip">Evaluator: {evaluator_backend}</span>
        </h2>
        <p style="margin:0.4rem 0 0 0;">Resume + JD aware role inference, tone-controlled interviewing, and AI-grounded evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="step-chip">1. Resume + Optional JD</span>
    <span class="step-chip">2. AI Profile Analysis</span>
    <span class="step-chip">3. Retrieve Questions (RAG)</span>
    <span class="step-chip">4. Answer + Tone-based Evaluation</span>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("Workflow")
    role_mode = st.radio("Interview Setup", ROLE_MODES, key="role_mode", horizontal=True)
    manual_role = st.selectbox("Target Role", ROLES, key="manual_role", disabled=role_mode != ROLE_MODES[0])
    st.select_slider("Interviewer Tone", options=INTERVIEW_TONES, key="interview_tone")
    question_count = st.slider("Number of Questions", min_value=3, max_value=12, value=6)
    uploaded_file = st.file_uploader("Step 1: Upload Resume (PDF/TXT)", type=["pdf", "txt"])
    jd_uploaded_file = st.file_uploader("Optional: Upload Job Description (PDF/TXT)", type=["pdf", "txt"])
    jd_pasted = st.text_area(
        "Or paste Job Description text",
        value=st.session_state.jd_text,
        height=120,
        placeholder="Paste JD text here if you don't upload a JD file...",
    )

    if uploaded_file is not None:
        with st.spinner("Parsing resume..."):
            st.session_state.resume_text = extract_text_from_upload(uploaded_file)

    if jd_uploaded_file is not None:
        with st.spinner("Parsing job description..."):
            st.session_state.jd_text = extract_text_from_upload(jd_uploaded_file)
    elif jd_pasted.strip():
        st.session_state.jd_text = jd_pasted.strip()

    if st.button("Step 2: Analyze Role + Skills", use_container_width=True):
        if not st.session_state.resume_text:
            st.warning("Upload a resume first.")
        elif role_mode == ROLE_MODES[1] and not st.session_state.jd_text.strip():
            st.warning("Upload or paste a job description to infer role from JD.")
        else:
            selected_for_analysis = manual_role if role_mode == ROLE_MODES[0] else ""
            with st.spinner("AI is analyzing resume and job context..."):
                analysis = analyze_candidate_profile(
                    resume_text=st.session_state.resume_text,
                    job_description=st.session_state.jd_text,
                    selected_role=selected_for_analysis,
                )
            st.session_state.skills = analysis.get("skills", [])
            st.session_state.analysis_role = analysis.get("role", manual_role)
            st.session_state.analysis_focus = analysis.get("jd_focus", "")
            st.session_state.analysis_done = True

    if st.session_state.analysis_done:
        st.success("Profile analyzed")
        st.write(f"Role for interview: **{st.session_state.analysis_role or manual_role}**")
        if st.session_state.analysis_focus:
            _render_typing(
                st.empty(),
                f"JD focus: {st.session_state.analysis_focus}",
                f"focus_{hash(st.session_state.analysis_focus)}",
                words_per_minute=40,
            )

    if st.session_state.skills:
        skill_summary = ", ".join(st.session_state.skills)
        _render_typing(
            st.empty(),
            f"AI-detected skills: {skill_summary}",
            f"skills_{hash(skill_summary)}",
            words_per_minute=40,
        )

    if st.button("Step 3: Start Interview", use_container_width=True):
        if not st.session_state.analysis_done:
            st.warning("Run Step 2 first so AI can set interview role and skills.")
        else:
            role_for_interview = st.session_state.analysis_role or manual_role
            _start_interview(role_for_interview, question_count, jd_context=st.session_state.jd_text)
            st.rerun()

selected_role = st.session_state.analysis_role or st.session_state.get("manual_role", ROLES[0])

if st.session_state.pending_retake:
    role = st.session_state.pending_retake_role or selected_role
    count = int(st.session_state.pending_retake_count or question_count)
    st.session_state.pending_retake = False
    _start_interview(role, count, jd_context=st.session_state.jd_text)
    st.rerun()

with right_col:
    st.subheader("Interview Panel")
    tone_value = (st.session_state.interview_tone or "Medium").strip().lower()
    interview_role = st.session_state.interview_role or selected_role

    if not st.session_state.interview_started:
        st.info("Complete steps on the left to begin interview mode.")
    else:
        questions = st.session_state.questions
        sources = st.session_state.question_sources
        idx = st.session_state.current_index

        if st.session_state.active_question_index != idx:
            st.session_state.current_answer_input = st.session_state.answers[idx]
            st.session_state.active_question_index = idx

        st.write(f"Question {idx + 1} of {len(questions)}")
        source_label = sources[idx] if idx < len(sources) else "RAG"

        st.markdown(
            f"""
            <div class="question-card">
                <div class="question-source">Source: [{source_label}]</div>
                <div class="question-source">Interviewer tone: [{st.session_state.interview_tone}]</div>
                <div><strong>{questions[idx]}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Step 4: Your Answer",
            key="current_answer_input",
            height=160,
            placeholder="Explain your approach clearly with technical depth and an example...",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Submit Answer", use_container_width=True):
                st.session_state.answers[idx] = st.session_state.current_answer_input
                question_key = f"{interview_role}::{tone_value}::{questions[idx]}"
                with st.spinner(
                    f"AI interviewer reviewing your answer and preparing reference via {evaluator_backend}..."
                ):
                    evaluation = evaluate_answer(
                        questions[idx],
                        st.session_state.current_answer_input,
                        interview_role,
                        tone=tone_value,
                    )
                    st.session_state.reference_cache[question_key] = generate_reference_answer(
                        questions[idx],
                        interview_role,
                        tone=tone_value,
                    )
                st.session_state.evaluations[idx] = evaluation
                st.rerun()

        with c2:
            if st.button("Next Question", use_container_width=True):
                st.session_state.answers[idx] = st.session_state.current_answer_input
                if idx < len(questions) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
                else:
                    st.success("Interview completed. Scroll down for summary.")

        if st.session_state.evaluations[idx] is not None:
            ev = st.session_state.evaluations[idx]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Score", f"{ev['score']}/10")
            with m2:
                st.caption("Strengths")
                _render_typing(st.empty(), ev["strengths"], f"q{idx}_strengths_{hash(ev['strengths'])}", words_per_minute=40)
            with m3:
                st.caption("Weaknesses / Missing Points")
                _render_typing(
                    st.empty(),
                    ev["improvements"],
                    f"q{idx}_improvements_{hash(ev['improvements'])}",
                    words_per_minute=40,
                )

            question_key = f"{interview_role}::{tone_value}::{questions[idx]}"
            with st.expander("AI Reference Answer (Learn from this)", expanded=False):
                ref = st.session_state.reference_cache.get(question_key, "")
                if ref:
                    _render_typing(st.empty(), ref, f"q{idx}_model_{hash(ref)}", words_per_minute=40)
                else:
                    st.caption("Reference answer will appear after you submit this question.")

        completed = [x for x in st.session_state.evaluations if x is not None]
        if completed:
            avg = sum(x["score"] for x in completed) / len(completed)
            st.markdown(
                f"""
                <div class="summary-card">
                    <h4 style="margin:0 0 0.5rem 0;">Final Interview Summary</h4>
                    <p style="margin:0.2rem 0;">Role: <strong>{interview_role}</strong></p>
                    <p style="margin:0.2rem 0;">Questions Answered: <strong>{len(completed)}/{len(questions)}</strong></p>
                    <p style="margin:0.2rem 0;">Current Average Score: <strong>{avg:.2f}/10</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Retake Interview", use_container_width=True):
                st.session_state.pending_retake = True
                st.session_state.pending_retake_role = interview_role
                st.session_state.pending_retake_count = question_count
                st.rerun()
