import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline

from device_utils import get_device, get_pipeline_device


DEFAULT_EVALUATOR_PROVIDER = os.getenv("EVALUATOR_PROVIDER", "local").strip().lower()
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_FALLBACKS = ["Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
DEFAULT_PUTER_MODEL = os.getenv("PUTER_MODEL", "google/gemini-2.5-flash-lite")
PUTER_AUTH_TOKEN_ENV = "PUTER_AUTH_TOKEN"
PUTER_TIMEOUT_SECONDS = int(os.getenv("PUTER_TIMEOUT_SECONDS", "120"))
PUTER_BRIDGE_PATH = Path(__file__).with_name("puter_bridge.js")
KNOWN_ROLES = ["Backend", "AI Engineer", "Data Scientist", "Fullstack"]
TONE_LABELS = {"easy": "Easy", "medium": "Medium", "strict": "Strict"}
TONE_SCORE_BIAS = {"easy": 1, "medium": 0, "strict": -1}


def get_evaluator_provider() -> str:
    provider = DEFAULT_EVALUATOR_PROVIDER
    if provider in {"puter", "puter-gemini", "gemini"}:
        return "puter"
    return "local"


def get_evaluator_backend_label() -> str:
    if get_evaluator_provider() == "puter":
        return f"Puter Gemini ({DEFAULT_PUTER_MODEL})"
    return f"Local HF ({DEFAULT_EVALUATOR_MODEL})"


def _candidate_models() -> List[str]:
    candidates = [DEFAULT_EVALUATOR_MODEL]
    for name in MODEL_FALLBACKS:
        if name not in candidates:
            candidates.append(name)
    return candidates


def _build_evaluator(device_arg, model_name: str):
    return pipeline("text-generation", model=model_name, device=device_arg)


@lru_cache(maxsize=1)
def _get_local_evaluator_with_name() -> Tuple[Any, str]:
    models = _candidate_models()
    if get_device() == "mps":
        for model_name in models:
            try:
                return _build_evaluator(get_pipeline_device(), model_name), model_name
            except Exception:
                continue
    for model_name in models:
        try:
            return _build_evaluator(-1, model_name), model_name
        except Exception:
            continue
    raise RuntimeError("Failed to load local evaluator model.")


def _get_local_evaluator():
    evaluator, _ = _get_local_evaluator_with_name()
    return evaluator


def _extract_generated_text(output_item: Dict[str, Any], prompt: str) -> str:
    generated = output_item.get("generated_text")
    if isinstance(generated, list):
        if generated and isinstance(generated[-1], dict):
            return str(generated[-1].get("content") or generated[-1].get("text") or generated[-1]).strip()
        return " ".join(str(item) for item in generated).strip()
    if isinstance(generated, str):
        text = generated.strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    return str(output_item).strip()


def _run_local_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output_item = _get_local_evaluator()(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )[0]
    return _extract_generated_text(output_item, user_prompt)


def _extract_puter_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str):
            return raw["text"].strip()
        message = raw.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        chunks.append(str(item["text"]))
                    elif isinstance(item, str):
                        chunks.append(item)
                if chunks:
                    return " ".join(chunks).strip()
    return str(raw or "").strip()


def _run_puter_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    if not PUTER_BRIDGE_PATH.exists():
        raise RuntimeError(f"Puter bridge script missing: {PUTER_BRIDGE_PATH}")

    auth_token = os.getenv(PUTER_AUTH_TOKEN_ENV, "").strip()
    if not auth_token:
        raise RuntimeError(
            "Puter provider requires PUTER_AUTH_TOKEN. "
            "Run `npm run puter:login` in source/ai-interview-bot and export the returned token."
        )

    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": DEFAULT_PUTER_MODEL,
        "max_new_tokens": max_new_tokens,
    }

    proc = subprocess.run(
        ["node", str(PUTER_BRIDGE_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=PUTER_TIMEOUT_SECONDS,
        check=False,
        env={**os.environ, PUTER_AUTH_TOKEN_ENV: auth_token},
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "Unknown Puter error."
        raise RuntimeError(f"Puter chat failed: {err}")

    raw_output = proc.stdout.strip()
    if not raw_output:
        raise RuntimeError("Puter chat failed: empty stdout.")

    parsed: Optional[Dict[str, Any]] = None
    for line in reversed([ln.strip() for ln in raw_output.splitlines() if ln.strip()]):
        try:
            candidate = json.loads(line)
            if isinstance(candidate, dict):
                parsed = candidate
                break
        except Exception:
            continue
    if parsed is None:
        raise RuntimeError(f"Puter chat failed: invalid JSON output: {raw_output[:200]}")

    text = _extract_puter_text(parsed.get("response") if "response" in parsed else parsed)
    if not text:
        text = _extract_puter_text(parsed)
    if not text:
        raise RuntimeError("Puter chat failed: empty response text.")
    return text


def _run_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    if get_evaluator_provider() == "puter":
        return _run_puter_chat(system_prompt, user_prompt, max_new_tokens)
    return _run_local_chat(system_prompt, user_prompt, max_new_tokens)


def warmup_evaluator_model() -> None:
    _run_chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with one word: warm.",
        max_new_tokens=8,
    )


def _normalize_line(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"^(strengths?|weaknesses?|improvements?|score)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _coerce_score(value: Any) -> int:
    if isinstance(value, int):
        return max(1, min(10, value))
    match = re.search(r"\b(10|[1-9])\b", str(value or ""))
    if not match:
        return 1
    return int(match.group(1))


def _normalize_tone(tone: str) -> str:
    t = (tone or "").strip().lower()
    if t in TONE_LABELS:
        return t
    return "medium"


def _normalize_role_name(role_value: Any, fallback: str = "Backend") -> str:
    raw = str(role_value or "").strip().lower()
    if not raw:
        return fallback
    alias_map = {
        "backend": "Backend",
        "backend engineer": "Backend",
        "software engineer": "Backend",
        "ai engineer": "AI Engineer",
        "ml engineer": "AI Engineer",
        "machine learning engineer": "AI Engineer",
        "data scientist": "Data Scientist",
        "data science": "Data Scientist",
        "fullstack": "Fullstack",
        "full stack": "Fullstack",
        "full-stack": "Fullstack",
    }
    if raw in alias_map:
        return alias_map[raw]
    for known in KNOWN_ROLES:
        if known.lower() in raw:
            return known
    return fallback


def _coerce_skill_list(value: Any, top_k: int = 12) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        items = [str(x).strip() for x in value]
    elif isinstance(value, str):
        items = [x.strip() for x in re.split(r"[,;\n]+", value)]

    cleaned: List[str] = []
    seen = set()
    for item in items:
        skill = re.sub(r"\s+", " ", item).strip().lower()
        if not skill or len(skill) < 2:
            continue
        if skill in seen:
            continue
        seen.add(skill)
        cleaned.append(skill)
        if len(cleaned) >= top_k:
            break
    return cleaned


def _heuristic_role_from_text(job_desc: str, resume_text: str, fallback: str = "Backend") -> str:
    text = f"{job_desc}\n{resume_text}".lower()
    if any(k in text for k in ["pytorch", "tensorflow", "llm", "nlp", "mlops"]):
        return "AI Engineer"
    if any(k in text for k in ["hypothesis", "statistics", "ab test", "eda", "regression"]):
        return "Data Scientist"
    if any(k in text for k in ["react", "frontend", "full stack", "typescript", "next.js"]):
        return "Fullstack"
    return fallback


def _heuristic_skills(text: str, top_k: int = 12) -> List[str]:
    candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+.#/\-]{1,24}\b", text or "")
    preferred = {
        "python", "java", "go", "rust", "c++", "pytorch", "tensorflow", "sql", "postgres",
        "mongodb", "redis", "docker", "kubernetes", "aws", "gcp", "azure", "spark", "hadoop",
        "react", "node", "node.js", "typescript", "javascript", "fastapi", "django", "flask",
        "llm", "mlops", "airflow", "kafka", "faiss", "numpy", "pandas", "scikit", "xgboost",
    }
    out: List[str] = []
    seen = set()
    for token in candidates:
        t = token.lower()
        if t not in preferred or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= top_k:
            break
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]*", (text or "").lower())


def _is_low_signal_answer(answer: str) -> bool:
    tokens = _tokenize(answer)
    low_signal = {"idk", "bro", "whatever", "nothing", "no", "don't", "dont", "know", "god"}
    return len(tokens) <= 5 and any(tok in low_signal for tok in tokens)


def _is_profane(answer: str) -> bool:
    tokens = set(_tokenize(answer))
    banned = {
        "fuck", "fucking", "fucked", "shit", "bitch", "asshole", "dick", "cock", "pussy",
        "motherfucker", "bastard", "slut", "whore",
    }
    return bool(tokens.intersection(banned))


def _is_relevant_to_question(question: str, answer: str) -> bool:
    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer))
    if not a_tokens:
        return False
    overlap = len(q_tokens.intersection(a_tokens))
    return overlap >= 1


def _technical_richness(answer: str) -> int:
    tokens = set(_tokenize(answer))
    markers = {
        "missing", "outliers", "correlation", "correlations", "distribution", "distributions",
        "leakage", "regularization", "feature", "features", "scalable", "queue", "scheduler",
        "latency", "throughput", "monitoring", "logging", "retry", "retries", "idempotent",
        "docker", "kubernetes", "pytorch", "tensorflow", "sql", "redis", "cache",
        "histogram", "histograms", "boxplot", "boxplots", "describe", "dataset", "eda",
    }
    return len(tokens.intersection(markers))


def _heuristic_score(question: str, answer: str) -> int:
    q_tokens = set(_tokenize(question))
    a_tokens = _tokenize(answer)
    if not a_tokens:
        return 1

    overlap = len(q_tokens.intersection(set(a_tokens)))
    length = len(a_tokens)
    score = 1
    if length >= 4:
        score = 2
    if length >= 10:
        score = 3
    if overlap >= 1:
        score = max(score, 4)
    if overlap >= 3:
        score = max(score, 6)
    if overlap >= 6:
        score = max(score, 7)
    if length >= 25 and overlap >= 3:
        score += 1
    if length >= 45 and overlap >= 5:
        score += 1

    richness = _technical_richness(answer)
    if richness >= 3:
        score = max(score, 6)
    if richness >= 5 and length >= 20:
        score = max(score, 7)
    if richness >= 6 and overlap >= 1 and length >= 28:
        score = max(score, 8)

    if _is_low_signal_answer(answer):
        return 1
    if _is_profane(answer):
        return 1
    if not _is_relevant_to_question(question, answer):
        return min(score, 2)

    return max(1, min(10, score))


def _top_grounded_phrases(question: str, answer: str, limit: int = 2) -> List[str]:
    q_tokens = set(_tokenize(question))
    phrases = [p.strip() for p in re.split(r"[\n,;]+", answer or "") if p.strip()]
    ranked: List[Tuple[int, str]] = []
    for phrase in phrases:
        tokens = set(_tokenize(phrase))
        overlap = len(tokens.intersection(q_tokens))
        if overlap > 0:
            ranked.append((overlap, phrase))
    ranked.sort(key=lambda x: (-x[0], -len(x[1])))
    return [p for _, p in ranked[:limit]]


def _derive_strength(question: str, answer: str) -> str:
    if _is_profane(answer) or _is_low_signal_answer(answer) or not _is_relevant_to_question(question, answer):
        return "No clear technical strength shown."

    grounded = _top_grounded_phrases(question, answer, limit=2)
    if not grounded:
        return "No clear technical strength shown."
    if len(grounded) == 1:
        return f'You correctly mentioned "{grounded[0]}", which is relevant to the question.'
    return f'You covered relevant points such as "{grounded[0]}" and "{grounded[1]}".'


def _derive_weakness(question: str, answer: str) -> str:
    a_tokens = _tokenize(answer)
    q_tokens = set(_tokenize(question))
    overlap = len(q_tokens.intersection(set(a_tokens)))
    answer_l = (answer or "").lower()
    richness = _technical_richness(answer)

    missing: List[str] = []
    if len(a_tokens) < 12:
        missing.append("add more technical depth")
    if overlap < 2 and richness < 3:
        missing.append("align the response more directly with the asked question")
    if not any(k in answer_l for k in ["because", "therefore", "so that", "trade-off", "tradeoff", "pros", "cons"]):
        missing.append("explain why your approach works and include a trade-off")
    if not any(k in answer_l for k in ["example", "for instance", "e.g", "such as"]):
        missing.append("include one concrete example")

    if _is_profane(answer):
        return "The answer is unprofessional and not technically relevant; provide a respectful, technical response."
    if not missing:
        return "Add implementation details on scale, reliability, and monitoring to make the answer stronger."
    return "You should " + ", and ".join(missing[:2]) + "."


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    candidates = [raw]
    for match in re.finditer(r"\{[\s\S]*?\}", raw):
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def analyze_candidate_profile(resume_text: str, job_description: str, selected_role: str = "") -> Dict[str, Any]:
    safe_resume = (resume_text or "").strip()
    safe_jd = (job_description or "").strip()
    manual_role = _normalize_role_name(selected_role, fallback="Backend") if selected_role else ""
    fallback_role = manual_role or _heuristic_role_from_text(safe_jd, safe_resume, fallback="Backend")

    prompt = (
        "Analyze the candidate profile for technical interview preparation.\n"
        "Use resume and job description context to infer the best interview role and strongest technical skills.\n"
        "Return strict JSON only with keys: role, skills, jd_focus.\n"
        "- role must be one of: Backend, AI Engineer, Data Scientist, Fullstack\n"
        "- skills must be a list of 6 to 12 lowercase technical skills relevant to the role\n"
        "- jd_focus must be one short sentence summarizing what the job description emphasizes\n\n"
        f"Manual role override (empty means infer from JD): {selected_role}\n"
        f"Job Description:\n{safe_jd or '[none provided]'}\n\n"
        f"Resume:\n{safe_resume[:6000]}"
    )

    parsed: Optional[Dict[str, Any]] = None
    try:
        raw = _run_chat(
            system_prompt="You are an expert technical recruiter and hiring manager.",
            user_prompt=prompt,
            max_new_tokens=300,
        )
        parsed = _extract_json_block(raw)
    except Exception:
        parsed = None

    if parsed:
        role = _normalize_role_name(parsed.get("role"), fallback=fallback_role)
        if selected_role:
            role = manual_role
        skills = _coerce_skill_list(parsed.get("skills"), top_k=12)
        if not skills:
            skills = _heuristic_skills(f"{safe_resume}\n{safe_jd}", top_k=12)
        jd_focus = _normalize_line(parsed.get("jd_focus") or "Interview focus generated from resume and role context.")
        return {"role": role, "skills": skills, "jd_focus": jd_focus}

    return {
        "role": fallback_role,
        "skills": _heuristic_skills(f"{safe_resume}\n{safe_jd}", top_k=12),
        "jd_focus": "Interview focus inferred from available profile context.",
    }


def _looks_grounded_feedback(feedback: str, question: str, answer: str) -> bool:
    text = (feedback or "").strip().lower()
    if not text:
        return False
    if "no clear technical strength" in text:
        return True
    f_tokens = set(_tokenize(feedback))
    a_tokens = set(_tokenize(answer))
    q_tokens = set(_tokenize(question))
    return len(f_tokens.intersection(a_tokens.union(q_tokens))) >= 2


def _evaluate_with_model(question: str, answer: str, role: str, tone: str) -> Optional[Dict[str, str]]:
    tone_value = _normalize_tone(tone)
    tone_instruction = {
        "easy": "Be supportive and coaching-oriented while staying truthful to the answer.",
        "medium": "Be balanced and interview-realistic.",
        "strict": "Be demanding and rigorous; expect depth, trade-offs, and concrete technical detail.",
    }[tone_value]

    prompt = (
        "Evaluate the candidate answer for this interview question.\n"
        "Use only evidence present in the candidate answer. Do not invent claims.\n"
        "Return JSON only with keys: score, strengths, improvements.\n"
        "Rules:\n"
        "- score: integer 1 to 10\n"
        "- strengths: one concise sentence about what the candidate actually did well\n"
        "- improvements: one concise sentence on what is missing or can be improved\n"
        "- If answer is abusive/off-topic/very low-quality, set score <= 2 and reflect that clearly.\n\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]} ({tone_instruction})\n"
        f"Role: {role}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
    )
    raw = _run_chat(
        system_prompt="You are a strict technical interviewer and accurate grader.",
        user_prompt=prompt,
        max_new_tokens=220,
    )
    parsed = _extract_json_block(raw)
    if not parsed:
        return None
    return {
        "score": _coerce_score(parsed.get("score")),
        "strengths": _normalize_line(parsed.get("strengths")),
        "improvements": _normalize_line(parsed.get("improvements")),
    }


def _final_score(
    question: str,
    answer: str,
    strength: str,
    tone: str,
    model_score: Optional[int] = None,
) -> int:
    tone_value = _normalize_tone(tone)
    heuristic = _heuristic_score(question, answer)
    score = heuristic if model_score is None else model_score
    if model_score is not None:
        score = max(heuristic - 2, min(heuristic + 2, model_score))
    score += TONE_SCORE_BIAS[tone_value]
    if strength.lower().startswith("no clear technical strength"):
        score = min(score, 4)
    if _is_profane(answer):
        score = 1
    if _is_low_signal_answer(answer):
        score = min(score, 2)
    return max(1, min(10, score))


def evaluate_answer(question: str, answer: str, role: str, tone: str = "medium") -> Dict[str, str]:
    safe_answer = (answer or "").strip() or "No candidate answer provided."
    tone_value = _normalize_tone(tone)

    heuristic_strength = _normalize_line(_derive_strength(question, safe_answer))
    heuristic_weakness = _normalize_line(_derive_weakness(question, safe_answer))

    model_eval: Optional[Dict[str, str]] = None
    try:
        model_eval = _evaluate_with_model(question, safe_answer, role, tone=tone_value)
    except Exception:
        model_eval = None

    strengths = heuristic_strength
    improvements = heuristic_weakness
    model_score: Optional[int] = None

    if model_eval:
        maybe_strength = _normalize_line(model_eval.get("strengths", ""))
        maybe_improvements = _normalize_line(model_eval.get("improvements", ""))

        if maybe_strength and _looks_grounded_feedback(maybe_strength, question, safe_answer):
            strengths = maybe_strength
        if maybe_improvements and _looks_grounded_feedback(maybe_improvements, question, safe_answer):
            improvements = maybe_improvements

        model_score = _coerce_score(model_eval.get("score"))

    final_score = _final_score(question, safe_answer, strengths, tone=tone_value, model_score=model_score)
    return {"score": final_score, "strengths": strengths, "improvements": improvements}


@lru_cache(maxsize=512)
def generate_reference_answer(question: str, role: str, tone: str = "medium") -> str:
    tone_value = _normalize_tone(tone)
    tone_instruction = {
        "easy": "Keep wording approachable but still technical.",
        "medium": "Keep wording balanced and interview-ready.",
        "strict": "Make it highly rigorous with implementation specifics and trade-offs.",
    }[tone_value]

    answer = _run_chat(
        system_prompt="You are a candidate in a technical interview. Answer directly and clearly.",
        user_prompt=(
            f"Role: {role}\n"
            f"Interviewer tone: {TONE_LABELS[tone_value]} ({tone_instruction})\n"
            f"Question: {question}\n\n"
            "Provide the best candidate answer in 5 to 7 complete sentences.\n"
            "Write in first-person interview style, technically detailed, coherent, and concise.\n"
            "Do not provide coaching tips, headings, or bullet points."
        ),
        max_new_tokens=340,
    )

    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    cleaned = re.sub(r"^\s*model answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\d+\.\s*", " ", cleaned)
    cleaned = _normalize_line(cleaned)

    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    if len(_sentences(cleaned)) < 5 or not cleaned.endswith((".", "!", "?")):
        continuation = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                "Continue and finish this answer so it becomes 5 to 7 coherent complete sentences.\n"
                f"Current answer:\n{cleaned}"
            ),
            max_new_tokens=220,
        )
        cleaned = _normalize_line(f"{cleaned} {continuation}")

    sentences = _sentences(cleaned)
    if len(sentences) < 5:
        rewritten = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                "Write a fresh direct answer as if you are the candidate speaking in an interview.\n"
                "Return 5 to 7 complete coherent sentences only.\n"
                f"Question: {question}"
            ),
            max_new_tokens=280,
        )
        cleaned = _normalize_line(rewritten)
        sentences = _sentences(cleaned)

    if len(sentences) > 7:
        sentences = sentences[:7]
    cleaned = _normalize_line(" ".join(sentences))
    return f"Model Answer: {cleaned}"
