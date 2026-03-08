import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List

from transformers import pipeline

from device_utils import get_device, get_pipeline_device


DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_FALLBACKS = ["Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]


def _candidate_models() -> List[str]:
    candidates = [DEFAULT_EVALUATOR_MODEL]
    for name in MODEL_FALLBACKS:
        if name not in candidates:
            candidates.append(name)
    return candidates


def _build_evaluator(device_arg, model_name: str):
    return pipeline("text-generation", model=model_name, device=device_arg)


@lru_cache(maxsize=1)
def _get_evaluator():
    models = _candidate_models()
    if get_device() == "mps":
        for model_name in models:
            try:
                return _build_evaluator(get_pipeline_device(), model_name)
            except Exception:
                continue
    for model_name in models:
        try:
            return _build_evaluator(-1, model_name)
        except Exception:
            continue
    raise RuntimeError("Failed to load evaluator model.")


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


def _run_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output_item = _get_evaluator()(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )[0]
    return _extract_generated_text(output_item, user_prompt)


def warmup_evaluator_model() -> None:
    _run_chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with one word: warm.",
        max_new_tokens=6,
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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]*", (text or "").lower())


def _is_low_signal_answer(answer: str) -> bool:
    tokens = _tokenize(answer)
    low_signal = {"idk", "bro", "whatever", "nothing", "no", "don't", "dont", "know", "god"}
    if len(tokens) <= 5 and any(tok in low_signal for tok in tokens):
        return True
    return False


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
    ranked: List[tuple[int, str]] = []
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


def _final_score(question: str, answer: str, strength: str) -> int:
    heuristic = _heuristic_score(question, answer)
    score = heuristic
    if strength.lower().startswith("no clear technical strength"):
        score = min(score, 4)
    if _is_profane(answer):
        score = 1
    return max(1, min(10, score))


def evaluate_answer(question: str, answer: str, role: str) -> Dict[str, str]:
    safe_answer = (answer or "").strip() or "No candidate answer provided."
    _ = role  # role retained for API compatibility and future role-specific logic.
    strength = _derive_strength(question, safe_answer)
    weakness = _derive_weakness(question, safe_answer)
    final_score = _final_score(question, safe_answer, strength)

    return {
        "score": final_score,
        "strengths": _normalize_line(strength),
        "improvements": _normalize_line(weakness),
    }


@lru_cache(maxsize=512)
def generate_reference_answer(question: str, role: str) -> str:
    answer = _run_chat(
        system_prompt="You are a candidate in a technical interview. Answer directly and clearly.",
        user_prompt=(
            f"Role: {role}\n"
            f"Question: {question}\n\n"
            "Provide the best candidate answer in 4 to 6 complete sentences.\n"
            "Write in first-person interview style, technically detailed, coherent, and concise.\n"
            "Do not provide coaching tips, headings, or bullet points."
        ),
        max_new_tokens=260,
    )

    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    cleaned = re.sub(r"^\s*model answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\d+\.\s*", " ", cleaned)
    cleaned = _normalize_line(cleaned)

    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    if len(_sentences(cleaned)) < 4 or not cleaned.endswith((".", "!", "?")):
        continuation = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                "Continue and finish this answer so it becomes 4 to 6 coherent complete sentences.\n"
                f"Current answer:\n{cleaned}"
            ),
            max_new_tokens=180,
        )
        cleaned = _normalize_line(f"{cleaned} {continuation}")

    sentences = _sentences(cleaned)
    if len(sentences) < 4:
        rewritten = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                f"Write a fresh answer to this question in 4 to 6 complete coherent sentences.\nQuestion: {question}"
            ),
            max_new_tokens=220,
        )
        cleaned = _normalize_line(rewritten)
        sentences = _sentences(cleaned)

    if len(sentences) > 6:
        sentences = sentences[:6]
    cleaned = " ".join(sentences)
    cleaned = _normalize_line(cleaned)

    return f"Model Answer: {cleaned}"
