import json
import re
from functools import lru_cache
from typing import Dict, List

import torch
from transformers import pipeline


def _pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _get_evaluator():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device=_pipeline_device(),
    )


def warmup_evaluator_model() -> None:
    evaluator = _get_evaluator()
    evaluator("Warmup", max_new_tokens=1, do_sample=False)


def _extract_generated_text(output_item: Dict[str, str]) -> str:
    return output_item.get("generated_text") or output_item.get("summary_text") or str(output_item)


def _run_generation(prompt: str, max_new_tokens: int = 260) -> str:
    output_item = _get_evaluator()(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
    output = _extract_generated_text(output_item)
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    return output.strip()


def _extract_complete_sentences(text: str, max_sentences: int = 6) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.replace("<", "").replace(">", "")
    cleaned = re.sub(r"\b(score|strengths|improvements|model_answer)\s*:", "", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    complete = [p.strip() for p in parts if p.strip() and p.strip()[-1] in ".!?"]
    return " ".join(complete[:max_sentences]) if complete else ""


def _clean_line(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).replace("<", "").replace(">", "")
    if not cleaned:
        return fallback
    bad = {
        "one sentence", "short sentence", "short text", "none", "n/a", "not provided",
        "based strictly on the actual answer",
    }
    if cleaned.lower() in bad:
        return fallback
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _extract_json_object(text: str) -> Dict[str, str]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("JSON object not found")
    return json.loads(match.group(0))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]*", text.lower())


def _contextual_fallback(question: str, answer: str) -> Dict[str, str]:
    q_tokens = set(_tokenize(question))
    a_tokens = _tokenize(answer)
    a_set = set(a_tokens)

    if not a_tokens:
        return {
            "score": 1,
            "strengths": "No strengths detected.",
            "improvements": "Provide a relevant technical answer with clear structure and one concrete example.",
        }

    profane = any(w in a_set for w in {"fuck", "shit", "bitch", "asshole"})
    overlap = len(q_tokens & a_set)
    length = len(a_tokens)

    if profane:
        score = 1
        strengths = "No strengths detected."
        improvements = "Keep the answer professional and focus on technical content relevant to the question."
    else:
        score = 3
        if length >= 35:
            score += 2
        if length >= 80:
            score += 1
        if overlap >= 4:
            score += 2
        elif overlap >= 2:
            score += 1
        score = max(1, min(10, score))

        strengths = "No strengths detected."
        if overlap >= 2 and length >= 20:
            strengths = "You referenced relevant concepts from the question and provided useful structure."

        improvements = "Add deeper technical reasoning, explicit trade-offs, and one concrete implementation example."

    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
    }


def _build_eval_prompt(question: str, answer: str, role: str) -> str:
    return (
        "You are a strict senior technical interviewer.\n"
        "Evaluate ONLY what the candidate actually wrote. Do not hallucinate achievements.\n"
        "If answer is abusive/irrelevant, give low score and strengths='No strengths detected.'.\n\n"
        f"Role: {role}\n"
        f"Question: {question}\n"
        f"Candidate Answer: {answer}\n\n"
        "Return ONLY valid JSON with keys: score, strengths, improvements.\n"
        "Example:\n"
        '{"score":7,"strengths":"...","improvements":"..."}'
    )


def _repair_eval_json(raw: str, question: str, answer: str, role: str) -> str:
    prompt = (
        "Convert the following content to VALID JSON only with keys score, strengths, improvements.\n"
        "No markdown. No extra keys.\n\n"
        f"Role: {role}\nQuestion: {question}\nCandidate Answer: {answer}\n\n"
        f"Content:\n{raw}\n"
    )
    return _run_generation(prompt, max_new_tokens=180)


def _generate_reference_answer(question: str, role: str) -> str:
    prompt = (
        "You are an expert interview coach.\n"
        f"Role: {role}\n"
        f"Question: {question}\n\n"
        "Write a high-quality reference answer in 4 to 6 COMPLETE sentences. "
        "Include definition, practical approach, trade-offs, and one concrete example. "
        "Return only answer text."
    )
    raw = _run_generation(prompt, max_new_tokens=280)
    parsed = _extract_complete_sentences(raw, max_sentences=6)
    if parsed:
        return parsed
    return (
        "A strong answer should define the concept clearly, then explain an implementation approach with key decisions. "
        "It should discuss trade-offs such as performance, complexity, and maintainability. "
        "It should also mention testing and monitoring in production. "
        "Finally, it should include a practical example with measurable impact."
    )


def evaluate_answer(question: str, answer: str, role: str) -> Dict[str, str]:
    if not answer.strip():
        return {
            "score": 1,
            "strengths": "No strengths detected.",
            "improvements": "Provide a relevant technical answer with clear structure and one concrete example.",
            "model_answer": _generate_reference_answer(question, role),
        }

    raw = _run_generation(_build_eval_prompt(question, answer, role), max_new_tokens=220)

    try:
        parsed_obj = _extract_json_object(raw)
    except Exception:
        repaired = _repair_eval_json(raw, question, answer, role)
        try:
            parsed_obj = _extract_json_object(repaired)
        except Exception:
            parsed_obj = _contextual_fallback(question, answer)

    score = int(parsed_obj.get("score", 1)) if str(parsed_obj.get("score", "")).isdigit() else 1
    score = max(1, min(10, score))

    strengths = _clean_line(str(parsed_obj.get("strengths", "")), "No strengths detected.")
    improvements = _clean_line(
        str(parsed_obj.get("improvements", "")),
        "Provide a technically relevant and professional response to the question.",
    )

    result = {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "model_answer": _generate_reference_answer(question, role),
    }

    if any(w in answer.lower() for w in ["fuck", "shit", "bitch", "asshole"]):
        result["score"] = min(result["score"], 2)
        result["strengths"] = "No strengths detected."
        result["improvements"] = "Keep the answer professional and focused on the technical question."

    return result