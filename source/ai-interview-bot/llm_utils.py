import json
import re
from functools import lru_cache
from typing import Any, Dict, List

from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from device_utils import get_device, get_pipeline_device


def _evaluator_task() -> str:
    supported = set(PIPELINE_REGISTRY.get_supported_tasks())
    for task in ("text2text-generation", "text-generation"):
        if task in supported:
            return task
    raise RuntimeError("No supported text generation task found in transformers pipeline registry.")


def _build_evaluator(device_arg):
    return pipeline(
        _evaluator_task(),
        model="google/flan-t5-base",
        device=device_arg,
    )


@lru_cache(maxsize=1)
def _get_evaluator():
    if get_device() == "mps":
        try:
            return _build_evaluator(get_pipeline_device())
        except Exception:
            pass
    return _build_evaluator(-1)


def warmup_evaluator_model() -> None:
    evaluator = _get_evaluator()
    evaluator("Warmup", max_new_tokens=1, do_sample=False)


def _extract_generated_text(output_item: Dict[str, Any]) -> str:
    generated = output_item.get("generated_text") or output_item.get("summary_text") or output_item.get("text")
    if isinstance(generated, list):
        if generated and isinstance(generated[-1], dict):
            return str(generated[-1].get("content") or generated[-1].get("text") or generated[-1])
        return " ".join(str(item) for item in generated)
    if generated is not None:
        return str(generated)
    return str(output_item)


def _run_generation(prompt: str, max_new_tokens: int = 60) -> str:
    output_item = _get_evaluator()(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
    output = _extract_generated_text(output_item)
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    return output.strip()


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


def evaluate_answer(question: str, answer: str, role: str) -> Dict[str, str]:
    if not answer.strip():
        return {
            "score": 1,
            "strengths": "No strengths detected.",
            "improvements": "Provide a relevant technical answer with clear structure and one concrete example.",
        }

    raw = _run_generation(_build_eval_prompt(question, answer, role), max_new_tokens=60)

    try:
        parsed_obj = _extract_json_object(raw)
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
    }

    if any(w in answer.lower() for w in ["fuck", "shit", "bitch", "asshole"]):
        result["score"] = min(result["score"], 2)
        result["strengths"] = "No strengths detected."
        result["improvements"] = "Keep the answer professional and focused on the technical question."

    return result
