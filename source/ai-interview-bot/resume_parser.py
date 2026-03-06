import re
from functools import lru_cache
from typing import List

import torch
from pypdf import PdfReader
from transformers import pipeline

KNOWN_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust",
    "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "Django", "Flask", "FastAPI", "Spring Boot", "Node.js", "React",
    "Angular", "Vue.js", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "Machine Learning", "Deep Learning", "NLP", "PyTorch", "TensorFlow",
    "Scikit-learn", "Pandas", "NumPy", "Data Structures", "Algorithms",
    "System Design", "CI/CD", "Git", "Linux", "REST API", "GraphQL",
    "Kafka", "Spark", "Hadoop"
]


def _pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _get_skill_extractor():
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=_pipeline_device(),
    )


def warmup_skill_extractor() -> None:
    _get_skill_extractor()


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(resume_text: str, threshold: float = 0.45, top_k: int = 12) -> List[str]:
    cleaned_text = _compact_whitespace(resume_text)
    if not cleaned_text:
        return []

    sample_text = cleaned_text[:2500]
    classifier = _get_skill_extractor()
    result = classifier(sample_text, candidate_labels=KNOWN_SKILLS, multi_label=True)

    pairs = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)
    selected = [label for label, score in pairs if score >= threshold][:top_k]

    if not selected:
        lowered = cleaned_text.lower()
        selected = [skill for skill in KNOWN_SKILLS if skill.lower() in lowered][:top_k]

    return selected