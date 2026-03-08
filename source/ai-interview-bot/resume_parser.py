import re
from typing import List

from pypdf import PdfReader

KNOWN_SKILLS = [
    "python",
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "sql",
    "docker",
    "kubernetes",
    "aws",
    "react",
    "node",
]

SKILL_ALIASES = {
    "node": ["node", "node.js", "nodejs"],
}


def warmup_skill_extractor() -> None:
    """Keyword extraction does not require model warmup."""
    return None


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_skill(text: str, skill: str) -> bool:
    aliases = SKILL_ALIASES.get(skill, [skill])
    for alias in aliases:
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def _keyword_skill_extract(cleaned_text: str, top_k: int) -> List[str]:
    return [skill for skill in KNOWN_SKILLS if _contains_skill(cleaned_text, skill)][:top_k]


def extract_skills(resume_text: str, top_k: int = 12) -> List[str]:
    cleaned_text = _compact_whitespace(resume_text)
    if not cleaned_text:
        return []
    return _keyword_skill_extract(cleaned_text, top_k=top_k)
