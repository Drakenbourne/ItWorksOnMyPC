from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import faiss
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class InterviewQuestion:
    category: str
    text: str


def _sbert_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_question_line(row: str) -> InterviewQuestion:
    if "\t" in row:
        category, question = row.split("\t", 1)
        return InterviewQuestion(category=category.strip(), text=question.strip())

    if ":" in row:
        category, question = row.split(":", 1)
        if category.strip() and question.strip():
            return InterviewQuestion(category=category.strip(), text=question.strip())

    return InterviewQuestion(category="GENERAL", text=row.strip())


def _load_questions(file_path: str) -> List[InterviewQuestion]:
    questions: List[InterviewQuestion] = []
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        parsed = _parse_question_line(row)
        if parsed.text:
            questions.append(parsed)

    if not questions:
        raise ValueError("No interview questions found in interview_questions.txt")
    return questions


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", device=_sbert_device())


class RAGEngine:
    def __init__(self, questions: List[InterviewQuestion]):
        self.questions = questions
        self._model = _get_embedding_model()
        self._index = None
        self._build_index()

    def _build_index(self) -> None:
        corpus = [f"{q.category}. {q.text}" for q in self.questions]
        embeddings = self._model.encode(corpus, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

    def retrieve_questions(
        self,
        query: str,
        top_k: int = 3,
        preferred_categories: Optional[List[str]] = None,
    ) -> List[str]:
        if self._index is None:
            raise RuntimeError("FAISS index was not built")

        cleaned_query = query.strip() or "software engineering interview"
        query_vec = self._model.encode([cleaned_query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        search_k = min(max(top_k * 6, top_k), len(self.questions))
        _, indices = self._index.search(query_vec, search_k)

        preferred = {c.lower() for c in (preferred_categories or [])}
        primary: List[str] = []
        secondary: List[str] = []
        seen = set()

        for idx in indices[0]:
            q = self.questions[idx]
            key = q.text.lower()
            if key in seen:
                continue
            seen.add(key)

            if preferred and q.category.lower() in preferred:
                primary.append(q.text)
            else:
                secondary.append(q.text)

        ranked = primary + secondary
        return ranked[: min(top_k, len(ranked))]


@lru_cache(maxsize=1)
def get_rag_engine(file_path: str) -> RAGEngine:
    questions = _load_questions(file_path)
    return RAGEngine(questions)


def warmup_rag_system(file_path: str) -> None:
    _get_embedding_model()
    get_rag_engine(file_path)