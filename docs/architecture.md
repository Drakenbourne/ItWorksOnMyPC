# Architecture - AI Interview Preparation Bot

## Overview

The application uses a modular RAG pipeline to generate role-relevant interview questions and evaluate user answers.

## Components

1. `app.py` (Streamlit UI)
- Handles user interactions:
  - Resume upload
  - Role selection
  - Interview flow state
  - Question display and answer submission
  - Score summary

2. `resume_parser.py`
- Extracts text from PDF resumes using `pypdf`
- Uses HuggingFace zero-shot classification (`facebook/bart-large-mnli`) to identify technical skills

3. `rag_engine.py`
- Loads interview questions from `data/interview_questions.txt`
- Creates sentence embeddings with `all-MiniLM-L6-v2`
- Builds and queries a `FAISS` index
- Retrieves top-3 role-relevant questions

4. `llm_utils.py`
- Uses HuggingFace instruction model (`google/flan-t5-small`) to evaluate each answer
- Returns structured feedback:
  - `score` (1 to 10)
  - `strengths`
  - `improvements`

## Data Flow

1. User uploads resume PDF
2. PDF text is extracted
3. Skills are inferred from resume text
4. User selects target role
5. RAG engine retrieves top interview questions
6. User answers one question at a time
7. Evaluator model scores each answer
8. Final average score is shown

## Why RAG Here

RAG grounds question retrieval in a curated interview dataset while still using semantic search. This improves relevance over fixed question lists and keeps behavior deterministic and lightweight.
