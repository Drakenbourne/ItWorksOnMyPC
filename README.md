# ItWorksOnMyPC - AI Interview Preparation Bot

This project is an AI-powered Interview Preparation Bot built with a Retrieval Augmented Generation (RAG) pipeline.

## Features

- Resume upload (`PDF`) and text extraction using `PyPDF`
- Technical skill extraction using a HuggingFace `transformers` model
- RAG-based interview question retrieval using:
  - `sentence-transformers` (`all-MiniLM-L6-v2`)
  - `FAISS` vector database
- Interactive interview mode (question by question)
- Answer evaluation using a HuggingFace model
- Final interview score summary
- Streamlit UI for end-to-end workflow

## Project Structure

```text
ItWorksOnMyPC/
├── README.md
├── source/
│   └── ai-interview-bot/
│       ├── app.py
│       ├── rag_engine.py
│       ├── resume_parser.py
│       ├── llm_utils.py
│       ├── requirements.txt
│       └── data/
│           └── interview_questions.txt
└── docs/
    └── architecture.md
```

## Run Locally

1. Open terminal in `source/ai-interview-bot`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start app:
   - `streamlit run app.py`

## Notes

- First run downloads model weights, so startup may take longer.
- CPU execution is supported; GPU is optional.

- https://aiinterviewprep.streamlit.app/
