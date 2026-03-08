# ItWorksOnMyPC - AI Interview Preparation Bot

This project is an AI-powered Interview Preparation Bot built with a Retrieval Augmented Generation (RAG) pipeline.

## Features

- Resume upload (`PDF`) and text extraction using `PyPDF`
- AI profile analysis from resume + optional job description:
  - infer role automatically from JD, or use manual role selection
  - infer technical skills using LLM extraction
- RAG-based interview question retrieval using:
  - `sentence-transformers` (`all-MiniLM-L6-v2`)
  - `FAISS` vector database
- Interactive interview mode (question by question)
- Interviewer tone controls: `Easy`, `Medium`, `Strict`
- Answer evaluation with provider switch:
  - Local Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct` by default)
  - Puter Gemini bridge (`google/gemini-2.5-flash-lite` by default)
- Final interview score summary
- Streamlit UI for end-to-end workflow
- Apple Silicon Metal (`MPS`) support for PyTorch-backed models

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
│       ├── package.json
│       ├── puter_bridge.js
│       ├── puter_login.js
│       └── data/
│           └── interview_questions.txt
└── docs/
    └── architecture.md
```

## Run Locally

1. Open terminal in `source/ai-interview-bot`
2. Create and activate virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Start app:
   - `streamlit run app.py`

## Use Gemini via Puter (Optional)

Run these in `source/ai-interview-bot`:

1. Install JS dependency:
   - `npm install`
2. Login once and copy token:
   - `npm run puter:login`
3. Export token + provider:
   - `export PUTER_AUTH_TOKEN="<paste-token>"`
   - `export EVALUATOR_PROVIDER=puter`
4. Optional model override:
   - `export PUTER_MODEL=google/gemini-2.5-flash-lite`
5. Run app:
   - `streamlit run app.py`

## Notes

- First run downloads model weights, so startup may take longer.
- On Apple Silicon Macs, models use Metal GPU (`MPS`) when available.
- For Puter Gemini mode, RAG/embeddings still run locally; evaluator/reference generation run via Puter.
- Typing animation for AI outputs is enabled in the UI.
- https://aiinterviewprep.streamlit.app/

## Deployment (Railway / Fly.io)

### Railway
1. Push this repo to GitHub.
2. In Railway: New Project -> Deploy from GitHub Repo.
3. Railway will use `Dockerfile` automatically.
4. Set optional env vars if needed.
5. Deploy and open the generated URL.

### Fly.io
1. Install Fly CLI and login: `fly auth login`
2. From repo root run: `fly launch --no-deploy`
3. Keep `Dockerfile` and use the included `fly.toml` (update `app` name if needed).
4. Deploy: `fly deploy`
5. Open app: `fly open`

### Notes
- This app is CPU by default unless host machine provides GPU.
- If model load is slow, first boot may take longer.
