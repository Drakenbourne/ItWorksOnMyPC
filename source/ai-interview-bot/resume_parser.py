from typing import Optional

from pypdf import PdfReader

def warmup_skill_extractor() -> None:
    """Profile parsing does not require model warmup."""
    return None


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_name = (uploaded_file.name or "").lower()
    file_type = (uploaded_file.type or "").lower()
    if file_name.endswith(".pdf") or "pdf" in file_type:
        return extract_text_from_pdf(uploaded_file)

    try:
        raw = uploaded_file.getvalue()
    except Exception:
        raw = b""
    if not raw:
        return ""

    for encoding in ("utf-8", "latin-1", "utf-16"):
        try:
            return raw.decode(encoding, errors="ignore").strip()
        except Exception:
            continue
    return ""


def extract_text_from_optional_upload(uploaded_file) -> Optional[str]:
    text = extract_text_from_upload(uploaded_file)
    return text if text else None
