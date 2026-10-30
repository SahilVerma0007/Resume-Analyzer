"""Web server: serves the website and runs the resume analyzer."""
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import analyzer

BASE = Path(__file__).parent
MAX_BYTES = 5_000_000      # 5 MB per file
MAX_FILES = 5

app = FastAPI(title="AI Resume Analyzer")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def clean(o):
    """Turn sets, tuples and numpy numbers into plain JSON-friendly values."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return o.item()
        except Exception:
            return str(o)
    return o


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/limits")
def limits():
    return {"limits": analyzer.LIMITATIONS}


@app.post("/api/analyze")
def analyze_resumes(files: List[UploadFile] = File(...), jd: str = Form("")):
    jd = (jd or "")[:20000]
    out = []
    for f in files[:MAX_FILES]:
        name = f.filename or "resume"
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            out.append({"name": name, "error": "This file is bigger than 5 MB."})
            continue
        kind = analyzer.sniff_type(name, data)
        if kind == "doc":
            out.append({"name": name, "error": "Old .doc files are not supported. Save it as .docx or PDF and try again."})
            continue
        t0 = time.perf_counter()
        try:
            text, method = analyzer.read_resume_file(name, data)
        except Exception:
            out.append({"name": name, "error": "Could not open this file."})
            continue
        if not analyzer.is_readable(text):
            out.append({"name": name, "error": "Could not read enough clear text. Try a PDF or DOCX, or a clear, straight photo."})
            continue
        t1 = time.perf_counter()
        rating = analyzer.rate_resume(text, ocr=(method == "ocr"))
        t2 = time.perf_counter()
        jd_res = analyzer.analyze(text, jd) if jd.strip() else None
        t3 = time.perf_counter()
        print(f"TIMING file={name} kind={kind} method={method} chars={len(text)} "
              f"read={t1-t0:.2f}s rate={t2-t1:.2f}s jd={t3-t2:.2f}s total={t3-t0:.2f}s", flush=True)
        out.append({"name": name, "rating": clean(rating), "jd": clean(jd_res)})
    return JSONResponse({"results": out})
