---
title: ResumeIQ - AI Resume Analyzer
emoji: 📄
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# ResumeIQ - AI Resume Analyzer

Animated website + Python analyzer. Upload a resume (PDF, DOCX, TXT or photo) and get a 100-point rating.

## Files
- `static/index.html` - the whole website (screens + animations)
- `main.py` - server that runs the analyzer (FastAPI)
- `analyzer.py` - your resume analysis code
- `Dockerfile`, `requirements.txt` - tell the host how to install and run it

## Run on your own computer (optional)
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000

## Deploy free on Hugging Face Spaces
1. Create a free account at huggingface.co
2. Click your profile > **New Space**. Name: `resume-analyzer`. SDK: **Docker** (Blank). Hardware: **CPU basic (free)**. Visibility: Public.
3. In the Space, open **Files > Add file > Upload files**, drag in everything from this folder (including the `static` folder), then **Commit**.
4. Wait for the build (about 5-10 minutes the first time). Your site is live at `https://huggingface.co/spaces/<your-name>/resume-analyzer`.
