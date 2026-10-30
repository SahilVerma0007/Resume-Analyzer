"""Resume analyzer logic (from notebook v9). No Colab code here."""
# =====================  AI RESUME ANALYZER v8  (Google Colab)  =====================
# Run the cell, then upload one or more resumes (PDF, DOCX, TXT or an image). No job description needed.

JOB_DESCRIPTION = ""   # OPTIONAL. Leave empty to just rate the resume. Paste a job post to also see a match score.

USE_EMBEDDINGS = False   # True = also use sentence embeddings (needs: !pip install -q sentence-transformers)
CERT_POINTS_MAX = 10     # points (out of the 100-point rubric) for certification courses.

# ------------------------------------------------------------------------------
import io, re, html
import functools
from pypdf import PdfReader

import os
try:
    if os.environ.get("USE_SPACY", "1") == "0":
        raise RuntimeError("spaCy switched off by USE_SPACY=0")
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    except Exception:
        import importlib, subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
        importlib.invalidate_caches()
        NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
except Exception:
    NLP = None
print("spaCy loaded:", NLP is not None)

_EMB_MODEL = None

# ---------------------------- 1. Knowledge base -------------------------------
SKILLS = {
    "python": [], "java": [], "c++": ["cpp"], "javascript": ["js"], "sql": [],
    "html": [], "css": [], "react": ["reactjs"], "flask": [], "django": [],
    "fastapi": [], "streamlit": [],
    "machine learning": ["ml"], "deep learning": ["dl"],
    "nlp": ["natural language processing"], "computer vision": [],
    "data analysis": ["data analytics"], "data visualization": ["data visualisation"],
    "statistics": [], "feature engineering": [],
    "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"],
    "matplotlib": [], "seaborn": [], "tensorflow": [], "keras": [],
    "pytorch": [], "opencv": [], "spacy": [], "nltk": [],
    "transformers": [], "hugging face": ["huggingface"],
    "regression": [], "classification": [], "clustering": [],
    "neural networks": ["neural network"], "cnn": [], "lstm": [],
    "mysql": [], "postgresql": ["postgres"], "mongodb": [],
    "git": [], "github": [], "docker": [], "linux": [],
    "aws": [], "azure": [], "gcp": ["google cloud"], "mlops": [],
    "rest api": ["rest apis", "restful api"], "excel": [],
    "power bi": ["powerbi"], "tableau": [],
}
DEGREE_PATTERN = re.compile(
    r"\b(b\.?\s?tech|b\.?\s?e\.?|m\.?\s?tech|bachelor[^,\n]*|master[^,\n]*|"
    r"mba|b\.?\s?sc|m\.?\s?sc|b\.?\s?a\.?|m\.?\s?a\.?|ph\.?d|diploma)\b", re.I)
INSTITUTION_WORDS = ("university", "college", "institute", "school", "iit", "nit")
ROLE_WORDS = ("intern", "internship", "engineer", "developer", "analyst",
              "trainee", "assistant", "researcher", "associate")
SECTION_WORDS = {"education": ["education", "academic"], "projects": ["project"],
                 "experience": ["experience", "internship"], "skills": ["skills"]}
LIMITATIONS = [
    "A score is a guide, not a hiring decision. It cannot judge real ability.",
    "Skills are found by keywords, so unusual wording can be missed.",
    "Fancy PDF layouts (columns, icons, images) can break text extraction.",
    "Name, email, phone and gender words are removed before scoring to reduce bias, "
    "but bias can still enter through the job description or skill list.",
    "Similarity rewards resumes that copy the job description's wording.",
    "Job suggestions use simple hand-made skill profiles, not live job-market data.",
    "The rating rubric and its point values were designed by hand and are not validated on real hiring outcomes.",
    "The rubric is built for tech resumes, so non-tech resumes score low by design.",
    "Certifications are counted by name only. The tool cannot check that a certificate is real. Course lines are found by headings and keywords, so an unusual layout can still be missed.",
]

# ---------------------------- 2. Safeguards -----------------------------------
def anonymize(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " ", text)
    text = re.sub(r"\+?\d[\d\s().-]{8,}\d", " ", text)
    text = re.sub(r"(?im)^.*\b(date of birth|dob|gender|sex|marital status)\b.*$", " ", text)
    text = re.sub(r"\b(he|she|his|her|him|male|female|mr|mrs|ms)\b", " ", text, flags=re.I)
    return text

def check_parse_quality(text):
    warnings, words = [], text.split()
    if len(words) < 50:
        warnings.append("Very little text was found. Either the resume is very short, or the file is a scanned image.")
    if words and sum(len(w) for w in words) / len(words) < 3.2:
        warnings.append("Words look broken. A multi-column layout may have scrambled the text.")
    if re.search(r"(?:\b\w\b\s){6,}", text):
        warnings.append("Letters are spaced apart (like 'P y t h o n'). Fancy fonts can cause this.")
    symbols = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    if text and symbols / len(text) > 0.25:
        warnings.append("Many symbols found. Icons or graphics may have confused the parser.")
    return warnings

# ---------------------------- 3. Extraction -----------------------------------
def _has_term(text, term):
    p = r"(?<![A-Za-z0-9+#])" + re.escape(term) + r"(?![A-Za-z0-9+#])"
    return re.search(p, text, flags=re.I) is not None

def extract_skills(text):
    return {n for n, al in SKILLS.items() if any(_has_term(text, t) for t in [n] + al)}

def extract_education(text):
    degrees = sorted({m.group(0).strip() for m in DEGREE_PATTERN.finditer(text)})
    inst = [l.strip(" -•|:").strip() for l in text.splitlines()
            if any(w in l.lower() for w in INSTITUTION_WORDS)]
    cgpa = re.search(r"(?:cgpa|gpa)[:\s]*([\d.]+)", text, flags=re.I)
    return {"degrees": degrees[:4], "institutions": inst[:3], "cgpa": cgpa.group(1) if cgpa else None}

def extract_experience(text):
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _extract_experience_cached(text)}

@functools.lru_cache(maxsize=32)
def _extract_experience_cached(text):
    role_re = re.compile(r"\b(" + "|".join(ROLE_WORDS) + r")\b", re.I)
    roles = [l.strip(" -•|:").strip() for l in text.splitlines()
             if role_re.search(l) and not l.strip().lower().startswith("skills")]
    years = [float(y) for y in re.findall(r"(\d+(?:\.\d+)?)\+?\s*years?", text, flags=re.I)]
    orgs, dates = [], []
    if NLP is not None:
        doc = NLP(text[:6000])
        orgs = sorted({e.text.strip() for e in doc.ents if e.label_ == "ORG"})[:6]
        dates = sorted({e.text.strip() for e in doc.ents if e.label_ == "DATE"})[:6]
    return tuple({"roles": roles[:4], "years": max(years) if years else 0, "orgs": orgs, "dates": dates}.items())

# ---------------------------- 4. ML similarity --------------------------------
def tfidf_similarity(resume, jd):
    from sklearn.feature_extraction.text import TfidfVectorizer   # loaded only when a job description is used
    from sklearn.metrics.pairwise import cosine_similarity
    m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([resume, jd])
    return min(1.0, cosine_similarity(m[0:1], m[1:2])[0][0] / 0.5)

def embedding_similarity(resume, jd):
    global _EMB_MODEL
    if not USE_EMBEDDINGS:
        return None
    try:
        if _EMB_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        e = _EMB_MODEL.encode([resume[:3000], jd[:3000]])
        from sklearn.metrics.pairwise import cosine_similarity
        raw = cosine_similarity(e[0:1], e[1:2])[0][0]
        return float(min(1.0, max(0.0, (raw - 0.2) / 0.6)))
    except Exception:
        return None

# ---------------------------- 5. Feedback -------------------------------------
def build_tips(resume, missing, edu, exp):
    tips, low = [], resume.lower()
    for sec, words in SECTION_WORDS.items():
        if not any(w in low for w in words):
            tips.append(f"Add a '{sec}' section.")
    if missing:
        tips.append("Learn or add these skills if you have them: " + ", ".join(sorted(missing)[:5]) + ".")
    if not edu["degrees"]:
        tips.append("State your degree clearly (for example B.Tech CSE).")
    if not exp["roles"]:
        tips.append("Add an internship, project role, or training with your role title.")
    if not re.search(r"\d+\s*%|\b\d{2,}\b", resume):
        tips.append("Add numbers to achievements (accuracy, dataset size, users).")
    n = len(resume.split())
    if n < 80: tips.append("Resume is very short. Add more detail.")
    elif n > 900: tips.append("Resume is too long. Aim for one page.")
    return tips

# ---------------------------- 5b. Job role suggestions ------------------------
# Simplified skill profiles written by hand. They are NOT from live job-market data.
ROLES = {
    "Machine Learning Engineer": {
        "core": ["python", "machine learning", "scikit-learn", "pandas", "numpy"],
        "bonus": ["tensorflow", "pytorch", "docker", "git", "mlops", "feature engineering"]},
    "Data Scientist": {
        "core": ["python", "statistics", "machine learning", "pandas", "scikit-learn", "data analysis"],
        "bonus": ["regression", "classification", "clustering", "matplotlib", "nlp", "sql"]},
    "Data Analyst": {
        "core": ["sql", "excel", "python", "pandas", "data visualization"],
        "bonus": ["tableau", "power bi", "statistics", "matplotlib", "seaborn"]},
    "NLP Engineer": {
        "core": ["python", "nlp", "deep learning", "transformers"],
        "bonus": ["spacy", "nltk", "hugging face", "pytorch", "tensorflow"]},
    "Computer Vision Engineer": {
        "core": ["python", "opencv", "deep learning", "cnn"],
        "bonus": ["pytorch", "tensorflow", "keras", "numpy", "computer vision"]},
    "MLOps / Deployment Engineer": {
        "core": ["python", "docker", "git", "linux", "mlops"],
        "bonus": ["aws", "azure", "gcp", "fastapi", "flask", "rest api"]},
    "Backend Developer": {
        "core": ["python", "sql", "rest api", "git"],
        "bonus": ["flask", "django", "fastapi", "docker", "mysql", "postgresql", "mongodb"]},
    "Full Stack Developer": {
        "core": ["javascript", "react", "html", "css", "sql", "rest api"],
        "bonus": ["flask", "django", "git", "mongodb", "docker"]},
    "Frontend Developer": {
        "core": ["html", "css", "javascript", "react", "git"],
        "bonus": ["rest api", "sql"]},
}

def suggest_roles(resume_skills, top_n=4, credit=None):
    """Score the resume against each role profile. Core skills weigh 75%, bonus 25%."""
    ranked = []
    for role, p in ROLES.items():
        have_core = [s for s in p["core"] if s in resume_skills]
        have_bonus = [s for s in p["bonus"] if s in resume_skills]
        w = (lambda sk: credit.get(sk, 0)) if credit else (lambda sk: 1 if sk in resume_skills else 0)
        score = round(100 * (0.75 * sum(w(x) for x in p["core"]) / len(p["core"]) +
                             0.25 * sum(w(x) for x in p["bonus"]) / len(p["bonus"])))
        ranked.append({
            "role": role, "score": score,
            "have": have_core + have_bonus,
            "need": [s for s in p["core"] if s not in resume_skills],
            "need_bonus": [s for s in p["bonus"] if s not in resume_skills],
            "label": "Ready to apply" if score >= 65 else "Close: build a few skills" if score >= 35 else "Stretch goal",
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]

def skills_to_build(top_roles, resume_skills):
    """Skills missing from the best-fit roles, ordered by how many roles need them."""
    count = {}
    for r in top_roles[:3]:
        for sk in r["need"]:
            count[sk] = count.get(sk, 0) + 1
    ordered = sorted(count, key=lambda k: (-count[k], k))
    if not resume_skills:
        ordered = ["python", "sql", "git"]
    return ordered[:6]

# ---------------------------- 6. Main analysis --------------------------------
def analyze(resume_raw, jd):
    warnings = check_parse_quality(resume_raw)
    resume = anonymize(resume_raw)
    rs, js = extract_skills(resume), extract_skills(jd)
    matched, missing, extra = rs & js, js - rs, rs - js
    skill_score = len(matched) / len(js) if js else 0.0
    tfidf = tfidf_similarity(resume, jd)
    emb = embedding_similarity(resume, jd)
    if emb is None:
        final, method = 0.6 * skill_score + 0.4 * tfidf, "60% skills + 40% TF-IDF"
    else:
        final, method = 0.5 * skill_score + 0.2 * tfidf + 0.3 * emb, "50% skills + 20% TF-IDF + 30% embeddings"
    final = round(final * 100)
    edu, exp = extract_education(resume_raw), extract_experience(resume_raw)
    return {
        "score": final, "method": method,
        "verdict": "Strong match" if final >= 70 else "Partial match" if final >= 45 else "Weak match",
        "skill_score": round(skill_score * 100), "tfidf": round(tfidf * 100),
        "emb": None if emb is None else round(emb * 100),
        "matched": sorted(matched), "missing": sorted(missing), "extra": sorted(extra),
        "edu": edu, "exp": exp, "tips": build_tips(resume, missing, edu, exp),
        "warnings": warnings, "jd_skills": len(js),
        "roles": suggest_roles(rs), "build": skills_to_build(suggest_roles(rs), rs),
        "no_skills": len(rs) == 0,
    }

# ---------------------------- 6b. Resume rating (NO job description needed) ---
# A transparent 100-point rubric. Every point can be explained:
#   Skills 30 | Best job fit 20 | Evidence 20 | Structure 10 | Education 5 | Length 5 | Certifications 10
# Skills you PROVE (used in a project or job line) count fully.
# Skills only LISTED in the Skills section count 20%. This stops keyword stuffing.
HEADER_RE = re.compile(r"^(education|academic|projects?|experience|internship|certifications?|"
                       r"achievements?|summary|objective|work experience|interests|hobbies|certificates?|courses?|training|licen[sc]es)\b")
ACTION_RE = re.compile(r"\b(built|developed|created|designed|trained|deployed|implemented|analy[sz]ed|"
                       r"wrote|led|managed|organi[sz]ed|classifier|dashboard|pipeline|project)\b", re.I)
METRIC_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\b(?!(?:19|20)\d\d\b)\d{3,}\b")

def split_skill_lines(text):
    """Separate the Skills block from the rest, to tell 'listed' skills from 'proved' skills."""
    skill_lines, other, in_skills = [], [], False
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith(("skills", "technical skills", "key skills", "core skills")):
            in_skills = True
        elif in_skills and (not low or HEADER_RE.match(low)):
            in_skills = False
        (skill_lines if in_skills else other).append(line)
    return "\n".join(skill_lines), "\n".join(other)

# ---------------------------- 6c. Certifications and courses (v8) -------------
# v8 finds certificates and courses ANYWHERE in the resume, in three ways:
#   1) A section whose heading is about certificates, courses, licenses, training or credentials.
#      Headings may be UPPERCASE, use '&', carry icons, or be followed by blank lines.
#      The section runs until the next real heading (Education, Skills, Projects ...).
#   2) A one-line form such as "Certifications: Python (Coursera); SQL (Udemy)".
#   3) Single lines in ANY other section (Education, Achievements, Projects ...) that clearly
#      describe a course or certificate, for example "Completed NPTEL course on Data Structures".
_BULLET_CHARS = ("•●○◦▪▫■□◆◇❖➢➤➔►▶"
                 "✓✔✦★☆·‣⁃")

CERT_NOUNS = {"certification", "certifications", "certificate", "certificates", "course", "courses",
              "license", "licenses", "licence", "licences", "training", "trainings", "credential",
              "credentials", "mooc", "moocs", "workshop", "workshops", "accreditation", "accreditations"}
HEAD_FILLER = {"and", "&", "/", "or", "online", "professional", "development", "completed", "additional",
               "other", "extra", "technical", "learning", "e-learning", "elearning", "self", "paced",
               "specialization", "specializations", "specialisation", "specialisations", "program",
               "programs", "programme", "programmes", "my", "key", "relevant", "details", "certified",
               "industry", "short", "term", "professional", "licensure", "qualifications", "qualification"}
HEAD_MIXED = {"education", "academic", "academics", "skills", "achievements", "achievement", "awards",
              "honors", "honours", "projects", "extracurricular", "activities", "experience",
              "interests", "hobbies", "internships", "internship"}
OTHER_HEADS = {
    "education", "academic", "academics", "academic qualifications", "educational qualifications",
    "qualifications", "projects", "project", "personal projects", "academic projects", "experience",
    "work experience", "professional experience", "internships", "internship", "skills", "technical skills",
    "key skills", "core skills", "skills and tools", "tools and technologies", "technologies", "summary",
    "professional summary", "career objective", "objective", "profile", "about me", "achievements",
    "awards", "honors", "honours", "awards and achievements", "interests", "hobbies", "languages",
    "references", "declaration", "personal details", "personal information", "personal profile",
    "extracurricular", "extra curricular", "extracurricular activities", "activities", "publications",
    "positions of responsibility", "volunteering", "volunteer", "volunteer experience", "contact",
    "contact details", "links", "strengths", "responsibilities", "projects and internships",
    "education and skills", "work history", "employment history", "career summary"}

STRONG_COURSE_RE = re.compile(
    r"\b(certificat(?:e|es|ion|ions)|certified|nanodegree|specializations?|specialisations?|bootcamp|boot camp|"
    r"masterclass|courses?|moocs?|training programme?s?)\b", re.I)
PLATFORM_RE = re.compile(
    r"\b(coursera|udemy|edx|nptel|swayam|datacamp|udacity|simplilearn|great\s?learning|upgrad|linkedin learning|"
    r"freecodecamp|codecademy|deeplearning\.ai|infosys springboard|skillshare|pluralsight|internshala|"
    r"coding ninjas|scaler|alison|futurelearn)\b", re.I)
PROVIDERS = re.compile(r"\b(coursera|udemy|edx|nptel|swayam|google|aws|amazon|microsoft|ibm|infosys|great\s?learning|"
                       r"simplilearn|datacamp|kaggle|deeplearning\.ai|linkedin learning|udacity|upgrad|oracle|cisco|"
                       r"meta|hackerrank|freecodecamp|codecademy|nvidia|stanford|tcs|wipro|salesforce|red hat|"
                       r"servicenow|pmi|scrum|comptia|isc2|infosys springboard|internshala)\b", re.I)
NOISE_RE = re.compile(r"^(credential (id|url|link)|certificate (id|no|number|url|link)|licen[sc]e (no|number|id)|"
                      r"verif|issued|issue date|expir|valid (till|until)|id\s*[:#]|url\s*:|link\s*:|https?:|www\.)", re.I)
REJECT_BODY_RE = re.compile(
    r"\b(built|developed|created|designed|trained|deployed|implemented|analy[sz]ed|wrote|led|managed|organi[sz]ed|"
    r"intern|engineer|developer|analyst|trainee|assistant|researcher|teaching|taught|conducted)\b", re.I)
MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I)

def _norm_line(line):
    line = line.replace("\xa0", " ").replace("​", "")
    for ch in _BULLET_CHARS:
        line = line.replace(ch, "•")
    line = re.sub(r"[ \t]+", " ", line).strip()
    if re.fullmatch(r"(?:[A-Za-z&] ){3,}[A-Za-z&]", line):      # "C E R T I F I C A T I O N S"
        line = line.replace(" ", "")
    return line

def _clean_item(s):
    return s.strip(" \t-*•|:;,._>#=~–—").strip()

def _head_words(text):
    return re.findall(r"[a-z][a-z\-]*|&|/", text.lower())

def _cert_header(line):
    """Return (is_cert_heading, text_on_same_line, mixed_heading).
    A heading is made ONLY of heading-type words (certifications, courses, and, &, online ...), so a real
    item such as 'Python Course' is never mistaken for a heading."""
    clean = _clean_item(line)
    if not clean or len(clean) > 120:
        return False, "", False
    left, rest = clean, ""
    m = re.match(r"^(.{2,70}?)\s*(?::|\s[–—-]\s)\s*(.+)$", clean)
    if m:
        left, rest = m.group(1), m.group(2)
    words = _head_words(left)
    if not words or len(words) > 8 or re.search(r"\d", left):
        return False, "", False
    if not any(w in CERT_NOUNS for w in words):
        return False, "", False
    if not all(w in CERT_NOUNS or w in HEAD_FILLER or w in HEAD_MIXED for w in words):
        return False, "", False
    mixed = any(w in HEAD_MIXED for w in words)
    return True, rest.strip(), mixed

def _is_other_header(line):
    clean = _clean_item(line).lower().replace("&", " and ")
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean or len(clean.split()) > 7:
        return False
    left = clean.split(":")[0].strip()
    if left in OTHER_HEADS or clean in OTHER_HEADS:
        return True
    if re.fullmatch(r"(?:" + "|".join(re.escape(h) for h in sorted(OTHER_HEADS, key=len, reverse=True)) +
                    r")(?: (?:and|/) (?:" + "|".join(re.escape(h) for h in OTHER_HEADS) + r"))?", left):
        return True
    raw = _clean_item(line)
    first = clean.split()[0] if clean.split() else ""
    return (raw.isupper() or line.rstrip().endswith(":")) and any(
        clean == h or clean.startswith(h + " ") for h in OTHER_HEADS) and first != ""

def _split_commas(p):
    if len(PLATFORM_RE.findall(p)) >= 2 or len(STRONG_COURSE_RE.findall(p)) >= 2:
        return [x for x in re.split(r",\s*(?![^()]*\))", p) if x.strip()]
    return [p]

def _looks_like_course(item):
    return bool(STRONG_COURSE_RE.search(item) or PLATFORM_RE.search(item)
                or (PROVIDERS.search(item) and len(item.split()) >= 3))

def _valid_item(p, strict):
    if not p or len(p) > 220 or NOISE_RE.match(p):
        return False
    if len(re.sub(r"[\d\W_]+", "", MONTH_RE.sub("", p))) < 4:       # only a date or symbols
        return False
    if len(p.split()) < 2 and not PLATFORM_RE.search(p):
        return False
    if strict and not _looks_like_course(p):
        return False
    return True

def _merge_wrapped(lines):
    """Join lines that are the continuation of the previous course (long names wrap onto 2 lines)."""
    entries = []
    for ln in lines:
        bullet = ln.startswith("•")
        t = _clean_item(ln)
        if not t:
            continue
        if entries and not bullet:
            prev = entries[-1]
            frag = re.sub(r"[\d\W_]+", " ", MONTH_RE.sub("", t)).strip()
            provider_only = (PLATFORM_RE.match(t) or PROVIDERS.match(t)) and len(t.split()) <= 5 \
                and not STRONG_COURSE_RE.search(t)
            if (t[0].islower() or prev[-1:] in ",-–—&/(" or
                    prev.lower().endswith((" of", " and", " to", " in", " for", " with", " on", " by")) or
                    provider_only or not frag):
                entries[-1] = prev + " " + t
                continue
        entries.append(t)
    return entries

def _body_cert_line(ln):
    if len(ln) > 220 or len(ln.split()) > 32:
        return False
    if not (STRONG_COURSE_RE.search(ln) or PLATFORM_RE.search(ln)):
        return False
    if re.search(r"certif|nanodegree|specializ|specialis", ln, re.I):
        return True
    return not REJECT_BODY_RE.search(ln)

def _scan_certs(text):
    """Returns (found, remaining_text). found = list of (course text, where it was found)."""
    sections, cur, remaining = [], None, []
    taken = 0
    for raw in text.splitlines():
        ln = _norm_line(raw)
        if not ln:
            remaining.append("")
            continue
        is_head, inline, mixed = _cert_header(ln)
        if is_head:
            cur = {"lines": [], "strict": mixed}
            sections.append(cur)
            taken = 0
            if inline:
                cur["lines"].append(inline)
            continue
        if cur is not None and (_is_other_header(ln) or taken >= 40):
            cur = None
        if cur is not None:
            cur["lines"].append(ln)
            taken += 1
            continue
        remaining.append(ln)

    found, seen = [], []
    def add(item, where):
        key = re.sub(r"[^a-z0-9]", "", item.lower())
        if not key or any(key == k or key in k or k in key for k in seen):
            return
        seen.append(key)
        found.append((item, where))

    for sec in sections:
        for entry in _merge_wrapped(sec["lines"]):
            for part in re.split(r"[;|•]", entry):
                for p in _split_commas(part):
                    p = _clean_item(p)
                    if _valid_item(p, sec["strict"]):
                        add(p, "certifications / courses section")

    kept = []
    for ln in remaining:
        if ln and not _is_other_header(ln) and _body_cert_line(ln):
            for part in re.split(r"[;|•]", ln):
                p = _clean_item(part)
                if p and _valid_item(p, True):
                    add(p, "elsewhere in the resume")
        else:
            kept.append(ln)
    return found, "\n".join(kept)

def find_certifications(text):
    """List of (course text, where found)."""
    return _scan_certs(text)[0]

def extract_certifications(text):
    return [c for c, _ in find_certifications(text)]

def split_cert_lines(text):
    """Returns (course lines, the resume text without them). Kept for the skill-proof step."""
    found, rest = _scan_certs(text)
    return [c for c, _ in found], rest

def rate_resume(resume_raw, ocr=False):
    warnings = check_parse_quality(resume_raw)
    if ocr:
        warnings.append("Text was read from an image (OCR), so a few words may be wrong. A clear, straight photo gives the best result.")
    text = anonymize(resume_raw)
    _, no_certs = split_cert_lines(text)          # course names show learning, not use,
    _, rest = split_skill_lines(no_certs)         # so skills named only there count as 'listed'
    all_sk = extract_skills(text)
    proved = extract_skills(rest)
    listed = all_sk - proved
    credit = {k: 1.0 for k in proved}
    credit.update({k: 0.2 for k in listed})
    low = text.lower()
    edu, exp = extract_education(resume_raw), extract_experience(resume_raw)

    # 1. Skills (30): 12 fully proved skills = full marks
    skills_pts = min(30.0, 30 * sum(credit.values()) / 12)
    # 2. Best job fit (20): how well the skills fill the best-matching role
    roles = suggest_roles(all_sk, credit=credit)
    fit_pts = 20 * roles[0]["score"] / 100
    # 3. Evidence (20): numbers, a real role, and lines that describe things you built or did
    lines = [l for l in rest.splitlines() if l.strip()]
    metric_n = sum(1 for l in lines if METRIC_RE.search(l))
    action_n = sum(1 for l in lines if ACTION_RE.search(l))
    evid_pts = 8 * min(3, metric_n) / 3 + (6 if exp["roles"] else 0) + 6 * min(3, action_n) / 3
    # 4. Structure (10): four sections + contact email, 2 points each
    sections_found = [k for k, ws in SECTION_WORDS.items() if any(w in low for w in ws)]
    has_email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_raw))
    struct_pts = 2 * len(sections_found) + (2 if has_email else 0)
    # 5. Education (5)
    edu_pts = (2.5 if edu["degrees"] else 0) + (1.5 if edu["institutions"] else 0) + (1 if edu["cgpa"] else 0)
    # 6. Length (5)
    n = len(resume_raw.split())
    len_pts = 5 if 150 <= n <= 700 else 3 if (80 <= n < 150 or 700 < n <= 900) else 0

    certs_found = find_certifications(resume_raw)   # v8: searches the WHOLE resume, not only one section
    certs = [c for c, _ in certs_found]
    n_sec = sum(1 for _, w in certs_found if w.startswith('certifications'))
    n_body = len(certs_found) - n_sec
    cert_pts = round(CERT_POINTS_MAX * min(len(certs), 3) / 3, 1)   # 3 or more courses = full 10 points

    breakdown = [
        ("Technical skills", skills_pts, 30, f"{len(proved)} proved, {len(listed)} only listed"),
        ("Best job fit", fit_pts, 20, f"closest role: {roles[0]['role']}"),
        ("Evidence of work", evid_pts, 20, f"{metric_n} lines with numbers, {action_n} lines describing work"),
        ("Structure", struct_pts, 10, f"sections found: {', '.join(sections_found) or 'none'}"),
        ("Education", edu_pts, 5, "degree, institution, CGPA"),
        ("Length", len_pts, 5, f"{n} words"),
        ("Certifications", cert_pts, CERT_POINTS_MAX, f"{len(certs)} course(s) found ({n_sec} in a certifications/courses section, {n_body} elsewhere)"),
    ]
    score = min(100, round(sum(b[1] for b in breakdown)))

    tips = []
    if listed and len(listed) > len(proved):
        tips.append("Show where you used these skills (in a project or job line): " + ", ".join(sorted(listed)[:5]) + ".")
    if metric_n < 2:
        tips.append("Add numbers to your work: accuracy, dataset size, users, time saved.")
    if not exp["roles"]:
        tips.append("Add an internship, project role or training with your role title.")
    for sec in ("education", "projects", "experience", "skills"):
        if sec not in sections_found:
            tips.append(f"Add a '{sec}' section.")
    if not has_email:
        tips.append("Add an email address.")
    if not edu["degrees"]:
        tips.append("State your degree clearly (for example B.Tech CSE).")
    if len_pts < 5:
        tips.append("Aim for roughly one page: about 150 to 700 words.")
    if not certs:
        tips.append("Add a Certifications section with 1 to 3 relevant courses (course name and provider).")

    return {
        "score": score,
        "tier": "Strong" if score >= 75 else "Good" if score >= 55 else "Needs work" if score >= 35 else "Weak",
        "breakdown": breakdown, "proved": sorted(proved), "listed": sorted(listed),
        "roles": roles, "build": skills_to_build(roles, all_sk),
        "no_skills": len(all_sk) == 0, "edu": edu, "exp": exp,
        "tips": tips, "warnings": warnings, "certs": certs, "cert_where": certs_found,
    }

# ---------------------------- 7. Pretty report --------------------------------
def _chips(items, bg, fg, border="none"):
    if not items: return "<i style='color:#6B7596'>None</i>"
    return "".join(f"<span style='display:inline-block;margin:3px 5px 3px 0;padding:4px 11px;border-radius:8px;"
                   f"background:{bg};color:{fg};border:{border};font-size:14px'>{html.escape(i)}</span>" for i in items)

def _card(title, body):
    return (f"<div style='background:#fff;border:1.5px solid #DDE3F0;border-radius:16px;padding:16px 20px;"
            f"margin:10px 0'><div style='font-weight:700;font-size:17px;margin-bottom:8px'>{title}</div>{body}</div>")

# ---------------------------- 8. Read the resume ------------------------------
# ---------------------------- 8. Reading files (robust) -----------------------
# The file TYPE is detected from the file's first bytes, not from its name.
# So a Word file renamed .pdf, or a photo with the wrong ending, still opens correctly.
import zipfile

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()          # lets iPhone HEIC gallery photos open
except Exception:
    pass

def text_quality(text):
    """0 to 1: how much this looks like real, readable writing."""
    if not text or not text.strip():
        return 0.0
    ok = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:-|()/&%+#@'\"•–—_*[]")
    tokens = text.split()
    wordlike = sum(1 for t in tokens if sum(ch.isalpha() for ch in t) >= 2) / max(1, len(tokens))
    return 0.6 * ok / len(text) + 0.4 * wordlike

COMMON_WORDS = set("""the and of to in a for with on at by as is was are an or from using used use built
developed worked work project projects education skills experience internship intern university college
institute school bachelor master technology computer science engineering data software developer engineer
analyst summary objective achievements certifications languages github linkedin team lead design analysis
model models training trained deployed application applications system systems management research year years
award cgpa gpa email phone address contact responsible created implemented managed learning web mobile
database cloud api tools technical programming course courses certificate semester degree""".split())
COMMON_WORDS |= {w for k in SKILLS for w in re.findall(r"[a-z]+", k)}

def known_ratio(text):
    """Share of words that are common resume or English words. Gibberish scores near zero."""
    toks = re.findall(r"[a-z]+", text.lower())
    return sum(1 for t in toks if t in COMMON_WORDS) / max(1, len(toks))

def text_score(text):
    """Used to pick the best reading of a file: long, clean and made of real words."""
    return text_quality(text) * min(len(text.split()), 400) * (0.02 + known_ratio(text))

def clean_ctrl(text):
    """Some PDFs store bullets as control characters. Turn them into plain bullets."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "\u2022", text)

def is_readable(text):
    return (len(text.split()) >= 10 and text_quality(text) >= 0.8
            and (known_ratio(text) >= 0.06 or len(extract_skills(text)) >= 3))

def looks_garbled(text):
    return not is_readable(text)

def _ocr_best(img):
    """Try two page layouts and keep the better text."""
    import pytesseract
    best, best_score = "", -1
    for cfg in ("--psm 3", "--psm 6"):
        try:
            t = pytesseract.image_to_string(img, config=cfg)
        except Exception:
            continue
        sc = text_score(t)
        if sc > best_score:
            best, best_score = t, sc
    return best

def ocr_image(img):
    """Read text from a picture. Fixes phone rotation and sideways pictures."""
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img).convert("L")       # fix phone rotation, make grayscale
    if img.width > 3500:                                  # very large photos are slow
        k = 3500 / img.width
        img = img.resize((int(img.width * k), int(img.height * k)))
    elif img.width < 1500:                                # small pictures read badly, so enlarge
        k = 1500 / img.width
        img = img.resize((int(img.width * k), int(img.height * k)))
    img = ImageOps.autocontrast(img)
    best = _ocr_best(img)
    if not is_readable(best):                             # maybe the picture is sideways or upside down
        best_score = text_score(best)
        for angle in (90, 270, 180):
            t = _ocr_best(img.rotate(angle, expand=True))
            sc = text_score(t)
            if sc > best_score:
                best, best_score = t, sc
    return best

def read_pdf(data):
    """Returns (text, method). Tries two text readers, then OCR for scanned PDFs."""
    candidates = []
    try:
        r = PdfReader(io.BytesIO(data))
        candidates.append(("pypdf", clean_ctrl("\n".join((p.extract_text() or "") for p in r.pages))))
    except Exception:
        r = None
    fitz = None
    try:
        import pymupdf as fitz
    except Exception:
        try:
            import fitz
        except Exception:
            fitz = None
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            candidates.append(("pymupdf", clean_ctrl("\n".join(p.get_text("text", sort=True) for p in doc))))
        except Exception:
            pass
    good = [c for c in candidates if is_readable(c[1])]
    if good:
        best = max(good, key=lambda c: text_score(c[1]))
        return best[1], best[0]

    # No readable text: probably a scanned PDF or unusual fonts. Read the page pictures with OCR.
    try:
        from PIL import Image
        import pytesseract  # noqa: F401
    except ImportError:
        return (max(candidates, key=lambda c: len(c[1]))[1] if candidates else ""), "failed"
    pages = []
    if r is not None:                                     # route 1: pictures stored inside the PDF
        for page in list(r.pages)[:3]:
            try:
                for im in page.images:
                    try:
                        pages.append(ocr_image(Image.open(io.BytesIO(im.data))))
                    except Exception:
                        pass
            except Exception:
                pass
    text = "\n".join(pages)
    if not is_readable(text) and fitz is not None:        # route 2: draw each page, then OCR it
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in list(doc)[:3]:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                pages.append(ocr_image(Image.open(io.BytesIO(pix.tobytes("png")))))
            text = "\n".join(pages)
        except Exception:
            pass
    if is_readable(text):
        return text, "ocr"
    return (max(candidates, key=lambda c: len(c[1]))[1] if candidates else text), "failed"

def read_docx(data):
    """Read a Word file with no extra install (a .docx is a zip with XML inside). Includes tables."""
    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"<w:tab\s*/>", " ", xml)
    xml = re.sub(r"<w:br\s*/>", "\n", xml)
    xml = re.sub(r"</w:p>", "\n", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))

def read_text(data):
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            t = data.decode(enc)
            if text_quality(t) >= 0.8:
                return t
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")

def sniff_type(filename, data):
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml")
            return "docx"
        except Exception:
            return "unsupported"
    if (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:2] == b"BM"
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
            or (data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"mif1", b"hevc", b"heim", b"msf1"))):
        return "image"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc"                                      # old binary Word format
    return "text"

def read_resume_file(filename, data):
    """Returns (text, method). method is pypdf, pymupdf, docx, text, ocr or failed."""
    kind = sniff_type(filename, data)
    if kind == "pdf":
        return read_pdf(data)
    if kind == "docx":
        return read_docx(data), "docx"
    if kind == "image":
        try:
            from PIL import Image
            import pytesseract  # noqa: F401
        except ImportError:
            print("Image reading needs the tools from Cell 1 (tesseract-ocr, pytesseract, pillow). Run Cell 1 again.")
            return "", "failed"
        try:
            return ocr_image(Image.open(io.BytesIO(data))), "ocr"
        except Exception as e:
            print(f"Could not open this image ({e}). If it is an iPhone HEIC photo, share it as JPG and try again.")
            return "", "failed"
    if kind == "doc":
        print(f"{filename} is an old .doc file. Open it in Word or Google Docs and save it as .docx or PDF.")
        return "", "failed"
    if kind == "unsupported":
        print(f"{filename} is not a supported file type.")
        return "", "failed"
    return read_text(data), "text"

def ring_svg(score):
    c = 339.3
    color = "#2E9E5B" if score >= 70 else "#E0A800" if score >= 45 else "#C0392B"
    return (f"<svg width='140' height='140' viewBox='0 0 130 130'><circle cx='65' cy='65' r='54' fill='none' "
            f"stroke='#E7ECF8' stroke-width='12'/><circle cx='65' cy='65' r='54' fill='none' stroke='{color}' "
            f"stroke-width='12' stroke-linecap='round' stroke-dasharray='{c}' stroke-dashoffset='{c*(1-score/100)}' "
            f"transform='rotate(-90 65 65)'/><text x='65' y='75' text-anchor='middle' font-size='32' "
            f"font-weight='800' fill='#14213D'>{score}</text></svg>")

def show_rating(name, r, jd=None):
    top = (f"<div style='display:flex;align-items:center;gap:22px;flex-wrap:wrap'>{ring_svg(r['score'])}<div>"
           f"<div style='font-size:24px;font-weight:800'>{html.escape(name)}</div>"
           f"<div style='font-size:20px;font-weight:700;margin-top:2px'>{r['tier']} resume ({r['score']}/100)</div>"
           f"<div style='color:#6B7596;font-size:13px;margin-top:4px'>Rated on a 100-point rubric, no job description needed</div>"
           f"</div></div>")
    bars = ""
    for label, pts, mx, note in r["breakdown"]:
        pct = 100 * pts / mx
        bars += (f"<div style='margin:8px 0'><div style='display:flex;justify-content:space-between'>"
                 f"<b>{label}</b><span>{round(pts, 1):g} / {mx}</span></div>"
                 f"<div style='background:#E7ECF8;border-radius:6px;height:9px'><div style='background:#14213D;"
                 f"width:{pct:.0f}%;height:9px;border-radius:6px'></div></div>"
                 f"<div style='color:#6B7596;font-size:13px'>{html.escape(note)}</div></div>")
    role_rows = ""
    for ro in r["roles"]:
        rc = "#2E9E5B" if ro["score"] >= 65 else "#E0A800" if ro["score"] >= 35 else "#8A93AD"
        role_rows += (f"<div style='padding:9px 0;border-bottom:1px solid #EDF0F7'>"
                      f"<b style='font-size:16px'>{html.escape(ro['role'])}</b> "
                      f"<span style='background:{rc};color:#fff;border-radius:8px;padding:2px 9px;font-size:13px'>"
                      f"{ro['score']}% - {ro['label']}</span><br>"
                      f"<span style='color:#4A5578'>You have:</span> {html.escape(', '.join(ro['have']) or '-')}<br>"
                      f"<span style='color:#C0392B'>Build next:</span> {html.escape(', '.join(ro['need']) or 'nothing core, you are covered')}</div>")
    if r["no_skills"]:
        role_rows = "No tech skills were detected, so start with the basics below."
    e, x = r["edu"], r["exp"]
    found = ("<b>Degrees:</b> " + html.escape(", ".join(e["degrees"]) or "-") +
             "<br><b>Institutions:</b> " + html.escape("; ".join(e["institutions"]) or "-") +
             "<br><b>CGPA:</b> " + html.escape(e["cgpa"] or "-") +
             "<br><b>Roles:</b> " + html.escape("; ".join(x["roles"]) or "-") +
             "<br><b>Organizations (spaCy):</b> " + html.escape(", ".join(x["orgs"]) or "-") +
             "<br><b>Dates (spaCy):</b> " + html.escape(", ".join(x["dates"]) or "-"))
    tips = "".join(f"<div style='padding:5px 0;border-bottom:1px solid #EDF0F7'>- {html.escape(t)}</div>" for t in r["tips"]) \
           or "Structure looks good."
    warn = _card("Parsing warnings", "".join(f"<div style='color:#C0392B'>Warning: {html.escape(w)}</div>" for w in r["warnings"])) \
           if r["warnings"] else ""
    certs_html = "".join(f"<div style='padding:4px 0'>- {html.escape(c)} <span style='color:#6B7596;font-size:12px'>({html.escape(w)})</span></div>" for c, w in r["cert_where"]) \
                 or "<i style='color:#6B7596'>No certification courses found</i>"
    jd_card = ""
    if jd is not None:
        jd_card = _card(f"Match with your job description: {jd['score']}/100 ({jd['verdict']})",
                        "<b>Matched:</b> " + _chips(jd["matched"], "#FFE45E", "#14213D") +
                        "<br><b>Missing:</b> " + _chips(jd["missing"], "#fff", "#C0392B", "1.5px dashed #C0392B"))
    page = (f"<div style='background:#F4F6FB;color:#14213D;padding:16px;border-radius:18px;font-family:sans-serif'>"
            f"{_card('Resume rating', top)}{warn}{_card('How the score was built', bars)}{jd_card}"
            f"{_card('Skills you proved in your work', _chips(r['proved'], '#FFE45E', '#14213D'))}"
            f"{_card('Skills only listed (count 20%)', _chips(r['listed'], '#E7ECF8', '#4A5578'))}"
            f"{_card('Jobs you can aim for', role_rows)}"
            f"{_card('Skills to build next for tech jobs', _chips(r['build'], '#E4F6EA', '#1F7A44'))}"
            f"{_card('Certifications and courses found (out of 10 points)', certs_html)}"
            f"{_card('What was found in your resume', found)}{_card('Improvements', tips)}</div>")
    return page

