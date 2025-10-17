from fastapi import FastAPI, Request, Depends, HTTPException, status, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Any, Dict, List
import os, json, re, time, hmac, hashlib, base64

# =========================
# HF Transformers / Torch
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

MODEL_NAME = os.getenv("CURRICULUM_MODEL", "google/gemma-2b-it")
# Lower, but still overrideable via env
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "220"))
USE_4BIT = os.getenv("USE_4BIT", "0") == "1" and HAVE_BNB
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"  # <- NEW: skip heavy generations, use deterministic fallbacks

# CUDA perf boosts (no feature change)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

app = FastAPI(title="Curriculum Agent", version="1.3")

# ---------------------------
# Load model (GPU if available, optional 4-bit)
# ---------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

if USE_4BIT:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
else:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.bfloat16 if (_device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if _device == "cuda" else torch.float32
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=_dtype,
    )
    if _device == "cuda":
        mdl.to("cuda")

# Deterministic, low-latency defaults; we still set do_sample=False per call
gen = pipeline(
    "text-generation",
    model=mdl,
    tokenizer=tok,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    temperature=0.0,
    do_sample=False,
    return_full_text=False,
    device=0 if torch.cuda.is_available() else -1,
)

# Warmup to avoid first request spike
@app.on_event("startup")
def warmup():
    try:
        with torch.inference_mode():
            _ = gen("Hello", max_new_tokens=4, do_sample=False, temperature=0.0, truncation=True)
    except Exception:
        pass

# ---------------------------
# Security (API Key for Swagger Authorize)
# ---------------------------

API_KEY_NAME = "X-API-Key"
API_KEY_ENV = "CURRICULUM_API_KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        # Dev mode: allow all if no key set
        return None
    if api_key == expected:
        return api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

# ---------------------------
# JWT utilities (manual HS256)
# ---------------------------

JWT_SECRET_ENV = "CURRICULUM_JWT_SECRET"
JWT_DEFAULT_SECRET = "dev-jwt-secret-CHANGE-ME"
JWT_ALG = "HS256"
bearer_scheme = HTTPBearer(auto_error=False)

def _now_ts() -> int:
    return int(time.time())

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _b64url_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))

def _hmac_sha256(data: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, data, hashlib.sha256).digest()

def create_jwt(username: str, ttl_hours: int = 24) -> Dict[str, Any]:
    iat = _now_ts()
    exp = iat + max(1, ttl_hours) * 3600
    header = {"alg": JWT_ALG, "typ": "JWT"}
    payload = {"sub": username, "iat": iat, "exp": exp}
    header_b = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    secret = os.getenv(JWT_SECRET_ENV, JWT_DEFAULT_SECRET).encode("utf-8")
    signing_input = _b64url(header_b) + "." + _b64url(payload_b)
    sig = _b64url(_hmac_sha256(signing_input.encode("ascii"), secret))
    token = signing_input + "." + sig
    return {"access_token": token, "token_type": "bearer", "username": username, "issued_at": iat, "expires_at": exp}

def verify_jwt(token: str) -> Dict[str, Any] | None:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".", 3)
        header = json.loads(_b64url_dec(header_b64))
        if header.get("alg") != JWT_ALG:
            return None
        payload_bytes = _b64url_dec(payload_b64)
        payload = json.loads(payload_bytes)
        secret = os.getenv(JWT_SECRET_ENV, JWT_DEFAULT_SECRET).encode("utf-8")
        signing_input = header_b64 + "." + payload_b64
        sig_expected = _hmac_sha256(signing_input.encode("ascii"), secret)
        sig_given = _b64url_dec(sig_b64)
        if not hmac.compare_digest(sig_expected, sig_given):
            return None
        now = _now_ts()
        if isinstance(payload, dict) and isinstance(payload.get("exp"), int) and now > payload["exp"]:
            return None
        return payload
    except Exception:
        return None

def get_jwt_identity(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    claims = verify_jwt(credentials.credentials)
    if not claims:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return {"auth": "jwt", "user": claims.get("sub", "unknown"), "claims": claims}

def get_identity_combined(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Depends(api_key_header),
):
    # Prefer JWT
    if credentials and credentials.credentials:
        claims = verify_jwt(credentials.credentials)
        if claims:
            return {"auth": "jwt", "user": claims.get("sub", "unknown"), "claims": claims}
    # Fallback: API key
    expected = os.getenv(API_KEY_ENV)
    if expected and api_key == expected:
        return {"auth": "api_key", "user": "env-key", "claims": None}
    # Dev-open mode if nothing set
    if not os.getenv(JWT_SECRET_ENV) and not expected:
        return {"auth": "dev-open", "user": "anonymous", "claims": None}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

# ---------------------------
# Helpers
# ---------------------------

def extract_json(text: str):
    """
    Robustly extract the FIRST balanced JSON object or array from text.
    """
    if not isinstance(text, str):
        return text

    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            return candidate

    s = text
    start = None
    stack = []
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if start is None:
            if ch in "{[":
                start = i
                stack = [ch]
        else:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        break
                    opench = stack.pop()
                    if (opench == "{" and ch != "}") or (opench == "[" and ch != "]"):
                        return text.strip()
                    if not stack:
                        candidate = s[start:i+1]
                        candidate = candidate.strip()
                        try:
                            return json.loads(candidate)
                        except Exception:
                            return candidate
    return text.strip()

def _coerce_list_of_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x if isinstance(x, str) else str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                loaded = json.loads(s)
                if isinstance(loaded, list):
                    return [x if isinstance(x, str) else str(x) for x in loaded]
            except Exception:
                pass
        return [p.strip() for p in re.split(r"[\n,]", s) if p.strip()]
    return [str(value)]

def _coerce_modules(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value, start=1):
            out[f"Module {i}"] = item
        return out
    if isinstance(value, str):
        s = value.strip()
        try:
            loaded = json.loads(s)
            return _coerce_modules(loaded)
        except Exception:
            return {"Module 1": {"raw": s}}
    return {"Module 1": {"raw": value}}

def _extract_topics_from_markdown(desc: str) -> List[str]:
    if not isinstance(desc, str):
        return []
    bullets = []
    for line in desc.splitlines():
        m = re.match(r"\s*[*-]\s+(.*)", line.strip())
        if m:
            t = m.group(1).strip()
            if not re.match(r"(?i)topics?:\s*$", t):
                bullets.append(t)
    return bullets

def _extract_hours_from_text(text: str, default: int = 2) -> int:
    m = re.search(r"(?i)Estimated\s*Hours?\s*:\s*(\d+)", text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return default

def _normalize_questions(items: Any, want=5) -> List[str]:
    def clean(s: str) -> str:
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(s).strip(), flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        if not s or s in ("N/A","null","None","[]","{}"):
            return ""
        words = s.split()
        if len(words) > 25:
            s = " ".join(words[:25])
        return s
    arr: List[str] = []
    if isinstance(items, list):
        arr = [clean(x) for x in items if clean(x)]
    elif isinstance(items, str):
        try:
            j = json.loads(items)
            if isinstance(j, list):
                arr = [clean(x) for x in j if clean(x)]
            else:
                arr = [clean(items)]
        except Exception:
            arr = [clean(x) for x in items.splitlines() if clean(x)]
    elif isinstance(items, dict):
        for k in ("questions","assessments","items"):
            if isinstance(items.get(k), list):
                arr = [clean(x) for x in items[k] if clean(x)]
                break
        if not arr:
            arr = [clean(json.dumps(items, ensure_ascii=False))]
    arr = [x for x in arr if x][:want]
    while len(arr) < want:
        arr.append("Apply the module concepts to a realistic scenario and justify your approach and assumptions.")
    return arr

def _extract_first_array(text: str) -> List[Any] | None:
    if not isinstance(text, str):
        return None
    m = re.search(r"\[(?:.|\n)*\]", text)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except Exception:
        return None

def _stringify_question(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    q = item.get("question") or item.get("q") or ""
    typ = item.get("type") or item.get("kind") or ""
    opts = item.get("options") or item.get("choices") or []
    ans = item.get("correctAnswer") or item.get("answer") or None

    parts = []
    if q:
        parts.append(q.strip())
    if typ:
        parts.append(f"({typ})")
    if isinstance(opts, list) and opts:
        parts.append("Options: " + ", ".join(map(str, opts)))
    if ans:
        parts.append(f"Answer: {ans}")
    return " ".join(parts) if parts else json.dumps(item, ensure_ascii=False)

def normalize_assessments_output(x: Any) -> List[str]:
    """
    Always return exactly 5 strings (never raise).
    """
    def _clean(s: str) -> str:
        s = str(s).strip()
        if not s or s in ("N/A", "null", "None", "[]", "{}"):
            return ""
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)  # strip fences
        s = re.sub(r"\s+", " ", s)
        if s.startswith("{") or s.startswith("["):
            return ""
        words = s.split()
        if len(words) > 25:
            s = " ".join(words[:25])
        return s

    items: List[str] = []

    if isinstance(x, list):
        items = [i if isinstance(i, str) else _stringify_question(i) for i in x]
    elif isinstance(x, dict):
        arr = None
        for k in ("assessments", "questions", "items"):
            if isinstance(x.get(k), list):
                arr = x[k]; break
        items = [i if isinstance(i, str) else _stringify_question(i) for i in (arr if arr is not None else [x])]
    elif isinstance(x, str):
        try:
            j = json.loads(x)
            return normalize_assessments_output(j)
        except Exception:
            arr = _extract_first_array(x)
            if arr is not None:
                items = [i if isinstance(i, str) else _stringify_question(i) for i in arr]
            else:
                items = [p.strip() for p in x.splitlines() if p.strip()]
    else:
        items = [str(x)]

    cleaned = [c for c in (_clean(s) for s in items) if c]

    if len(cleaned) < 5:
        salvage: List[str] = []
        for s in cleaned:
            salvage.extend([t.strip() for t in re.split(r"[.?!]\s+", s) if t.strip()])
        cleaned = [c for c in salvage if c] or cleaned

    cleaned = cleaned[:5]
    while len(cleaned) < 5:
        cleaned.append("Explain one key concept from the course using a concrete, real-world example.")
    return cleaned

# ---------------------------
# NEW: Planner/Expander prompts + hours heuristic + SAFE FALLBACKS
# ---------------------------

PLANNER_PROMPT_TPL = """You are a curriculum planner.
Goal: produce EXACTLY 5 coherent modules for the course "{course_title}".
Audience level: {level}

Grounding (safe summary):
{grounding}

Rules:
- Output STRICT JSON: {{"modules":[{{"module_no":1,"title":"..." ,"outcomes":["...","...","..."]}}, ... up to 5 ]]}}
- outcomes must be measurable and specific to the module (2–4 per module).
- No descriptions, topics, or assessments here. STRICT JSON only.
"""

EXPANDER_PROMPT_TPL = """You are a curriculum expander.
Audience level: {level}

Grounding (safe summary):
{grounding}

Module shell (JSON):
{module_shell}

Add these fields:
- "description": 2–3 sentences, natural and grounded
- "topics": 4–6 bullet items
- "estimated_hours_hint": one of ["very_light","light","medium","deep"]
- "assessments": 3–4 strings, each 18–26 words, each clearly referencing one module outcome by text in brackets like [Outcome: <outcome text>].

Output STRICT JSON with exactly these keys.
"""

def _hours_from_hint(hint: str) -> int:
    table = {"very_light": 2, "light": 3, "medium": 4, "deep": 5}
    return table.get((hint or "").strip().lower(), 3)

def _gen_json(prompt: str, max_new_tokens=400) -> Any:
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, truncation=True)[0]["generated_text"]
    parsed = extract_json(out)
    if isinstance(parsed, (dict, list)):
        return parsed
    # retry: same prompt + explicit JSON reminder
    out2 = gen(
        prompt + "\n\nReturn ONLY valid JSON matching the structure described. No prose or markdown.",
        max_new_tokens=min(220, max_new_tokens),
        do_sample=False,
        temperature=0.0,
        truncation=True
    )[0]["generated_text"]
    parsed2 = extract_json(out2)
    return parsed2 if isinstance(parsed2, (dict, list)) else {}

def _fallback_shells(course_title: str, level: str, grounding: str, learning_outcomes: List[str]) -> List[Dict[str, Any]]:
    base = (course_title or "Course").strip()
    titles = [
        f"Foundations of {base}",
        f"Working with Data for {base}",
        f"Exploratory Analysis in {base}",
        f"Modeling Basics for {base}",
        f"Evaluation and Communication in {base}",
    ]
    los = [o for o in (learning_outcomes or []) if o] or ["Identify key concepts", "Apply basics to simple tasks", "Interpret simple results"]
    chunks = [los[i:i+3] for i in range(0, min(len(los), 15), 3)]
    while len(chunks) < 5:
        chunks.append(["Apply concepts in a small scenario", "Interpret results correctly"])
    shells=[]
    for i in range(5):
        shells.append({"module_no": i+1, "title": titles[i], "outcomes": chunks[i][:3]})
    return shells

def _fallback_expand(shell: Dict[str, Any]) -> Dict[str, Any]:
    title = shell.get("title","Module")
    outs = _coerce_list_of_str(shell.get("outcomes", []))[:3]
    desc = f"{title} introduces core ideas with realistic examples. Learners practice key skills through guided activities and short reflections to reinforce understanding."
    topics = ["Key terminology and concepts","Typical workflow and steps","Hands-on activity with a small dataset","Common pitfalls and best practices","Short reflection on learning"]
    asses = [f"Using a small dataset, demonstrate the skill, explain decisions, and justify your approach. [Outcome: {oc}]" for oc in outs[:3]]
    if len(asses)<3:
        asses.append("Describe your approach to a simple scenario and explain trade-offs and next steps. [Outcome: apply basics]")
    return {"description": desc, "topics": topics, "estimated_hours_hint": "light", "assessments": asses}

# ---------------------------
# Models (Pydantic v2)
# ---------------------------

class Common(BaseModel):
    # Accept both "course_title" and "title"
    course_title: str = Field(alias="title")
    level: str = "Beginner"
    grounding: str = ""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )

class ModulesReq(Common):
    # Accept "learning_outcomes" and "outcomes"
    learning_outcomes: List[str] = Field(default_factory=list, alias="outcomes")

    @field_validator("learning_outcomes", mode="before")
    @classmethod
    def _normalize_outcomes_input(cls, v):
        return _coerce_list_of_str(v)

class AssessReq(Common):
    # Accept anything for "modules" and normalize
    modules: Dict[str, Any]

    @field_validator("modules", mode="before")
    @classmethod
    def _normalize_modules_input(cls, v):
        return _coerce_modules(v)

# ---------- Orchestrator request/response models ----------
class DesignReq(BaseModel):
    title: str = Field(..., description="Course title")
    level: str = Field("Beginner", description="Audience level")
    grounding: str = Field("", description="Optional context/notes")

class DesignResp(BaseModel):
    title: str
    outcomes: List[str]
    modules: Dict[str, Dict[str, Any]]
    assessments: List[str]

# ---------------------------
# Error diagnostics for 422
# ---------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation failed",
            "errors": exc.errors(),
            "received": body.decode("utf-8", errors="ignore"),
        },
    )

# ---------------------------
# Health
# ---------------------------

@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "fast_mode": FAST_MODE,
        "model_name": MODEL_NAME,
        "endpoints": [
            "/design_course",
            "/design_course_auth",
            "/outcomes",
            "/modules",
            "/assessments",
            "/auth/jwt",
            "/auth/whoami",
            "/docs",
        ],
    }

# ---------------------------
# Auth Endpoints (JWT)
# ---------------------------

class JWTCreateReq(BaseModel):
    username: str = Field(..., description="Username for the JWT 'sub' claim")
    ttl_hours: int = Field(24, ge=1, le=24*365, description="Token lifetime in hours (default 24)")

class JWTCreateResp(BaseModel):
    access_token: str
    token_type: str
    username: str
    issued_at: int
    expires_at: int

@app.post("/auth/jwt", response_model=JWTCreateResp, tags=["auth"])
def auth_jwt(req: JWTCreateReq):
    """Generate a JWT for Swagger 'Authorize' (Authorization: Bearer <token>)."""
    return create_jwt(req.username, req.ttl_hours)

@app.get("/auth/whoami", tags=["auth"])
def whoami(identity: Dict[str, Any] = Depends(get_identity_combined)):
    """Identity derived from Bearer JWT or X-API-Key (JWT preferred)."""
    return identity

# ---------------------------
# Core Routes
# ---------------------------

@app.post("/outcomes")
def outcomes(req: Common):
    prompt = f"""
You are a curriculum designer.
Course: {req.course_title}
Level: {req.level}
Grounding (summarized, redacted):
{req.grounding}

Task: Generate 5–6 clear, measurable learning outcomes using Bloom's verbs.
Constraints: No placeholders like [REDACTED] or <...>; use neutral, course-relevant wording.
Return ONLY valid JSON array.
"""
    out = gen(prompt, max_new_tokens=180, do_sample=False, temperature=0.0, truncation=True)[0]["generated_text"]
    raw = extract_json(out)
    return {"outcomes": _coerce_list_of_str(raw)[:6]}

@app.post("/modules")
def modules(req: ModulesReq):
    """
    Two-stage generation with graceful fallbacks (never 500).
    FAST_MODE: skip heavy planner/expander and return deterministic fallbacks quickly.
    """
    # FAST PATH: deterministic fallbacks (instant)
    if FAST_MODE:
        shells = _fallback_shells(req.course_title, req.level, req.grounding, req.learning_outcomes)[:5]
        modules_out: Dict[str, Dict[str, Any]] = {}
        for sh in shells:
            module_no = int(sh.get("module_no", 0)) or len(modules_out) + 1
            title = sh.get("title") or f"Module {module_no}"
            outcomes_list = _coerce_list_of_str(sh.get("outcomes", []))[:4]
            expanded = _fallback_expand({"title": title, "outcomes": outcomes_list})
            modules_out[f"Module {module_no}"] = {
                "Title": title,
                "Description": expanded["description"],
                "Topics": expanded["topics"],
                "Estimated Hours": _hours_from_hint(expanded["estimated_hours_hint"]),
                "Module Outcomes": outcomes_list,
                "Assessment Questions": _normalize_questions(expanded["assessments"], want=5),
            }
        keys = sorted(modules_out.keys(), key=lambda k: int(re.findall(r"\d+", k)[0]))
        modules_out = {f"Module {i+1}": modules_out[keys[i]] for i in range(5)}
        return {"modules": modules_out}

    # NORMAL PATH: Stage 1 planner
    try:
        planner_prompt = PLANNER_PROMPT_TPL.format(
            course_title=req.course_title,
            level=req.level,
            grounding=req.grounding or "Beginner scope"
        )
        plan = _gen_json(planner_prompt, max_new_tokens=280)
        shells = (plan.get("modules") if isinstance(plan, dict) else None) or []
    except Exception as e:
        print(f"[modules] planner exception: {e}")
        shells = []
    if not isinstance(shells, list) or len(shells) < 5:
        shells = _fallback_shells(req.course_title, req.level, req.grounding, req.learning_outcomes)
    shells = shells[:5]
    if len(shells) < 5:
        for i in range(len(shells)+1, 6):
            shells.append({"module_no": i, "title": f"Module {i}", "outcomes": ["Identify key concepts", "Apply basics to simple tasks"]})

    # Stage 2: expander
    modules_out: Dict[str, Dict[str, Any]] = {}
    for sh in shells:
        module_no = int(sh.get("module_no", 0)) or len(modules_out) + 1
        title = sh.get("title") or f"Module {module_no}"
        outcomes_list = _coerce_list_of_str(sh.get("outcomes", []))[:4]

        expanded = {}
        try:
            expander_prompt = EXPANDER_PROMPT_TPL.format(
                level=req.level,
                grounding=req.grounding or "Use safe beginner context relevant to the course title.",
                module_shell=json.dumps({"module_no": module_no, "title": title, "outcomes": outcomes_list}, ensure_ascii=False)
            )
            expanded = _gen_json(expander_prompt, max_new_tokens=360)
            if not isinstance(expanded, dict):
                expanded = {}
        except Exception as e:
            print(f"[modules] expander exception (module {module_no}): {e}")
            expanded = {}

        if not expanded or not all(k in expanded for k in ("description", "topics", "estimated_hours_hint", "assessments")):
            expanded = _fallback_expand({"title": title, "outcomes": outcomes_list})

        desc = expanded.get("description") or ""
        topics = expanded.get("topics") or []
        hours = _hours_from_hint(expanded.get("estimated_hours_hint") or "light")
        assessments_norm = _normalize_questions(expanded.get("assessments") or [], want=5)
        if outcomes_list and not any("[Outcome:" in q for q in assessments_norm):
            assessments_norm[0] = f"{assessments_norm[0]} [Outcome: {outcomes_list[0]}]"

        modules_out[f"Module {module_no}"] = {
            "Title": title,
            "Description": desc,
            "Topics": topics if isinstance(topics, list) else _coerce_list_of_str(topics),
            "Estimated Hours": int(hours),
            "Module Outcomes": outcomes_list,
            "Assessment Questions": assessments_norm
        }

    keys = sorted(modules_out.keys(), key=lambda k: int(re.findall(r"\d+", k)[0]))
    modules_out = {f"Module {i+1}": modules_out[keys[i]] for i in range(5)}
    return {"modules": modules_out}

@app.post("/assessments")
def assessments(req: AssessReq):
    prompt = f"""
You are an assessment designer.
Course: {req.course_title}
Modules: {req.modules}
Grounding: {req.grounding}

OUTPUT FORMAT (very strict):
Return EXACTLY a JSON array of 5 plain strings. Nothing before or after the array.
Rules:
- Each string is 15–25 words.
- No quotes inside the strings, no markdown, no numbering, no explanations.
- No code fences, no extra text outside the array.
"""
    out = gen(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0,
        truncation=True,
    )[0]["generated_text"]
    raw = extract_json(out)
    return {"assessments": normalize_assessments_output(raw)}

# ---------------------------
# Orchestrators (standalone use)
# ---------------------------

@app.post("/design_course", response_model=DesignResp, tags=["orchestrator"], dependencies=[Depends(get_api_key)])
def design_course(req: DesignReq):
    outcomes_prompt = f"""
You are a curriculum designer.
Course: {req.title}
Level: {req.level}
Grounding (summarized, redacted):
{req.grounding}

Task: Generate 5–6 clear, measurable learning outcomes using Bloom's verbs.
Constraints: No placeholders like [REDACTED] or <...>; use neutral, course-relevant wording.
Return ONLY valid JSON array.
"""
    outcomes_text = gen(outcomes_prompt, max_new_tokens=180, do_sample=False, temperature=0.0, truncation=True)[0]["generated_text"]
    outcomes_list = _coerce_list_of_str(extract_json(outcomes_text))[:6]

    modules_dict = modules(ModulesReq(title=req.title, level=req.level, grounding=req.grounding, outcomes=outcomes_list))["modules"]

    assessments_prompt = f"""
You are an assessment designer.
Course: {req.title}
Modules: {modules_dict}
Grounding: {req.grounding}

OUTPUT FORMAT (very strict):
Return EXACTLY a JSON array of 5 plain strings. Nothing before or after the array.
Rules:
- Each string is 15–25 words.
- No quotes inside the strings, no markdown, no numbering, no explanations.
- No code fences, no extra text outside the array.
"""
    assessments_text = gen(assessments_prompt, max_new_tokens=180, do_sample=False, temperature=0.0, truncation=True)[0]["generated_text"]
    assessments_list = normalize_assessments_output(extract_json(assessments_text))

    return {
        "title": req.title,
        "outcomes": outcomes_list,
        "modules": modules_dict,
        "assessments": assessments_list,
    }

@app.post("/design_course_auth", response_model=DesignResp, tags=["orchestrator"], dependencies=[Depends(get_identity_combined)])
def design_course_auth(req: DesignReq):
    return design_course(req)
