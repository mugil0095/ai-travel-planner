import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Tried in order when the primary model is overloaded (503) or rate limited (429). Comma-separated.
# Free-tier quotas are per model per day, so the chain also spreads usage across quotas.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
).split(",") if m.strip()]
DB_PATH = os.getenv("DB_PATH", "data/checkpoints.sqlite")
# Human reject/modify rounds allowed per plan before only "approve" is accepted.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "5"))

# OpenStreetMap services require an identifying User-Agent.
USER_AGENT = "ai-travel-planner/1.0 (take-home project)"
