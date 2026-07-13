import os
from pathlib import Path
from dotenv import load_dotenv

# ===========================
# Project Root Directory
# ===========================

PROJECT_ROOT = Path(__file__).resolve().parent

# ===========================
# Data Folders
# ===========================

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ===========================
# Document Categories
# ===========================

DOCUMENT_TYPES = [
    "acts",
    "judgments",
    "tax",
    "pov"
]

# ===========================
# OCR Configuration
# ===========================

OCR_TEXT_THRESHOLD = 25

# ===========================
# Output Files
# ===========================

PAGES_JSON = PROCESSED_DATA_DIR / "pages.json"

CLEAN_PAGES_JSON = PROCESSED_DATA_DIR / "clean_pages.json"

CHUNKS_JSON = PROCESSED_DATA_DIR / "chunks.json"

CHUNKS_WITH_METADATA_JSON = PROCESSED_DATA_DIR / "chunks_with_metadata.json"

# --------------------------------------------------
# Chunking Configuration
# --------------------------------------------------

CHUNK_SIZE = 400

CHUNK_OVERLAP = 80

MIN_TRAILING_CHUNK_SIZE = 50

# ===========================
# Graph Configuration
# ===========================

GRAPH_DIR = PROCESSED_DATA_DIR / "graph"

RELATIONSHIPS_JSON = GRAPH_DIR / "relationships.json"

OKF_DIR = PROJECT_ROOT / "knowledge"

NEO4J_URI = os.getenv("NEO4J_URI")

NEO4J_USER = os.getenv("NEO4J_USER")

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ===========================
# Embedding Configuration
# ===========================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIM = 384

CHUNKS_WITH_EMBEDDINGS_JSON = PROCESSED_DATA_DIR / "chunks_with_embeddings.json"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ===========================
# Elasticsearch Configuration
# ===========================
load_dotenv() 

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")

ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

ELASTIC_INDEX_NAME = "legal_rag_chunks"

# ===========================
# Reranking Configuration
# ===========================

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_TOP_K = 4

# ===========================
# LLM Configuration
# ===========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

LLM_CONTEXT_TOKEN_BUDGET = 6500

LLM_RESERVED_TOKEN_BUDGET = 1500

# ===========================
# Evaluation Configuration
# ===========================

GOLDEN_SET_CSV = PROJECT_ROOT / "evaluation" / "golden_set.csv"

EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


#===========================
# Promting Configuration
#===========================
SYSTEM_PROMPT_FILE = PROJECT_ROOT / "rag" / "system_prompt.txt"

MAX_CONTEXT_CHARACTERS = 40000