from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RESTAURANTS_FILE = DATA_DIR / "restaurants.csv"
MENUS_FILE = DATA_DIR / "restaurant-menus.csv"

CLEAN_RESTAURANTS_FILE = DATA_DIR / "restaurants_clean.csv"
CLEAN_MENUS_FILE = DATA_DIR / "menus_clean.csv"

DATABASE_FILE = DATA_DIR / "fitness_home_rag_v1.db"

DOCUMENT_FILE = DATA_DIR / "rag_documents.jsonl"

EMBEDDING_FILE = DATA_DIR / "embeddings.npy"

FAISS_INDEX_FILE = DATA_DIR / "faiss.index"

SUMMARY_FILE = DATA_DIR / "processing_summary.json"