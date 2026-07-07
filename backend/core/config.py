# ==========================================================
# Embedding Configuration
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIMENSION = 384

NORMALIZE_EMBEDDINGS = True


# ==========================================================
# FAISS Configuration
# ==========================================================

TOP_K = 5

FAISS_INDEX_TYPE = "IndexFlatIP"


# ==========================================================
# RAG Configuration
# ==========================================================

MAX_RETRIEVAL_RESULTS = 5

SIMILARITY_THRESHOLD = 0.50


# ==========================================================
# LoRA Configuration
# ==========================================================

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05


# ==========================================================
# Generation Configuration
# ==========================================================

MAX_NEW_TOKENS = 512

TEMPERATURE = 0.3