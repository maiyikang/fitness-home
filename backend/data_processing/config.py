from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = PROJECT_ROOT / "data"

RAW_RESTAURANTS_FILE = DATA_FOLDER / "restaurants.csv"
RAW_MENUS_FILE = DATA_FOLDER / "restaurant-menus.csv"

CLEAN_RESTAURANTS_FILE = DATA_FOLDER / "restaurants_clean.csv"
CLEAN_MENUS_FILE = DATA_FOLDER / "menus_clean.csv"

DATABASE_FILE = DATA_FOLDER / "fitness_home_rag_v1.db"

SUMMARY_FILE = DATA_FOLDER / "processing_summary.json"


# ==========================================================
# Data Processing
# ==========================================================

MAX_RESTAURANTS = 5000

MAX_MENUS_PER_RESTAURANT = 30


# ==========================================================
# Nutrition
# ==========================================================

DEFAULT_MEAL_CALORIES = 650

LOW_CALORIE_THRESHOLD = 500

HIGH_CALORIE_THRESHOLD = 800

HIGH_PROTEIN_THRESHOLD = 30

HIGH_FIBER_THRESHOLD = 7


# ==========================================================
# Future RAG Settings
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

TOP_K = 5

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


# ==========================================================
# Future LoRA Settings
# ==========================================================

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4

EPOCHS = 3


# ==========================================================
# Evaluation
# ==========================================================

USE_CALORIE_MAE = True

USE_NUTRITION_MAE = True

USE_FORMAT_SUCCESS_RATE = True

USE_CONSTRAINT_SUCCESS_RATE = True

USE_RELEVANCE_SCORE = True