from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lora.inference import LoraInference
from lora.prompt_builder import build_recommendation_prompt
from rag.retriever import retrieve

app = FastAPI(
    title="Fitness Home RAG + Llama API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LoraInference()


class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def home():
    return {
        "message": "Fitness Home Backend Running"
    }


def build_safe_answer(
    selected_restaurant: dict,
    raw_answer: str,
) -> str:
    restaurant_name = selected_restaurant.get("restaurant_name", "")
    calories = selected_restaurant.get("average_calories", "")
    protein = selected_restaurant.get("average_protein", "")
    fiber = selected_restaurant.get("average_fiber", "")

    reason = raw_answer.strip()

    if "Reason:" in reason:
        reason = reason.split("Reason:", 1)[1].strip()

    if "Recommended Restaurant:" in reason:
        reason = reason.split("Recommended Restaurant:", 1)[0].strip()

    if len(reason) < 20:
        reason = (
            "This restaurant is recommended because it matches your nutrition "
            "request based on the retrieved evidence."
        )

    return (
        f"Recommended Restaurant:\n"
        f"{restaurant_name}\n\n"
        f"Nutrition Summary:\n"
        f"{calories} kcal\n"
        f"{protein} g protein\n"
        f"{fiber} g fiber\n\n"
        f"Recommended Reason:\n"
        f"{reason}"
    )


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    retrieved_results = retrieve(
        query=request.query,
        top_k=request.top_k,
    )

    if len(retrieved_results) == 0:
        return {
            "query": request.query,
            "retrieved_count": 0,
            "retrieved_results": [],
            "generated_answer": "No suitable restaurant was found.",
        }

    prompt = build_recommendation_prompt(
        user_query=request.query,
        retrieved_results=retrieved_results,
    )

    raw_answer = llm.generate(prompt)

    generated_answer = build_safe_answer(
        selected_restaurant=retrieved_results[0],
        raw_answer=raw_answer,
    )

    return {
        "query": request.query,
        "retrieved_count": len(retrieved_results),
        "retrieved_results": retrieved_results,
        "generated_answer": generated_answer,
    }