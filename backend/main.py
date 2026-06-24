from fastapi import FastAPI

app = FastAPI(
    title="Fitness Home API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Fitness Home Python Backend Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }