from fastapi import FastAPI
from endpoints import router
import uvicorn


app = FastAPI(
    title="Deep Learning API",
    description="API for Deep Learning Model Predictions",
    version="1.0.0",
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
