from fastapi import FastAPI

import uvicorn

from app.api import home
from app.api import item
from app.api import ml

app = FastAPI(
    title="ML + Htmx",
    description="ML + Htmx"
)

app.include_router(home.router)
app.include_router(item.router, prefix="/item", tags=["item"])
app.include_router(ml.router, prefix="/ml", tags=["ML"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001)