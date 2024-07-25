from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import uvicorn

from app import router
from app.api import item
from app.api import ml

app = FastAPI(
    title="ML + Htmx",
    description="ML + Htmx"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router.router)
app.include_router(item.router, prefix="/item", tags=["item"])
app.include_router(ml.router, prefix="/ml", tags=["ML"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001)