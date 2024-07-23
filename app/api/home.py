from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="templates")

# Dummy datastore
items = ["Item 1", "Item 2"]

@router.get("/", response_class=HTMLResponse)
def get_items(request: Request):
    return templates.TemplateResponse("items.html", {"request": request, "items": items})