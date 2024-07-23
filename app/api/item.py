from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="templates")

# Dummy datastore
items = ["Item 1", "Item 2"]

@router.post("/add-item")
def add_item(request: Request, item: str = Form(...)):
    items.append(item)
    return templates.TemplateResponse("partials/item.html", {"request": request, "item": item})