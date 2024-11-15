from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get("/stocks", response_class=HTMLResponse)
def stocks(request: Request):
    return templates.TemplateResponse("stocks.html", {"request": request})

@router.get("/realestate", response_class=HTMLResponse)
def realestate(request: Request):
    return templates.TemplateResponse("realestate.html", {"request": request})

@router.get("/ai", response_class=HTMLResponse)
def ai(request: Request):
    return templates.TemplateResponse("ai.html", {"request": request})