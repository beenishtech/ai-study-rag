from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from api import process_and_store_pdfs, query_rag_and_grok

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request, "answer": None, "status": None}
    )

@app.post("/upload")
async def upload_pdfs(request: Request, files: list[UploadFile] = File(...)):
    try:
        num_files, num_chunks = process_and_store_pdfs(files)
        status_msg = f"Successfully processed {num_files} PDFs and saved {num_chunks} chunks to ChromaDB!"
    except Exception as e:
        status_msg = f"Error processing PDFs: {str(e)}"

    return templates.TemplateResponse(
        name="index.html",
        context={"request": request, "status": status_msg}
    )

@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    try:
        answer, context = query_rag_and_grok(question)
    except Exception as e:
        answer = f"Error getting answer: {str(e)}"
        context = ""

    return templates.TemplateResponse(
        name="index.html",
        context={
            "request": request, 
            "question": question,
            "answer": answer,
            "context": context
        }
    )