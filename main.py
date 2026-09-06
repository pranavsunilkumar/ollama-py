from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #supporting lib for fast api

from llm import create_llm
from rag import rag_chain, ingest_data

app = FastAPI(
    title="simple RAG Agent API",
    version="1.0.0",
    description="A simple RAG Agent API using fastapi and langchain"
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_methods=["*"],
    allow_origins=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict:
    return{"message":"welcome to the simple RAG Agent API"}

@app.get("/health")
def health_check():
    return {"status":"ok"}

data =set()

@app.post("/add-item")
def add_item(item:str):
    data.add(item)
    return{"message":f"Item '{item}' added succesfully"}

@app.get("/get-Items")
def get_items():
    return{"Items":list(data)}

@app.put("/update-item")
def update_item(old_item:str,new_item:str):
    if old_item in data:
        data.remove(old_item)
        data.add(new_item)
        return {"message": f"Item '{old_item}' updated to '{new_item}' successfully."}

@app.delete("/delete-item")
def delete_item(item:str):
    if item in data:
        data.remove(item)
        return {"message": f"Item '{item}' deleted successfully."}

    #LLM and RAG Routes

class QueryRequest(BaseModel):
    query: str

@app.post("/llm-query")
def llm_query(request: QueryRequest):
    llm = create_llm()
    response = llm.invoke(request.query)
    return {"response": response.content}

@app.post("/ask-rag")
def ask_rag(request:QueryRequest):
    answer=rag_chain(request.query)
    return{
        "question":request.query,
        "answer":answer
    }