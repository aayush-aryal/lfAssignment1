from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import initialize_agent
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()
agent=initialize_agent()
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://127.0.0.1:5500/",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    thread_id: str



@app.get("/")
def read_root():
    return {"message":"Hello World"}


@app.post("/query")
async def ask_agent(request:QueryRequest):
    response_text=""
    config = {"configurable": {"thread_id": request.thread_id}}
    result = agent.invoke(
    {
        "messages": [{"role": "user", "content":request.query }],
    }, # type: ignore
    config=config) # type: ignore
    return {"response":result["messages"][-1].content} # type: ignore
